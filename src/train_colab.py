"""
NeuroScan — Google Colab Training Notebook
==========================================
Copy each cell block into a separate Colab cell.
Runtime -> Change runtime type -> T4 GPU before running.
"""

# ══════════════════════════════════════════════════════════════
# CELL 1 — Mount Google Drive
# ══════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

# ══════════════════════════════════════════════════════════════
# CELL 2 — Verify GPU & TF version
# ══════════════════════════════════════════════════════════════
import tensorflow as tf
print("TF version  :", tf.__version__)
print("GPU devices :", tf.config.list_physical_devices('GPU'))
print("GPU name    :", tf.test.gpu_device_name())

# ══════════════════════════════════════════════════════════════
# CELL 3 — Unzip dataset (run only ONCE)
# ══════════════════════════════════════════════════════════════
import os, zipfile

DRIVE_ZIP   = '/content/drive/MyDrive/NeuroScan/dataset_cropped.zip'
EXTRACT_DIR = '/content/dataset_cropped'

if not os.path.exists(EXTRACT_DIR):
    print("Extracting dataset...")
    with zipfile.ZipFile(DRIVE_ZIP, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    print("Done!")
else:
    print("Dataset already extracted — skipping.")

# Verify class counts
for split in ['train', 'val', 'test']:
    path = os.path.join(EXTRACT_DIR, split)
    if os.path.exists(path):
        classes = [c for c in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, c))]
        for cls in classes:
            n = len(os.listdir(os.path.join(path, cls)))
            if n > 0:
                print(f"  {split:5s} | {cls:12s} | {n} images")

# ══════════════════════════════════════════════════════════════
# CELL 4 — FULL TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)
from sklearn.metrics import classification_report, confusion_matrix

# ── CONFIG ──────────────────────────────────────────────────
DATA_DIR    = '/content/dataset_cropped'
MODELS_DIR  = '/content/drive/MyDrive/NeuroScan/models'
RESULTS_DIR = '/content/drive/MyDrive/NeuroScan/results'

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
SEED        = 42

PHASE_A_EPOCHS = 20
PHASE_A_LR     = 1e-4
PHASE_B_EPOCHS = 10
PHASE_B_LR     = 1e-5
FINE_TUNE_AT   = 15

CLASSES   = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_CLASSES = len(CLASSES)

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── DATA GENERATORS ─────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale            = 1.0 / 255.0,
    rotation_range     = 20,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    horizontal_flip    = True,
    zoom_range         = 0.15,
    brightness_range   = [0.8, 1.2],
    shear_range        = 0.1,
    fill_mode          = 'nearest'
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size = IMG_SIZE, batch_size = BATCH_SIZE,
    class_mode  = 'categorical', classes = CLASSES,
    shuffle=True, seed=SEED
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size = IMG_SIZE, batch_size = BATCH_SIZE,
    class_mode  = 'categorical', classes = CLASSES, shuffle=False
)
test_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size = IMG_SIZE, batch_size = BATCH_SIZE,
    class_mode  = 'categorical', classes = CLASSES, shuffle=False
)
print(f"Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")

# ── MODEL ───────────────────────────────────────────────────
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

x      = base_model.output
x      = layers.GlobalAveragePooling2D(name='gap')(x)
x      = layers.Dense(256, activation='relu', name='dense_256')(x)
x      = layers.Dropout(0.5, name='dropout')(x)
x      = layers.BatchNormalization(name='batch_norm')(x)
output = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)
model  = Model(inputs=base_model.input, outputs=output)

trainable = sum(1 for l in model.layers if l.trainable)
total     = len(model.layers)
print(f"Model built: {trainable}/{total} trainable layers")
print(f"Trainable params: {sum(tf.size(v).numpy() for v in model.trainable_variables):,}")

# ── CALLBACKS HELPER ────────────────────────────────────────
def get_callbacks(phase_name):
    return [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath       = os.path.join(MODELS_DIR, f'best_model_{phase_name}.keras'),
            monitor        = 'val_accuracy',
            save_best_only = True, verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(RESULTS_DIR, f'log_{phase_name}.csv'), append=True)
    ]

# ── PHASE A: Train head only ─────────────────────────────────
print("\n" + "="*55)
print("  PHASE A — Training custom head (base frozen)")
print("="*55)
model.compile(
    optimizer = tf.keras.optimizers.Adam(PHASE_A_LR),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)
history_a = model.fit(
    train_gen,
    epochs          = PHASE_A_EPOCHS,
    validation_data = val_gen,
    callbacks       = get_callbacks('phaseA'),
    verbose=1
)

# ── PHASE B: Fine-tune top VGG16 layers ─────────────────────
print("\n" + "="*55)
print(f"  PHASE B — Fine-tuning (unfreezing from layer {FINE_TUNE_AT})")
print("="*55)
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer = tf.keras.optimizers.Adam(PHASE_B_LR),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)
history_b = model.fit(
    train_gen,
    initial_epoch   = len(history_a.history['loss']),
    epochs          = len(history_a.history['loss']) + PHASE_B_EPOCHS,
    validation_data = val_gen,
    callbacks       = get_callbacks('phaseB'),
    verbose=1
)

# ── SAVE FINAL MODEL ────────────────────────────────────────
final_path = os.path.join(MODELS_DIR, 'best_model.keras')
model.save(final_path)
print(f"\nModel saved -> {final_path}")

# ══════════════════════════════════════════════════════════════
# CELL 5 — Plot training curves
# ══════════════════════════════════════════════════════════════
def plot_history(ha, hb):
    acc_a  = ha.history['accuracy'];     val_acc_a  = ha.history['val_accuracy']
    loss_a = ha.history['loss'];         val_loss_a = ha.history['val_loss']
    acc_b  = hb.history['accuracy'];     val_acc_b  = hb.history['val_accuracy']
    loss_b = hb.history['loss'];         val_loss_b = hb.history['val_loss']

    acc_all  = acc_a  + acc_b;   val_acc_all  = val_acc_a  + val_acc_b
    loss_all = loss_a + loss_b;  val_loss_all = val_loss_a + val_loss_b
    epochs   = range(1, len(acc_all) + 1)
    phase_b_start = len(acc_a)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('NeuroScan — Training Curves (Phase A + B)', fontsize=14, fontweight='bold')

    for ax, metric, val_metric, ylabel in [
        (axes[0], acc_all,  val_acc_all,  'Accuracy'),
        (axes[1], loss_all, val_loss_all, 'Loss')
    ]:
        ax.plot(epochs, metric,     color='#00bfff', lw=2, label='Train')
        ax.plot(epochs, val_metric, color='#ff6b35', lw=2, label='Val')
        ax.axvline(phase_b_start, color='green', ls='--', lw=1.5, label='Fine-tune start')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(ylabel); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved -> {out}")

plot_history(history_a, history_b)

# ══════════════════════════════════════════════════════════════
# CELL 6 — Final evaluation + Confusion Matrix
# ══════════════════════════════════════════════════════════════
test_gen.reset()
preds  = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
ax.set_title('NeuroScan — Confusion Matrix', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.show()

acc = np.mean(y_pred == y_true) * 100
print(f"\nFinal Test Accuracy: {acc:.2f}%")
print(f"Model: {final_path}")
print(f"Results: {RESULTS_DIR}")
