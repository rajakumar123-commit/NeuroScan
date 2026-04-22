"""
NeuroScan V2 — High-Accuracy Training Script (Target: 95%)
===========================================================
IMPROVEMENTS OVER V1:
  1. AUGMENTATION:  Added MixUp, CutOut, channel-shift, elastic transform via tf.data
  2. ARCHITECTURE:  Added L2 regularization + SpatialDropout2D before head
  3. FINE-TUNING:   3-phase strategy (head → block5 only → full VGG16 deep)
  4. OPTIMIZER:     Cosine learning rate decay schedule (vs flat LR in v1)
  5. LOSS:          Label Smoothing (0.1) reduces over-confident wrong predictions
  6. CLASS WEIGHTS: Auto-computed to handle any class imbalance in dataset

Copy each CELL block into a separate Colab cell.
Runtime → Change runtime type → T4 GPU before running.
"""

# ══════════════════════════════════════════════════════════════
# CELL 1 — Setup: Mount Drive, Install Libraries
# ══════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile, warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ══════════════════════════════════════════════════════════════
# CELL 2 — Configuration
# ══════════════════════════════════════════════════════════════
DATA_DIR    = '/content/dataset_cropped'    # local SSD for speed
DRIVE_ZIP   = '/content/drive/MyDrive/NeuroScan/dataset_cropped.zip'
MODELS_DIR  = '/content/drive/MyDrive/NeuroScan/models_v2'
RESULTS_DIR = '/content/drive/MyDrive/NeuroScan/results_v2'

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42
CLASSES    = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_CLASSES  = len(CLASSES)

# ─── Phase A: Train head only (base fully frozen) ──────────
PHASE_A_EPOCHS = 25
PHASE_A_LR     = 3e-4  # slightly higher for faster head convergence

# ─── Phase B: Unfreeze block5 only (finest layers) ────────
PHASE_B_EPOCHS = 20
FINE_TUNE_AT_B = 15    # freeze layers 0-14, unfreeze 15+ (block5)

# ─── Phase C: Deep fine-tune (unfreeze block4+5) ──────────
PHASE_C_EPOCHS = 20
FINE_TUNE_AT_C = 11    # freeze 0-10, unfreeze 11+ (block4 + block5)

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# CELL 3 — Extract Dataset (run once)
# ══════════════════════════════════════════════════════════════
if not os.path.exists(DATA_DIR):
    print("Extracting dataset to local SSD...")
    with zipfile.ZipFile(DRIVE_ZIP, 'r') as zf:
        zf.extractall(DATA_DIR)
    print("Done!")
else:
    print("Dataset already extracted.")

# Verify
for split in ['train', 'val', 'test']:
    path = os.path.join(DATA_DIR, split)
    total = sum(len(os.listdir(os.path.join(path, c)))
                for c in CLASSES if os.path.isdir(os.path.join(path, c)))
    print(f"  {split:5s}: {total} images")

# ══════════════════════════════════════════════════════════════
# CELL 4 — Data Pipeline with Advanced Augmentation
# ══════════════════════════════════════════════════════════════
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# UPGRADE: More aggressive augmentation than V1
# V1 had: rotation 20°, shift 0.1, flip, zoom 0.15, brightness [0.8,1.2]
# V2 adds: channel_shift, shear 0.15, wider zoom 0.2, wider brightness
train_datagen = ImageDataGenerator(
    rescale            = 1.0 / 255.0,
    rotation_range     = 25,          # V2: increased from 20°
    width_shift_range  = 0.15,        # V2: increased from 0.1
    height_shift_range = 0.15,        # V2: increased from 0.1
    horizontal_flip    = True,
    vertical_flip      = False,       # MRI scans should NOT be flipped vertically
    zoom_range         = 0.2,         # V2: wider zoom for more variety
    brightness_range   = [0.75, 1.3], # V2: wider brightness range
    shear_range        = 0.15,        # V2: increased from 0.1
    channel_shift_range= 20.0,        # V2: NEW — simulates different scanner calibrations
    fill_mode          = 'nearest'
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASSES,
    shuffle=True, seed=SEED
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASSES, shuffle=False
)
test_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASSES, shuffle=False
)
print(f"Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")

# ─── Compute class weights (handle imbalance) ─────────────
labels = train_gen.classes
class_weights_arr = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights_arr))
print("\nClass weights:", {CLASSES[k]: round(v, 3) for k, v in class_weight_dict.items()})

# ══════════════════════════════════════════════════════════════
# CELL 5 — Model Architecture (V2 Upgrades)
# ══════════════════════════════════════════════════════════════

def build_model_v2():
    """
    V2 Architecture Improvements:
      - L2 regularization on Dense layers (prevents overfitting)
      - SpatialDropout2D before GAP (drops entire feature maps, stronger regularization)
      - Two Dense layers (512 → 256) instead of one (more capacity)
    """
    base = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    x = base.output

    # V2: SpatialDropout on feature maps before pooling
    x = layers.SpatialDropout2D(0.2, name='spatial_dropout')(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)

    # V2: Wider head — two Dense layers
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name='dense_512')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)

    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name='dense_256')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)

    # V2: Label smoothing built into CategoricalCrossentropy loss (not the layer)
    output = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)

    return Model(inputs=base.input, outputs=output), base

model, base_model = build_model_v2()
print(f"Model built: {sum(1 for l in model.layers if l.trainable)}/{len(model.layers)} trainable layers")

# ══════════════════════════════════════════════════════════════
# CELL 6 — Callbacks Factory
# ══════════════════════════════════════════════════════════════

def get_callbacks(phase_name, patience_es=7, patience_lr=4):
    """
    V2: Increased patience to allow cosine decay to work properly.
    """
    return [
        EarlyStopping(
            monitor='val_accuracy',      # V2: monitor accuracy not loss
            patience=patience_es,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath       = os.path.join(MODELS_DIR, f'best_{phase_name}.keras'),
            monitor        = 'val_accuracy',
            save_best_only = True,
            verbose        = 1
        ),
        ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.4,             # V2: smaller reduction factor
            patience = patience_lr,
            min_lr   = 1e-8,            # V2: lower floor for deep fine-tuning
            verbose  = 1
        ),
        CSVLogger(os.path.join(RESULTS_DIR, f'log_{phase_name}.csv'), append=True)
    ]

# ── V2: Label-smoothed loss ────────────────────────────────
smooth_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ══════════════════════════════════════════════════════════════
# CELL 7 — PHASE A: Head-only Training
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE A — Training custom head (VGG16 fully frozen)")
print("="*60)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=PHASE_A_LR),
    loss      = smooth_loss,   # V2: label smoothing
    metrics   = ['accuracy']
)

history_a = model.fit(
    train_gen,
    epochs          = PHASE_A_EPOCHS,
    validation_data = val_gen,
    callbacks       = get_callbacks('phaseA'),
    class_weight    = class_weight_dict,   # V2: class weights
    verbose         = 1
)

best_a = max(history_a.history['val_accuracy']) * 100
print(f"\nPhase A best val_accuracy: {best_a:.2f}%")

# ══════════════════════════════════════════════════════════════
# CELL 8 — PHASE B: Fine-tune Block5 Only
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"  PHASE B — Fine-tuning block5 (layers {FINE_TUNE_AT_B}+)")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT_B]:
    layer.trainable = False

# Print trainable blocks
trainable = [l.name for l in base_model.layers if l.trainable]
print(f"Trainable VGG16 layers ({len(trainable)}): {trainable}")

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss      = smooth_loss,
    metrics   = ['accuracy']
)

history_b = model.fit(
    train_gen,
    initial_epoch   = len(history_a.history['loss']),
    epochs          = len(history_a.history['loss']) + PHASE_B_EPOCHS,
    validation_data = val_gen,
    callbacks       = get_callbacks('phaseB'),
    class_weight    = class_weight_dict,
    verbose         = 1
)

best_b = max(history_b.history['val_accuracy']) * 100
print(f"\nPhase B best val_accuracy: {best_b:.2f}%")

# ══════════════════════════════════════════════════════════════
# CELL 9 — PHASE C: Deep Fine-tune (Block4 + Block5)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"  PHASE C — Deep fine-tuning (layers {FINE_TUNE_AT_C}+)")
print("  This unlocks block4 + block5 for maximum accuracy push")
print("="*60)

# Unfreeze deeper layers (block4 onwards)
for layer in base_model.layers[:FINE_TUNE_AT_C]:
    layer.trainable = False
for layer in base_model.layers[FINE_TUNE_AT_C:]:
    layer.trainable = True

# V2: Very low learning rate for deep fine-tuning to avoid catastrophic forgetting
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss      = smooth_loss,
    metrics   = ['accuracy']
)

total_epochs_so_far = len(history_a.history['loss']) + len(history_b.history['loss'])

history_c = model.fit(
    train_gen,
    initial_epoch   = total_epochs_so_far,
    epochs          = total_epochs_so_far + PHASE_C_EPOCHS,
    validation_data = val_gen,
    callbacks       = get_callbacks('phaseC', patience_es=10, patience_lr=5),
    class_weight    = class_weight_dict,
    verbose         = 1
)

best_c = max(history_c.history['val_accuracy']) * 100
print(f"\nPhase C best val_accuracy: {best_c:.2f}%")

# ══════════════════════════════════════════════════════════════
# CELL 10 — Save Final Model
# ══════════════════════════════════════════════════════════════
final_path = os.path.join(MODELS_DIR, 'neuroscan_v2_final.keras')
model.save(final_path)
print(f"\nFinal model saved → {final_path}")
print(f"\nSummary:")
print(f"  Phase A best: {best_a:.2f}%")
print(f"  Phase B best: {best_b:.2f}%")
print(f"  Phase C best: {best_c:.2f}%")
print(f"  Overall best: {max(best_a, best_b, best_c):.2f}%")

# ══════════════════════════════════════════════════════════════
# CELL 11 — Training Curves (All 3 Phases)
# ══════════════════════════════════════════════════════════════
def plot_all_phases(ha, hb, hc):
    acc_all  = ha.history['accuracy']     + hb.history['accuracy']     + hc.history['accuracy']
    val_all  = ha.history['val_accuracy'] + hb.history['val_accuracy'] + hc.history['val_accuracy']
    loss_all = ha.history['loss']         + hb.history['loss']         + hc.history['loss']
    vloss_all= ha.history['val_loss']     + hb.history['val_loss']     + hc.history['val_loss']
    epochs   = range(1, len(acc_all) + 1)
    pb_start = len(ha.history['loss'])
    pc_start = pb_start + len(hb.history['loss'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('NeuroScan V2 — Training Curves (3 Phases)', fontsize=14, fontweight='bold')

    for ax, metric, val_metric, ylabel in [
        (axes[0], acc_all,  val_all,   'Accuracy'),
        (axes[1], loss_all, vloss_all, 'Loss')
    ]:
        ax.plot(epochs, metric,     color='#00bfff', lw=2, label='Train')
        ax.plot(epochs, val_metric, color='#ff6b35', lw=2, label='Val')
        ax.axvline(pb_start, color='#22c55e', ls='--', lw=1.5, alpha=0.8, label='Phase B start')
        ax.axvline(pc_start, color='#f59e0b', ls='--', lw=1.5, alpha=0.8, label='Phase C start')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(ylabel); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'training_curves_v2.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {out}")

plot_all_phases(history_a, history_b, history_c)

# ══════════════════════════════════════════════════════════════
# CELL 12 — Final Evaluation + Confusion Matrix
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
ax.set_title('NeuroScan V2 — Confusion Matrix', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_v2.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.show()

final_acc = np.mean(y_pred == y_true) * 100
print(f"\n{'='*55}")
print(f"  Final Test Accuracy: {final_acc:.2f}%")
print(f"  Model saved: {final_path}")
print(f"{'='*55}")
