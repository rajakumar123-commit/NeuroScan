"""
NeuroScan — EfficientNetB4 Training (Target: 97%+)
===================================================
EfficientNetB4 vs VGG16:
  - 4x fewer parameters yet significantly more accurate
  - Uses compound scaling (depth + width + resolution together)
  - State-of-the-art for medical imaging classification
  - Expected: 96-98% on this brain tumor dataset

Instructions:
  1. Runtime -> Change runtime type -> T4 GPU
  2. Run each cell in order
  3. Expected total time: ~50-70 minutes
"""

# ══════════════════════════════════════════════════════════════
# CELL 1 — Mount Drive & Imports
# ══════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

import os, shutil, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))


# ══════════════════════════════════════════════════════════════
# CELL 2 — Config & Dataset
# ══════════════════════════════════════════════════════════════
MODELS_DIR  = '/content/drive/MyDrive/NeuroScan/models'
RESULTS_DIR = '/content/drive/MyDrive/NeuroScan/results'
DRIVE_DATA  = '/content/drive/MyDrive/NeuroScan/dataset_cropped'
LOCAL_DATA  = '/content/dataset_cropped'
CLASSES     = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_CLASSES   = len(CLASSES)

# EfficientNetB4 native resolution is 380x380
# We use 260x260 — good accuracy vs speed tradeoff on T4
IMG_SIZE    = (260, 260)
BATCH_SIZE  = 16          # smaller batch for larger images
SEED        = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Copy dataset to local SSD for fast training
if not os.path.exists(LOCAL_DATA):
    print("Copying dataset to SSD...")
    shutil.copytree(DRIVE_DATA, LOCAL_DATA)
    print("Done!")
else:
    print("Dataset already on SSD.")

# ── Data Generators ──────────────────────────────────────────
# EfficientNet uses its own internal preprocessing — DO NOT rescale to [0,1]
# Instead pass raw [0,255] uint8 and let EfficientNetB4 preprocess internally
train_datagen = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
    rotation_range         = 30,
    width_shift_range      = 0.15,
    height_shift_range     = 0.15,
    horizontal_flip        = True,
    zoom_range             = 0.2,
    brightness_range       = [0.7, 1.4],
    shear_range            = 0.15,
    channel_shift_range    = 25.0,
    fill_mode              = 'nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    LOCAL_DATA + '/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASSES, shuffle=True, seed=SEED
)
val_gen = val_datagen.flow_from_directory(
    LOCAL_DATA + '/val', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASSES, shuffle=False
)
test_gen = val_datagen.flow_from_directory(
    LOCAL_DATA + '/test', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASSES, shuffle=False
)

print(f"\nTrain:{train_gen.samples} | Val:{val_gen.samples} | Test:{test_gen.samples}")
print(f"Image size : {IMG_SIZE}")
print(f"Batch size : {BATCH_SIZE}")


# ══════════════════════════════════════════════════════════════
# CELL 3 — Build EfficientNetB4 Model
# ══════════════════════════════════════════════════════════════
def build_efficientnet():
    """
    EfficientNetB4 Transfer Learning Model:
      - Base: EfficientNetB4 pretrained on ImageNet (19M parameters)
      - Head: GAP → Dense(256) → Dropout → BatchNorm → Softmax(4)
    """
    base = EfficientNetB4(
        weights     = 'imagenet',
        include_top = False,
        input_shape = (*IMG_SIZE, 3)
    )
    base.trainable = False   # freeze for Phase A

    x = base.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(
        512, activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='dense_512'
    )(x)
    x = layers.Dropout(0.4, name='dropout_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dense(
        256, activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='dense_256'
    )(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    output = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)

    model = Model(inputs=base.input, outputs=output)
    return model, base

model, base_model = build_efficientnet()
total     = len(model.layers)
trainable = sum(1 for l in model.layers if l.trainable)
total_params = model.count_params()
print(f"\nEfficientNetB4 built:")
print(f"  Total layers    : {total}")
print(f"  Trainable layers: {trainable} (Phase A — head only)")
print(f"  Total parameters: {total_params:,}")

# Label smoothing loss
smooth_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)


# ══════════════════════════════════════════════════════════════
# CELL 4 — Phase A: Train Head Only (base frozen)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE A — Training custom head (EfficientNetB4 frozen)")
print("="*60)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss      = smooth_loss,
    metrics   = ['accuracy']
)

callbacks_a = [
    EarlyStopping(monitor='val_accuracy', patience=7,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath       = f'{MODELS_DIR}/effnet_phaseA.keras',
        monitor        = 'val_accuracy',
        save_best_only = True, verbose=1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                      patience=3, min_lr=1e-7, verbose=1),
]

history_a = model.fit(
    train_gen,
    epochs          = 20,
    validation_data = val_gen,
    callbacks       = callbacks_a,
    verbose         = 1
)

best_a = max(history_a.history['val_accuracy']) * 100
print(f"\nPhase A best val_accuracy: {best_a:.2f}%")


# ══════════════════════════════════════════════════════════════
# CELL 5 — Phase B: Fine-tune top layers
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE B — Fine-tuning EfficientNetB4 (top 30 layers)")
print("="*60)

# Unfreeze last 30 layers of EfficientNetB4
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

finetune_layers = [l.name for l in base_model.layers if l.trainable]
print(f"Fine-tuning {len(finetune_layers)} base layers")

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss      = smooth_loss,
    metrics   = ['accuracy']
)

callbacks_b = [
    EarlyStopping(monitor='val_accuracy', patience=8,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath       = f'{MODELS_DIR}/effnet_phaseB.keras',
        monitor        = 'val_accuracy',
        save_best_only = True, verbose=1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                      patience=3, min_lr=1e-8, verbose=1),
]

history_b = model.fit(
    train_gen,
    initial_epoch   = len(history_a.history['loss']),
    epochs          = len(history_a.history['loss']) + 25,
    validation_data = val_gen,
    callbacks       = callbacks_b,
    verbose         = 1
)

best_b = max(history_b.history['val_accuracy']) * 100
print(f"\nPhase B best val_accuracy: {best_b:.2f}%")

# Save final model
final_path = f'{MODELS_DIR}/neuroscan_efficientnet_final.keras'
model.save(final_path)
print(f"Model saved → {final_path}")


# ══════════════════════════════════════════════════════════════
# CELL 6 — Final Evaluation + Confusion Matrix
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
ax.set_title('NeuroScan EfficientNetB4 — Confusion Matrix',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
cm_path = f'{RESULTS_DIR}/confusion_matrix_efficientnet.png'
plt.savefig(cm_path, dpi=150)
plt.show()

final_acc = np.mean(y_pred == y_true) * 100
print(f"\n{'='*55}")
print(f"  VGG16 baseline        : 89.40%")
print(f"  VGG16 Phase C         : 92.26%")
print(f"  EfficientNetB4 Phase A: {best_a:.2f}%")
print(f"  EfficientNetB4 Phase B: {best_b:.2f}%")
print(f"  Final Test Accuracy   : {final_acc:.2f}%")
print(f"  Model saved to        : {final_path}")
print(f"{'='*55}")
