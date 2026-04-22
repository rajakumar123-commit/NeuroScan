"""
NeuroScan — Phase C Resume Training (Start from existing 90.7% model)
======================================================================
This notebook RESUMES from best_phaseB.keras — NO retraining from scratch.
Estimated time: ~10-15 minutes on T4 GPU.

Instructions:
  1. Runtime → Change runtime type → T4 GPU
  2. Run each cell in order
"""

# ══════════════════════════════════════════════════════════════
# CELL 1 — Mount Drive & Setup
# ══════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile, numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))


# ══════════════════════════════════════════════════════════════
# CELL 2 — Load Existing Model + Dataset
# ══════════════════════════════════════════════════════════════
MODELS_DIR = '/content/drive/MyDrive/NeuroScan/models'
DATA_DIR   = '/content/dataset_cropped'
CLASSES    = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load the 90.7% model — skip the 20 mins of Phase A & B!
print("Loading trained model...")
model = tf.keras.models.load_model(f'{MODELS_DIR}/best_phaseB.keras')
print(f"Model loaded! Layers: {len(model.layers)}")

# Extract dataset to local SSD (skip if already done)
if not os.path.exists(DATA_DIR):
    print("Extracting dataset to SSD...")
    with zipfile.ZipFile('/content/drive/MyDrive/NeuroScan/dataset_cropped.zip') as z:
        z.extractall(DATA_DIR)
    print("Done!")
else:
    print("Dataset already on SSD.")

# Data generators — stronger augmentation for Phase C
train_gen = ImageDataGenerator(
    rescale            = 1./255,
    rotation_range     = 25,
    width_shift_range  = 0.15,
    height_shift_range = 0.15,
    horizontal_flip    = True,
    zoom_range         = 0.2,
    brightness_range   = [0.75, 1.3],
    shear_range        = 0.15,
    channel_shift_range= 20.0,   # Simulates different scanner calibrations
    fill_mode          = 'nearest'
).flow_from_directory(
    DATA_DIR + '/train', target_size=(224, 224),
    batch_size=32, class_mode='categorical',
    classes=CLASSES, shuffle=True, seed=42
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    DATA_DIR + '/val', target_size=(224, 224),
    batch_size=32, class_mode='categorical',
    classes=CLASSES, shuffle=False
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    DATA_DIR + '/test', target_size=(224, 224),
    batch_size=32, class_mode='categorical',
    classes=CLASSES, shuffle=False
)

print(f"\nTrain:{train_gen.samples} | Val:{val_gen.samples} | Test:{test_gen.samples}")


# ══════════════════════════════════════════════════════════════
# CELL 3 — Phase C: Unfreeze block4+5, Deep Fine-Tune
# ══════════════════════════════════════════════════════════════

# Unfreeze block4 + block5 (layer index 11 onwards in VGG16)
# Keeps block1, block2, block3 frozen (basic edge/texture detectors — no need to touch)
for i, layer in enumerate(model.layers):
    layer.trainable = i >= 11

unfrozen = [l.name for l in model.layers if l.trainable]
print(f"Unfrozen layers ({len(unfrozen)}): {unfrozen}")

# Very low LR to avoid catastrophic forgetting
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss      = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics   = ['accuracy']
)

history = model.fit(
    train_gen,
    epochs          = 15,
    validation_data = val_gen,
    callbacks=[
        EarlyStopping(
            monitor              = 'val_accuracy',
            patience             = 6,
            restore_best_weights = True,
            verbose              = 1
        ),
        ModelCheckpoint(
            filepath       = f'{MODELS_DIR}/neuroscan_final.keras',
            monitor        = 'val_accuracy',
            save_best_only = True,
            verbose        = 1
        ),
        ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.4,
            patience = 3,
            min_lr   = 1e-8,
            verbose  = 1
        ),
    ],
    verbose=1
)

print(f"\nPhase C best val_accuracy: {max(history.history['val_accuracy'])*100:.2f}%")


# ══════════════════════════════════════════════════════════════
# CELL 4 — Final Evaluation + Confusion Matrix
# ══════════════════════════════════════════════════════════════
test_gen.reset()
preds  = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
ax.set_title('NeuroScan — Phase C Confusion Matrix', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{MODELS_DIR}/confusion_matrix_phaseC.png', dpi=150)
plt.show()

final_acc = np.mean(y_pred == y_true) * 100
print(f"\n{'='*50}")
print(f"  Final Test Accuracy : {final_acc:.2f}%")
print(f"  Model saved to      : {MODELS_DIR}/neuroscan_final.keras")
print(f"{'='*50}")
