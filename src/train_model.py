"""
NeuroScan — Phase 2: VGG16 Transfer Learning Training Pipeline
==============================================================
Architecture:
  Input (224x224x3)
    -> VGG16 Base (13 frozen Conv layers, pretrained ImageNet)
    -> GlobalAveragePooling2D
    -> Dense(256, relu) + Dropout(0.5)
    -> Dense(4, softmax) -> [glioma, meningioma, notumor, pituitary]

Training Strategy:
  Phase A: Train only the custom head (base frozen)     - 20 epochs max
  Phase B: Fine-tune top 4 VGG16 layers (base partial)  - 10 epochs max

Callbacks:
  EarlyStopping   - patience=5, monitors val_loss
  ModelCheckpoint - saves best weights only
  ReduceLROnPlateau - halves LR when val_loss plateaus
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress TF info/warning logs

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
BASE_DIR      = r"F:\NeuroScan"
DATA_DIR      = os.path.join(BASE_DIR, "dataset_cropped")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
SEED          = 42

# Phase A — head training
PHASE_A_EPOCHS    = 20
PHASE_A_LR        = 1e-4

# Phase B — fine-tuning top layers
PHASE_B_EPOCHS    = 10
PHASE_B_LR        = 1e-5
FINE_TUNE_AT      = 15   # unfreeze VGG16 layers from this index onwards

CLASSES           = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_CLASSES         = len(CLASSES)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# STEP 1 — Data Generators
# ─────────────────────────────────────────────────────────────────
def build_generators():
    """
    Training generator uses heavy augmentation to prevent overfitting.
    Val/Test generators only rescale — no augmentation on evaluation data.
    """
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
        target_size  = IMG_SIZE,
        batch_size   = BATCH_SIZE,
        class_mode   = 'categorical',
        classes      = CLASSES,
        shuffle      = True,
        seed         = SEED
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size  = IMG_SIZE,
        batch_size   = BATCH_SIZE,
        class_mode   = 'categorical',
        classes      = CLASSES,
        shuffle      = False
    )

    test_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size  = IMG_SIZE,
        batch_size   = BATCH_SIZE,
        class_mode   = 'categorical',
        classes      = CLASSES,
        shuffle      = False
    )

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────────
# STEP 2 — Model Architecture
# ─────────────────────────────────────────────────────────────────
def build_model():
    """
    Hybrid VGG16 Transfer Learning model:
      - VGG16 base (frozen)
      - Custom classification head on top
    """
    # Load VGG16 without the top classification layers
    base_model = VGG16(
        weights       = 'imagenet',
        include_top   = False,
        input_shape   = (*IMG_SIZE, 3)
    )

    # Freeze all base model layers
    base_model.trainable = False

    # Build custom head
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    x = layers.BatchNormalization(name='batch_norm')(x)
    output = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model, base_model


# ─────────────────────────────────────────────────────────────────
# STEP 3 — Training Callbacks
# ─────────────────────────────────────────────────────────────────
def get_callbacks(phase_name):
    model_path  = os.path.join(MODELS_DIR, f'best_model_{phase_name}.keras')
    csv_path    = os.path.join(RESULTS_DIR, f'training_log_{phase_name}.csv')

    callbacks = [
        EarlyStopping(
            monitor              = 'val_loss',
            patience             = 5,
            restore_best_weights = True,
            verbose              = 1
        ),
        ModelCheckpoint(
            filepath        = model_path,
            monitor         = 'val_accuracy',
            save_best_only  = True,
            verbose         = 1
        ),
        ReduceLROnPlateau(
            monitor   = 'val_loss',
            factor    = 0.5,
            patience  = 3,
            min_lr    = 1e-7,
            verbose   = 1
        ),
        CSVLogger(csv_path, append=True)
    ]
    return callbacks, model_path


# ─────────────────────────────────────────────────────────────────
# STEP 4 — Plot & Save Training Curves
# ─────────────────────────────────────────────────────────────────
def plot_history(history, phase_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'NeuroScan Training — {phase_name}', fontsize=14, fontweight='bold')

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Acc',  color='#00bfff', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Acc',    color='#ff6b35', lw=2)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss', color='#00bfff', lw=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss',   color='#ff6b35', lw=2)
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f'training_curves_{phase_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot saved] {out_path}")


# ─────────────────────────────────────────────────────────────────
# STEP 5 — Confusion Matrix
# ─────────────────────────────────────────────────────────────────
def evaluate_model(model, test_gen):
    print("\n" + "=" * 55)
    print("  FINAL EVALUATION ON TEST SET")
    print("=" * 55)

    test_gen.reset()
    preds      = model.predict(test_gen, verbose=1)
    y_pred     = np.argmax(preds, axis=1)
    y_true     = test_gen.classes

    # Classification Report
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES, yticklabels=CLASSES,
        ax=ax
    )
    ax.set_title('NeuroScan — Confusion Matrix', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Confusion matrix saved] {cm_path}")

    # Overall accuracy
    acc = np.mean(y_pred == y_true) * 100
    print(f"\n  Test Accuracy: {acc:.2f}%")
    return acc


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NEUROSCAN - VGG16 TRAINING PIPELINE")
    print("=" * 55)
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPU devices: {tf.config.list_physical_devices('GPU')}")
    print(f"  Classes    : {CLASSES}")
    print(f"  Batch size : {BATCH_SIZE}")
    print("=" * 55)

    # ── Data ────────────────────────────────────────────
    print("\n[1/5] Building data generators...")
    train_gen, val_gen, test_gen = build_generators()

    print(f"\n  Train samples : {train_gen.samples}")
    print(f"  Val   samples : {val_gen.samples}")
    print(f"  Test  samples : {test_gen.samples}")

    # ── Model ───────────────────────────────────────────
    print("\n[2/5] Building VGG16 model...")
    model, base_model = build_model()
    model.summary()

    # ── Phase A: Head Training ───────────────────────────
    print("\n[3/5] Phase A — Training custom head (base frozen)...")
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=PHASE_A_LR),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )

    callbacks_a, model_path_a = get_callbacks('phaseA')
    history_a = model.fit(
        train_gen,
        epochs          = PHASE_A_EPOCHS,
        validation_data = val_gen,
        callbacks       = callbacks_a,
        verbose         = 1
    )
    plot_history(history_a, 'PhaseA_Head')

    # ── Phase B: Fine-tuning ─────────────────────────────
    print(f"\n[4/5] Phase B — Fine-tuning (unfreezing VGG16 layers from index {FINE_TUNE_AT})...")
    base_model.trainable = True

    # Freeze layers before FINE_TUNE_AT, unfreeze from FINE_TUNE_AT onwards
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=PHASE_B_LR),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  VGG16 trainable layers: {trainable_count}")

    callbacks_b, model_path_b = get_callbacks('phaseB_finetune')
    history_b = model.fit(
        train_gen,
        initial_epoch   = len(history_a.history['loss']),
        epochs          = len(history_a.history['loss']) + PHASE_B_EPOCHS,
        validation_data = val_gen,
        callbacks       = callbacks_b,
        verbose         = 1
    )
    plot_history(history_b, 'PhaseB_Finetune')

    # ── Save final model ─────────────────────────────────
    final_model_path = os.path.join(MODELS_DIR, 'best_model.keras')
    model.save(final_model_path)
    print(f"\n  [Model saved] {final_model_path}")

    # ── Evaluate ─────────────────────────────────────────
    print("\n[5/5] Evaluating on test set...")
    model.load_weights(model_path_b)
    final_acc = evaluate_model(model, test_gen)

    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE")
    print(f"  Final Test Accuracy : {final_acc:.2f}%")
    print(f"  Model saved to      : {final_model_path}")
    print(f"  Results saved to    : {RESULTS_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
