import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def run_phase_c():
    MODELS_DIR = '/content/drive/MyDrive/NeuroScan/models'
    
    print("\n" + "="*55)
    print("  🔥 PHASE C: THE NUCLEAR OPTION (Pushing for 99%)")
    print("="*55)

    # 1. Load the absolute best weights from Phase B
    print("Loading 96.90%+ Phase B model...")
    model = tf.keras.models.load_model(f'{MODELS_DIR}/neuroscan_efficientnet_final.keras')

    # 2. UNFREEZE THE ENTIRE NETWORK (All 482 layers)
    # This destroys the safety rails. The lowest edge-detectors will now morph for MRI data.
    print("Unfreezing ALL EfficientNetB4 Layers...")
    base_model = model.layers[0]
    base_model.trainable = True
    
    for layer in base_model.layers:
        layer.trainable = True

    # 3. RECOMPILE WITH A MICROSCOPIC LEARNING RATE
    # 1e-6 guarantees we don't accidentally destroy the 96.90% achievement
    smooth_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), 
                  loss=smooth_loss, 
                  metrics=['accuracy'])

    # 4. TRAIN FOR A FINAL SPRINT
    history_c = model.fit(
        train_gen, # Assuming train_gen and val_gen are still in Colab RAM from Phase B
        epochs=15,
        validation_data=val_gen,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
            ModelCheckpoint(f'{MODELS_DIR}/neuroscan_99_percent_dream.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-8, verbose=1)
        ], verbose=1
    )
    
    return model

# To run in Colab, literally just define this function and call it:
# final_model = run_phase_c()
