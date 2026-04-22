import numpy as np
import tensorflow as tf
import os
import argparse
from preprocess import process_single_image

# Configuration
# Point this to where you will download the model from Google Drive
MODEL_PATH = r'F:\NeuroScan\models\neuroscan_efficientnet_final.keras'
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_mri(image_path, model_path=MODEL_PATH):
    """
    Takes a raw MRI image path, applies the same preprocessing used during training,
    loads the trained EfficientNetB4 model, and prints the prediction with confidence scores.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        print("Please download 'neuroscan_efficientnet_final.keras' from your Google Drive and place it in the models/ folder.")
        return

    # 1. Preprocess the raw image using our Phase 1 pipeline
    print(f"Preprocessing image: {image_path}...")
    processed_img = process_single_image(image_path)
    
    if processed_img is None:
        print("Error: Image failed quality gate or could not be read.")
        return

    import cv2
    # 2. Upgrade to EfficientNetB4 Target Size (260x260)
    img_resized = cv2.resize(processed_img, (260, 260))

    # 3. Generate Test-Time Augmentation (TTA) Batch
    # Normal, Horizontal Flip, Vertical Flip
    img_normal = img_resized
    img_hf = cv2.flip(img_resized, 1)
    img_vf = cv2.flip(img_resized, 0)
    
    # EfficientNet internally processes pixels via its specific mathematical function
    tta_batch = np.array([img_normal, img_hf, img_vf], dtype=np.float32)
    tta_batch = tf.keras.applications.efficientnet.preprocess_input(tta_batch)

    # 4. Load model (Suppressing verbose TF logs)
    print("Loading AI model...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TF info messages
    model = tf.keras.models.load_model(model_path)

    # 5. Predict using TTA (The Double-Tap)
    print("Running 3-Angle TTA diagnostic prediction...\n")
    predictions = model.predict(tta_batch, verbose=0)
    # Statistically average the 3 probability vectors
    confidences = np.mean(predictions, axis=0)
    
    predicted_idx = np.argmax(confidences)
    predicted_class = CLASSES[predicted_idx]
    confidence_score = confidences[predicted_idx] * 100

    # 5. Output Results
    print("=" * 45)
    print(" NEUROSCAN - MRI ANALYSIS RESULTS")
    print("=" * 45)
    print(f"  Diagnosis  : {predicted_class.upper()}")
    print(f"  Confidence : {confidence_score:.2f}%")
    print("-" * 45)
    print(" Breakdown:")
    for i, cls in enumerate(CLASSES):
        print(f"  - {cls.ljust(12)}: {confidences[i]*100:>6.2f}%")
    print("=" * 45)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze raw MRI scan for brain tumors.')
    parser.add_argument('image_path', type=str, help='Path to the raw MRI image')
    args = parser.parse_args()
    
    predict_mri(args.image_path)
