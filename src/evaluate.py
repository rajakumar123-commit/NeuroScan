import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = r"F:\NeuroScan"
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "Testing")

IMG_SIZE = (260, 260)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_with_tta(model, img_array):
    # Base image
    img_base = preprocess_input(img_array)
    
    # Horizontal flip
    img_hflip = np.fliplr(img_array)
    img_hflip = preprocess_input(img_hflip)
    
    # Vertical flip
    img_vflip = np.flipud(img_array)
    img_vflip = preprocess_input(img_vflip)
    
    # Batch predict
    batch = np.array([img_base, img_hflip, img_vflip])
    preds = model.predict(batch, verbose=0)
    
    # Average predictions
    avg_pred = np.mean(preds, axis=0)
    return avg_pred

def main():
    print("=" * 62)
    print("  NeuroScan — Official Evaluation")
    print("  Model  : neuroscan_efficientnet_final.keras")
    print(f"  Test   : {TEST_DIR}  (1600 held-out images)")
    print("=" * 62)

    model_path = os.path.join(MODELS_DIR, "neuroscan_efficientnet_final.keras")
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model file not found at {model_path}")
        print("Please download it from Colab and place it in the models directory.")
        return

    print("\n  Loading model...")
    model = load_model(model_path)
    print("  Model loaded.")

    print("\n  Evaluating test set (Sequential mode for low RAM)...")
    
    y_true = []
    y_pred = []
    
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = os.path.join(TEST_DIR, cls_name)
        if not os.path.exists(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        print(f"  [{cls_name:>12}] processing {len(images)} images...")
        
        for i, img_name in enumerate(images):
            img_path = os.path.join(cls_dir, img_name)
            
            # Load and predict
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            
            # TTA prediction
            pred_probs = predict_with_tta(model, img_array)
            pred_class = np.argmax(pred_probs)
            
            y_true.append(cls_idx)
            y_pred.append(pred_class)
            
            if (i + 1) % 100 == 0:
                print(f"    -> {i + 1}/{len(images)} done")

    # Generate Metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('NeuroScan — Confusion Matrix', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    report_path = os.path.join(RESULTS_DIR, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\n  [OK] Confusion matrix saved -> {cm_path}")
    print(f"  [OK] Classification report saved -> {report_path}")

    print("\n" + "=" * 62)
    print("  NeuroScan — EfficientNetB4 + TTA — Classification Report")
    print("=" * 62)
    print("\n" + report)
    
    acc = np.mean(np.array(y_pred) == np.array(y_true)) * 100
    print(f"\n  Verified Test Accuracy : {acc:.2f}%")
    print(f"  Total images evaluated : {len(y_true)}")
    print(f"  Correct predictions    : {np.sum(np.array(y_pred) == np.array(y_true))}")
    print(f"  Wrong  predictions     : {np.sum(np.array(y_pred) != np.array(y_true))}")
    print("\n  Model  : neuroscan_efficientnet_final.keras")
    print("  Arch   : EfficientNetB4 (ImageNet pretrained, fine-tuned)")
    print("  TTA    : 3-view (normal, horizontal flip, vertical flip)")
    print("=" * 62)

if __name__ == "__main__":
    main()
