"""
NeuroScan — Flask Web Application Backend
=========================================
Serves the NeuroScan UI and provides an API endpoint for MRI scan analysis.

Routes:
  GET  /         → Serves the main UI dashboard (index.html)
  POST /predict  → Accepts uploaded MRI image, runs preprocessing + EfficientNetB4 inference
                   + Grad-CAM, returns JSON results
"""

import os
import sys
import json
import uuid
import numpy as np

# Add src directory to path
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(SRC_DIR))

from flask import Flask, request, jsonify, render_template, send_from_directory

# Suppress TensorFlow logs before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Limit TF from pre-allocating all RAM — grow on demand instead
tf.config.set_soft_device_placement(True)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
# Also limit CPU memory pre-allocation
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from preprocess import process_single_image
from grad_cam import generate_gradcam

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'neuroscan_efficientnet_final.keras')
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
HEATMAP_DIR = os.path.join(os.path.dirname(__file__), 'static', 'heatmaps')
CLASSES    = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Class display info
CLASS_META = {
    'glioma':      {'label': 'Glioma',      'color': '#ef4444', 'icon': '⚠️'},
    'meningioma':  {'label': 'Meningioma',  'color': '#f97316', 'icon': '⚠️'},
    'notumor':     {'label': 'No Tumor',    'color': '#22c55e', 'icon': '✅'},
    'pituitary':   {'label': 'Pituitary',   'color': '#f59e0b', 'icon': '⚠️'},
}

os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# MODEL AUTO-DOWNLOAD (for Render — model too large for GitHub)
# ─────────────────────────────────────────────────────────────────
def download_model_if_missing():
    """Downloads model from Google Drive if not present (Render deployment)."""
    if os.path.exists(MODEL_PATH):
        return  # Already there, skip

    gdrive_url = os.environ.get('GDRIVE_MODEL_URL', '')
    if not gdrive_url:
        print("[WARNING] Model file missing and GDRIVE_MODEL_URL not set.")
        return

    print(f"[INFO] Model not found. Downloading from Google Drive...")
    try:
        import gdown
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        gdown.download(gdrive_url, MODEL_PATH, quiet=False, fuzzy=True)
        print("[INFO] Model downloaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")

# ─────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Download model if on Render and missing
download_model_if_missing()

# Load model once on startup (expensive operation)
print("Loading NeuroScan model...")
if not os.path.exists(MODEL_PATH):
    print(f"[WARNING] Model not found at {MODEL_PATH}")
    print("          Please place 'neuroscan_efficientnet_final.keras' in the models/ folder.")
    model = None
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully.")


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts an MRI image upload, runs the full NeuroScan pipeline:
      1. Save uploaded image
      2. OpenCV Preprocessing (preprocess.py)
      3. EfficientNetB4 Prediction with TTA (model.predict)
      4. Grad-CAM Heatmap Generation (grad_cam.py)
      5. Return JSON results
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Please place neuroscan_efficientnet_final.keras in the models/ folder.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    # Check extension
    allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Invalid file type: {ext}. Please upload a JPG or PNG.'}), 400

    try:
        # ── 1. Save uploaded file ────────────────────────
        uid = str(uuid.uuid4())[:8]
        upload_filename = f"{uid}{ext}"
        upload_path = os.path.join(UPLOAD_DIR, upload_filename)
        file.save(upload_path)

        # ── 2. Preprocess ────────────────────────────────
        processed_img = process_single_image(upload_path)
        if processed_img is None:
            return jsonify({'error': 'Image failed quality gate. Please upload a clear MRI scan.'}), 422

        import cv2
        # ── 3. Predict (Test-Time Augmentation) ──────────
        img_resized = cv2.resize(processed_img, (260, 260))

        img_normal = img_resized
        img_hf     = cv2.flip(img_resized, 1)
        img_vf     = cv2.flip(img_resized, 0)

        tta_batch = np.array([img_normal, img_hf, img_vf], dtype=np.float32)
        tta_batch = tf.keras.applications.efficientnet.preprocess_input(tta_batch)

        predictions = model.predict(tta_batch, verbose=0)
        confidences = np.mean(predictions, axis=0)

        pred_idx    = int(np.argmax(confidences))
        pred_class  = CLASSES[pred_idx]
        confidence  = float(confidences[pred_idx]) * 100

        breakdown = {cls: round(float(confidences[i]) * 100, 2) for i, cls in enumerate(CLASSES)}

        # ── 4. Grad-CAM ──────────────────────────────────
        heatmap_filename = f"heatmap_{uid}.jpg"
        heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
        gradcam_tensor = np.expand_dims(tta_batch[0], axis=0)
        generate_gradcam(upload_path, gradcam_tensor, model, heatmap_path)

        # ── 5. Return JSON ───────────────────────────────
        meta = CLASS_META[pred_class]
        return jsonify({
            'success':        True,
            'diagnosis':      meta['label'],
            'class':          pred_class,
            'confidence':     round(confidence, 2),
            'color':          meta['color'],
            'icon':           meta['icon'],
            'breakdown':      breakdown,
            'heatmap_url':    f"/static/heatmaps/{heatmap_filename}",
            'upload_url':     f"/static/uploads/{upload_filename}",
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 50)
    print("  NeuroScan Web UI")
    print(f"  Open: http://0.0.0.0:{port}")
    print("=" * 50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)
