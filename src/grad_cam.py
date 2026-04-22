"""
NeuroScan — Grad-CAM Explainability Module
==========================================
Generates Gradient-weighted Class Activation Maps (Grad-CAM) to visualize 
which regions of the MRI the CNN focused on when making its prediction.

This is critical for clinical explainability and Viva presentations:
  "The model isn't a black box — we can see exactly where the tumor was detected."

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017)
"""

import numpy as np
import tensorflow as tf
import cv2
import os


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    """
    Computes the Grad-CAM heatmap for a given image and model.
    
    Args:
        img_array   : Preprocessed image tensor, shape (1, 260, 260, 3), EfficientNet-preprocessed float32
        model       : Loaded Keras model
        last_conv_layer_name: Name of the last convolutional layer to extract gradients from
        
    Returns:
        heatmap     : 2D numpy array of shape (H, W) with values in [0, 1]
    """
    # Build a sub-model that outputs the last conv layer AND the final predictions
    grad_model = tf.keras.models.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradients of the top predicted class w.r.t. the last conv layer output
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        # Get the index of the predicted class
        pred_index = tf.argmax(predictions[0])
        # Extract the softmax score for that class
        class_channel = predictions[:, pred_index]

    # Gradients of class score w.r.t. convolutional layer output
    grads = tape.gradient(class_channel, conv_outputs)

    # Global Average Pooling over spatial dimensions to get per-filter weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) != 0:
        heatmap = heatmap / tf.reduce_max(heatmap)
    
    return heatmap.numpy()


def overlay_gradcam(original_img_path, heatmap, alpha=0.45):
    """
    Overlays the Grad-CAM heatmap onto the original raw MRI image.

    Args:
        original_img_path: Path to the original (raw, unprocessed) MRI image.
        heatmap           : 2D numpy heatmap from make_gradcam_heatmap()
        alpha             : Blend strength of the heatmap overlay (0=no overlay, 1=full overlay)

    Returns:
        superimposed_img  : BGR numpy image suitable for cv2.imwrite / web serving
    """
    # Read the original image at full size
    img = cv2.imread(original_img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {original_img_path}")
    
    img_h, img_w = img.shape[:2]

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_w, img_h))

    # Convert to 8-bit and apply JET colormap (blue=low, red=high activation)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend: original image + colored heatmap
    superimposed = cv2.addWeighted(img, 1 - alpha, colored_heatmap, alpha, 0)

    return superimposed


def generate_gradcam(original_img_path, preprocessed_tensor, model, save_path):
    """
    Full pipeline: computes Grad-CAM heatmap and saves the overlaid image.

    Args:
        original_img_path  : Path to the raw input MRI scan.
        preprocessed_tensor: Shape (1,260,260,3), EfficientNet-preprocessed float32 — same tensor used for prediction.
        model              : The loaded Keras model.
        save_path          : Output path to save the heatmap image (e.g. PNG file).

    Returns:
        save_path          : The path where the heatmap was saved.
    """
    # 1. Compute the heatmap
    heatmap = make_gradcam_heatmap(preprocessed_tensor, model)

    # 2. Overlay onto the original raw image
    overlaid = overlay_gradcam(original_img_path, heatmap)

    # 3. Save result
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlaid)
    
    return save_path
