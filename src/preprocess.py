"""
NeuroScan — Improved Hybrid Preprocessing Pipeline
===================================================
Implements the IMPROVED pipeline from the architectural diagram:

EXISTING:  Median filter → CLAHE → Otsu threshold → Morphological close → Pad → Resize 224×224
  NEW:     + Adaptive gamma correction   (normalises brightness across different scanners)
  NEW:     + Skull stripping             (removes bone, keeps only soft brain tissue)
UPGRADED:  + Quality gate on crop        (skips bad/corrupt scans automatically)
  NEW:     + Resume check               (skip already-processed files on restart)
"""

import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# STEP A ─ Adaptive gamma correction (NEW)
# Normalises brightness across different hospital scanners.
# ─────────────────────────────────────────────────────────────────
def adaptive_gamma_correction(image):
    """Automatically correct image brightness using mean luminance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_val = np.mean(gray)

    # Target mean brightness = 128 (mid-gray)
    if mean_val == 0:
        return image
    gamma = np.log(128.0 / 255.0) / np.log(mean_val / 255.0)
    gamma = np.clip(gamma, 0.4, 2.5)   # clamp to safe range

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


# ─────────────────────────────────────────────────────────────────
# STEP B ─ CLAHE local contrast enhancement (EXISTING)
# ─────────────────────────────────────────────────────────────────
def apply_clahe(gray):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ─────────────────────────────────────────────────────────────────
# STEP C ─ Skull stripping (NEW)
# Removes hard bone structures, keeps only soft brain tissue.
# ─────────────────────────────────────────────────────────────────
def skull_strip(gray):
    """
    Attempt to remove the bright skull ring using a flood-fill strategy
    combined with morphological erosion on the brain boundary.
    Returns a mask where only soft-tissue brain pixels are white.
    """
    # Use Otsu threshold to separate background from tissue
    _, brain_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological close to fill holes inside the brain
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel_close)

    # Erode to trim the bright skull ring off the outer edge
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    brain_mask = cv2.erode(brain_mask, kernel_erode, iterations=2)

    return brain_mask


# ─────────────────────────────────────────────────────────────────
# STEP D ─ Largest-contour crop with Quality Gate (UPGRADED)
# Skips bad scans automatically instead of crashing.
# ─────────────────────────────────────────────────────────────────
QUALITY_GATE_MIN_AREA_RATIO = 0.05   # brain must occupy at least 5% of frame

def crop_brain_contour(image, mask):
    """
    Find the brain boundary via contours and crop to its bounding box.
    Quality gate: returns None if the detected brain region is too small
    (indicating a corrupt/noisy scan).
    """
    total_area = image.shape[0] * image.shape[1]

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # Quality gate: reject scans where the detected brain is too tiny
    if area / total_area < QUALITY_GATE_MIN_AREA_RATIO:
        return None

    x, y, w, h = cv2.boundingRect(c)

    # Add a small 5-pixel padding around the crop
    pad = 5
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)

    return image[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────────
# STEP E ─ Square padding (EXISTING)
# ─────────────────────────────────────────────────────────────────
def pad_to_square(image):
    """Pad image to square using mean pixel value (avoids distortion)."""
    h, w = image.shape[:2]
    if h == w:
        return image
    size = max(h, w)
    mean_color = [int(np.mean(image[:, :, c])) for c in range(3)]
    padded = np.full((size, size, 3), mean_color, dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = image
    return padded


# ─────────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────
def process_single_image(image_path):
    """
    Full improved pipeline:
      Raw MRI
        → Median filter (noise)
        → Adaptive gamma correction      [NEW]
        → Grayscale
        → CLAHE (contrast enhancement)  [EXISTING]
        → Skull stripping               [NEW]
        → Otsu thresholding             [EXISTING]
        → Morphological close           [EXISTING]
        → Largest contour crop + QGate  [UPGRADED]
        → Pad to square
        → Resize 224×224
    Returns the processed image, or None if quality gate rejects it.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 1. Median filter — noise removal
    img = cv2.medianBlur(img, 5)

    # 2. Adaptive gamma correction — normalise brightness across scanners [NEW]
    img = adaptive_gamma_correction(img)

    # 3. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. CLAHE — local contrast boost [EXISTING]
    gray_clahe = apply_clahe(gray)

    # 5. Skull stripping — remove bone ring [NEW]
    brain_mask = skull_strip(gray_clahe)

    # 6. Largest contour crop + quality gate [UPGRADED]
    cropped = crop_brain_contour(img, brain_mask)

    if cropped is None or cropped.size == 0:
        # Quality gate triggered — fall back to full-image resize
        cropped = img

    # 7. Pad to square — mean-value padding
    squared = pad_to_square(cropped)

    # 8. Resize to 224×224 — VGG16 standard input size
    final = cv2.resize(squared, (224, 224))

    return final


# ─────────────────────────────────────────────────────────────────
# DATASET RUNNER
# ─────────────────────────────────────────────────────────────────
def process_dataset(src_dir, dest_dir):
    """
    Walk every split/class in the source dataset and apply the full pipeline.
    Resume check: already-processed images are skipped automatically [NEW].
    """
    splits = ['train', 'val', 'test']
    total_processed = 0
    total_skipped = 0
    total_rejected = 0

    for split in splits:
        split_src = os.path.join(src_dir, split)
        split_dest = os.path.join(dest_dir, split)

        if not os.path.exists(split_src):
            continue

        classes = sorted(os.listdir(split_src))
        for cls in classes:
            cls_src = os.path.join(split_src, cls)
            cls_dest = os.path.join(split_dest, cls)
            if not os.path.isdir(cls_src):
                continue
            os.makedirs(cls_dest, exist_ok=True)

            files = [f for f in os.listdir(cls_src)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            print(f"\n  [{split.upper()}] {cls} — {len(files)} images")
            for filename in tqdm(files, ncols=80, leave=False):
                dest_path = os.path.join(cls_dest, filename)

                # ── Resume check: skip already processed files [NEW] ──
                if os.path.exists(dest_path):
                    total_skipped += 1
                    continue

                img_path = os.path.join(cls_src, filename)
                result = process_single_image(img_path)

                if result is not None:
                    cv2.imwrite(dest_path, result)
                    total_processed += 1
                else:
                    total_rejected += 1  # unreadable file

    print("\n" + "=" * 55)
    print(" PREPROCESSING COMPLETE - IMPROVED PIPELINE")
    print("=" * 55)
    print(f"  [OK] Processed  : {total_processed} images")
    print(f"  [>>] Resumed    : {total_skipped} already done (skipped)")
    print(f"  [X]  Rejected   : {total_rejected} (unreadable files)")
    print(f"  [D]  Output dir : {dest_dir}")
    print("=" * 55)


if __name__ == "__main__":
    SOURCE_DIR = r"F:\NeuroScan\dataset"
    TARGET_DIR = r"F:\NeuroScan\dataset_cropped"

    print("=" * 55)
    print(" NEUROSCAN - IMPROVED HYBRID PREPROCESSING PIPELINE")
    print("=" * 55)
    print("  Steps: Median -> Gamma -> CLAHE -> Skull strip")
    print("         -> Contour crop + QGate -> Pad -> 224x224")
    print("=" * 55)

    process_dataset(SOURCE_DIR, TARGET_DIR)
