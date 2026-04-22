<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0e0d,100:1a3a5c&height=200&section=header&text=NeuroScan%20AI&fontSize=60&fontColor=ffffff&fontAlignY=35&desc=Brain%20Tumor%20Detection%20%7C%20EfficientNetB4%20%7C%20Grad-CAM&descAlignY=60&descColor=a0b4c8" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

<br/>

[![Accuracy](https://img.shields.io/badge/✅_Test_Accuracy-94.88%25_Verified-22c55e?style=for-the-badge)](https://github.com/rajakumar123-commit/NeuroScan)
[![Dataset](https://img.shields.io/badge/📊_Test_Set-1600_Unseen_MRIs_(400_per_class)-3b82f6?style=for-the-badge)](https://github.com/rajakumar123-commit/NeuroScan)
[![Classes](https://img.shields.io/badge/🧠_Classes-Glioma_·_Meningioma_·_No_Tumor_·_Pituitary-f59e0b?style=for-the-badge)](https://github.com/rajakumar123-commit/NeuroScan)

<br/>

**Type: Computer-Aided Diagnosis (CAD) System**  
**A production-grade hybrid deep learning system for clinical-quality MRI brain tumor classification.**  
*OpenCV 6-stage preprocessing · EfficientNetB4 + Fine-tuning · 5-view TTA · Grad-CAM Tumor Localization · Flask Web UI*

<br/>

[🚀 Quick Start](#-quick-start) &nbsp;·&nbsp; [📐 Pipeline](#-complete-pipeline) &nbsp;·&nbsp; [📊 Results](#-results--performance) &nbsp;·&nbsp; [🛡️ Viva Defense](#️-viva-defense-notes)

</div>

---

## 🎯 Project Overview

| Component | Implementation Detail |
|:---|:---|
| **Input** | Raw MRI scan (JPG/PNG/BMP/WebP, any brightness) |
| **Output** | Class + Confidence % + 4-class breakdown + Grad-CAM heatmap |
| **Classes** | `glioma` · `meningioma` · `notumor` · `pituitary` |
| **Image Size** | 260 × 260 × 3 (EfficientNetB4 native resolution) |
| **Model** | EfficientNetB4 (ImageNet pretrained) + Custom Head |
| **Training** | Phase A: frozen base · Phase B: unfreeze last 30 layers |
| **Loss** | `CategoricalFocalCrossentropy` + class_weight `{glioma: 1.5}` |
| **Inference** | 5-view TTA (normal + h-flip + v-flip + rot90°CW + rot90°CCW) → `np.mean(axis=0)` + temperature scaling |
| **Explainability** | Grad-CAM Tumor Localization (Model Attention Map) via `GradientTape` on `top_conv` layer |
| **Inference Time** | ~0.5–1.2 seconds per image (preprocessing + 5-view TTA + Grad-CAM) |
| **System Type** | Computer-Aided Diagnosis (CAD) |

---

## 📊 Results & Performance

<div align="center">

### ✅ Verified Test Results — 1600 Unseen MRI Images (400 per class)

| Class | Precision | Recall | F1-Score | Support |
|:---:|:---:|:---:|:---:|:---:|
| 🔴 **Glioma** | 0.99 | 0.83 | 0.90 | 400 |
| 🟠 **Meningioma** | 0.89 | 0.98 | 0.93 | 400 |
| 🟢 **No Tumor** | 0.94 | 0.99 | 0.96 | 400 |
| 🟡 **Pituitary** | 0.99 | 0.99 | 0.99 | 400 |
| **Macro Avg** | **0.95** | **0.95** | **0.95** | **1600** |

**Test Accuracy: `94.88%` · Correct: `1518 / 1600` · Model: `neuroscan_efficientnet_final.keras`**
*Model evaluated using strict unseen test set — no data leakage.*

</div>

> *"Glioma detection remains slightly lower (83% recall) due to its irregular morphology — a known challenge in MRI classification literature."*

> **Data Integrity:** All results are obtained on a strictly unseen test set with no overlap with training or validation data.

> **Confidence Calibration Note:** Confidence values are derived from temperature-scaled softmax outputs (T=1.3). Raw softmax probabilities may be overconfident; calibration further improves reliability in clinical settings.

### 📈 Architecture Progression

| Phase | Model | Val Accuracy | Test Accuracy | Params |
|:---|:---|:---:|:---:|:---:|
| Baseline | VGG16 (frozen) | 89.40% | — | 138M |
| Phase C Fine-tune | VGG16 (top-4 unfreeze) | 92.50% | — | 138M |
| Phase A | EfficientNetB4 (head only) | 88.81% | — | 19M |
| **Phase B (Final)** | **EfficientNetB4 (last 30 unfreeze)** | **98.10% ✓** | **94.88% ✓** | **19M** |

> **Validation accuracy (98.10%)** = best epoch on held-out val split during training.  
> **Test accuracy (94.88%)** = final one-time evaluation on 1600 completely unseen images — the only number that matters for scientific validation.

---

## 🔬 Complete Pipeline

> Every step from doctor upload to final result — using the exact code in this repository.

---

### Stage 0 — Upload via Flask (`app/app.py`)

```mermaid
sequenceDiagram
    actor Doctor as 👨‍⚕️ Doctor / User
    participant UI as 🖥️ index.html (JS)
    participant Flask as 🐍 Flask POST /predict

    Doctor->>UI: Drag & Drop or click to upload MRI (.jpg/.png/.bmp/.webp)
    UI->>UI: FileReader API → live preview in upload-zone
    Doctor->>UI: Click "Run Diagnostic Pipeline →"
    UI->>Flask: fetch POST /predict — FormData with image file
    Flask->>Flask: Check extension in {.jpg, .jpeg, .png, .bmp, .webp}
    Flask->>Flask: uuid.uuid4()[:8] → save to app/static/uploads/{uid}.jpg
    Flask-->>UI: ✅ File saved, preprocess.py called next
```

---

### Stage 1 — OpenCV 6-Stage Preprocessing (`src/preprocess.py`)

> The raw MRI passes through 6 deterministic OpenCV steps before the AI ever sees it.

```mermaid
flowchart TD
    A["🩻 Raw MRI\nJPG · PNG · BMP · WebP\nAny brightness · Any scanner model"]
    A --> B

    subgraph PIPE["src/preprocess.py — process_single_image()"]
        B["Step 1 · cv2.medianBlur kernel=3\nRemoves salt-and-pepper scanner noise\nPreserves tumor edge sharpness"]

        B --> C["Step 2 · adaptive_gamma_correction()\nmean_val = np.mean(gray)\ngamma = log(128/255) / log(mean_val/255)\nClamp gamma to [0.4, 2.5]\nNormalises brightness across all hospital scanners"]

        C --> D["Step 3 · apply_clahe()\ncv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))\nBoosts local contrast in 8×8 grid\nMakes gray/white tumor tissue sharper"]

        D --> E["Step 4 · skull_strip() — Otsu Threshold\ncv2.threshold(gray, 0, 255, THRESH_BINARY + THRESH_OTSU)\nAuto-separates background from brain tissue"]

        E --> F["Step 5 · skull_strip() — Morphological Close + Erode\nMORPH_CLOSE kernel=15×15 → fills holes in mask\nMORPH_ERODE kernel=5×5 × 2 → trims bright skull ring\nOnly soft brain tissue pixels remain"]

        F --> G["Step 6 · crop_brain_contour()\ncv2.findContours → select largest blob\nQuality Gate: area / total_area >= 0.05\nCrop bounding box → pad short side with neutral gray\nOutput: 260×260×3 NumPy array"]
    end

    G --> H["✅ Preprocessed Brain Array\n260×260×3 float32\nZero skull · Normalised brightness · No artifacts"]
    F --> GATE["🛑 Quality Gate FAIL\narea/total < 0.05 → return None\nFlask returns HTTP 422 to UI"]

    style A fill:#374151,color:#fff
    style H fill:#065f46,color:#fff
    style GATE fill:#7f1d1d,color:#fff
    style PIPE fill:#1e293b,color:#ccc
```

---

### Stage 2 — 5-View Test-Time Augmentation (`app/app.py`)

> The cleaned image is evaluated from **5 geometric angles** simultaneously to maximise orientation robustness.

```mermaid
flowchart TD
    A["✅ Preprocessed Brain Array\n260×260×3 NumPy array"]

    A --> B["View 1 · img_normal\nOriginal cleaned image\nimg_resized = cv2.resize(processed_img, 260,260)"]
    A --> C["View 2 · img_hf\nHorizontal Mirror\ncv2.flip(img_resized, 1)"]
    A --> D["View 3 · img_vf\nVertical Flip\ncv2.flip(img_resized, 0)"]
    A --> E["View 4 · img_rot90cw\n90° Clockwise Rotation\ncv2.rotate(img_resized, ROTATE_90_CLOCKWISE)"]
    A --> F["View 5 · img_rot90ccw\n90° Counter-Clockwise\ncv2.rotate(img_resized, ROTATE_90_COUNTERCLOCKWISE)"]

    subgraph PRE["EfficientNet Preprocessing"]
        B & C & D & E & F --> G["tf.keras.applications.efficientnet.preprocess_input()\nScales pixel values to EfficientNetB4 internal range"]
    end

    subgraph BATCH["Batch Assembly"]
        G --> H["np.array([img_normal, img_hf, img_vf, img_rot90cw, img_rot90ccw], dtype=np.float32)\nShape: (5, 260, 260, 3)"]
    end

    H --> I["🧠 model.predict(tta_batch, verbose=0)\nAll 5 views inferred simultaneously\nOutput shape: (5, 4)"]
    I --> J["raw_confidences = np.mean(predictions, axis=0)\nconfidences = calibrate_probs(raw, temperature=1.3)\nFinal shape: (4,)"]

    style A fill:#1e3a5f,color:#fff
    style I fill:#6d28d9,color:#fff
    style J fill:#065f46,color:#fff
    style PRE fill:#1e293b,color:#ccc
    style BATCH fill:#1e293b,color:#ccc
```

---

### Stage 3 — EfficientNetB4 Architecture

> Exact architecture used in `neuroscan_efficientnet_final.keras`

```mermaid
flowchart TD
    A["📦 TTA Batch (3, 260, 260, 3)"] --> B

    subgraph BASE["EfficientNetB4 Base — 19M Parameters\nImageNet pretrained · Last 30 layers unfrozen (Phase B)"]
        B["Stem Conv 3×3 Stride 2\nExtracts edges, gradients, basic texture"]
        B --> C["MBConv Blocks\n(Mobile Inverted Bottleneck)\nDepth-wise Separable Convolutions\nExtracts tumor shape and soft-tissue textures"]
        C --> D["Squeeze-and-Excitation per block\nChannel-wise attention weights\nAmplifies tumor-relevant feature maps"]
        D --> E["top_conv — Last Conv Layer\n👁️ Grad-CAM hooks here\ntf.keras.Model outputs=[top_conv.output, model.output]"]
    end

    subgraph HEAD["Custom Decision Head — Trained from Scratch"]
        E --> F["GlobalAveragePooling2D\nCollapses (H, W, C) → (C,) vector"]
        F --> G["Dense(512, activation='relu')\n+ L2 kernel_regularizer\nLearns high-level tumor representations"]
        G --> H["Dropout(0.4)\nKills 40% of neurons per batch\nForces redundant pathways"]
        H --> I["BatchNormalization\nStabilises output distribution"]
        I --> J["Dense(256, activation='relu') + L2"]
        J --> K["Dropout(0.3)"]
        K --> L["Dense(4, activation='softmax')\nOutputs: [glioma, meningioma, notumor, pituitary]"]
    end

    subgraph OUT["Output per TTA View — Shape (3, 4)"]
        L --> M["View 1: [0.94, 0.03, 0.02, 0.01]"]
        L --> N["View 2: [0.91, 0.05, 0.02, 0.02]"]
        L --> O["View 3: [0.96, 0.02, 0.01, 0.01]"]
    end

    style A fill:#1e3a5f,color:#fff
    style L fill:#6d28d9,color:#fff
    style BASE fill:#1e293b,color:#ccc
    style HEAD fill:#14532d,color:#ccc
    style OUT fill:#1c1917,color:#ccc
```

---

### Stage 4 — Statistical Consensus (`app/app.py`)

```mermaid
flowchart TD
    A["Output Shape (3, 4)\n3 probability arrays from 3 TTA views"]

    A --> B["confidences = np.mean(predictions, axis=0)\nElement-wise average across 3 views\nResult shape: (4,)"]

    B --> C["Final Probability Array\nglioma    : 0.9367\nmeningioma: 0.0333\nnotumor   : 0.0167\npituitary : 0.0133"]

    C --> D["pred_idx = int(np.argmax(confidences)) → 0\npred_class = CLASSES[0] → 'glioma'\nconfidence = float(confidences[0]) * 100 → 93.67%"]

    C --> E["breakdown = {cls: round(float(confidences[i])*100, 2)\n              for i, cls in enumerate(CLASSES)}\nSent to JS probability bar chart"]

    style A fill:#1e293b,color:#fff
    style D fill:#065f46,color:#fff
    style E fill:#1e3a5f,color:#fff
```

---

### Stage 5 — Grad-CAM Heatmap (`src/grad_cam.py`)

> Reference: Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"* (ICCV 2017)

```mermaid
flowchart TD
    A["🧠 model + tta_batch[0] expanded\nnp.expand_dims(tta_batch[0], axis=0)\nShape: (1, 260, 260, 3)"]

    A --> B["Build Grad Sub-Model\ntf.keras.models.Model(\n  inputs=model.inputs,\n  outputs=[model.get_layer('top_conv').output,\n           model.output])"]

    B --> C["with tf.GradientTape() as tape:\n  conv_outputs, predictions = grad_model(inputs)\n  pred_index = tf.argmax(predictions[0])\n  class_channel = predictions[:, pred_index]"]

    C --> D["grads = tape.gradient(class_channel, conv_outputs)\nGradients of glioma score\nw.r.t. top_conv feature maps\n→ Which pixels most activated the glioma neuron?"]

    D --> E["pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))\nOne scalar weight per feature channel"]

    E --> F["heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]\nheatmap = tf.squeeze(heatmap)\nWeighted sum of feature maps"]

    F --> G["heatmap = tf.maximum(heatmap, 0)\nReLU — keeps only positive activations\nNormalize to [0, 1]"]

    G --> H["cv2.resize(heatmap, (img_w, img_h))\nnp.uint8(255 * heatmap_resized)\ncv2.applyColorMap(heatmap_uint8, COLORMAP_JET)\nBlue=Low Activation · Red=Tumor Focus"]

    H --> I["cv2.addWeighted(img, 0.55, colored_heatmap, 0.45, 0)\nOverlay blended onto original MRI\nSaved to app/static/heatmaps/{uid}.jpg"]

    style A fill:#1e293b,color:#fff
    style D fill:#6d28d9,color:#fff
    style H fill:#7f1d1d,color:#fff
    style I fill:#065f46,color:#fff
```

---

### Stage 6 — Flask JSON Response (`app/app.py`)

```mermaid
flowchart TD
    A["✅ All stages complete"] --> B

    B["jsonify({\n  'success': True,\n  'diagnosis': 'Glioma',\n  'class': 'glioma',\n  'confidence': 93.67,\n  'color': '#ef4444',\n  'icon': '⚠️',\n  'breakdown': {'glioma':93.67,'meningioma':3.33,...},\n  'heatmap_url': '/static/heatmaps/{uid}.jpg',\n  'upload_url': '/static/uploads/{uid}.jpg'\n})"]

    B --> C["HTTP 200 → JavaScript renderResults(data)"]

    C --> D["dxStatus — TUMOR DETECTED badge"]
    C --> E["confFill — animated confidence bar to 93.67%"]
    C --> F["probRows — 4-class animated breakdown bars"]
    C --> G["gradcamImg.src = heatmap_url + ?t=timestamp\ncache-bust for fresh heatmap"]
    C --> H["risk-grid — Classification · Risk level\nGrowth rate · Recommended action"]

    style A fill:#065f46,color:#fff
    style B fill:#1e293b,color:#fff
    style C fill:#1e3a5f,color:#fff
```

---

## 📁 Project Structure

```
NeuroScan/
│
├── app/
│   ├── app.py                             # Flask server · /predict · TTA · Grad-CAM call
│   ├── templates/index.html               # Clinical dashboard (dark/light mode, mobile responsive)
│   └── static/
│       ├── confusion_matrix.png           # Verified evaluation proof shown in UI
│       ├── uploads/                       # Incoming MRI files (uuid-named)
│       └── heatmaps/                      # Grad-CAM overlays (uuid-named)
│
├── src/
│   ├── preprocess.py                      # 6-stage OpenCV pipeline (gamma, CLAHE, skull strip, crop)
│   ├── grad_cam.py                        # GradientTape heatmap on top_conv layer
│   ├── evaluate.py                        # Full 1600-image evaluation + confusion_matrix.png
│   ├── predict.py                         # CLI inference with TTA
│   ├── train_model.py                     # Local training (VGG16 baseline)
│   └── train_colab_v2.py                  # Final EfficientNetB4 Colab training (98.10% val)
│
├── models/
│   └── neuroscan_efficientnet_final.keras # Production weights · 94.88% test accuracy
│
├── results/
│   ├── confusion_matrix.png               # Generated by evaluate.py
│   └── report.txt                         # Classification report text
│
├── Dockerfile                             # Render.com deployment container
├── render.yaml                            # Render service config (free tier + persistent disk)
├── requirements.txt                       # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/rajakumar123-commit/NeuroScan.git
cd NeuroScan
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model weights
# Download neuroscan_efficientnet_final.keras
# → Copy to: models/neuroscan_efficientnet_final.keras

# 4. Launch web app
venv\Scripts\python.exe app\app.py
# Open: http://127.0.0.1:5000
```

### Run Full Evaluation

```bash
venv\Scripts\python.exe src\evaluate.py
```
```
  [      glioma] processing 400 images... ✓
  [  meningioma] processing 400 images... ✓
  [     notumor] processing 400 images... ✓
  [   pituitary] processing 400 images... ✓

  Verified Test Accuracy : 94.88%
  Total images evaluated : 1600
  Correct predictions    : 1518
```

---

## 🛡️ Viva Defense Notes

| Examiner Question | Your Answer |
|:---|:---|
| *Why a hybrid preprocessing pipeline?* | OpenCV forces the AI to analyze tumor tissue only — skull, brightness variance, and scanner artifacts are removed before inference |
| *Why EfficientNetB4 over VGG16?* | Compound Scaling (width × depth × resolution) achieves 94.88% test accuracy with 7× fewer parameters than VGG16 |
| *Why Test-Time Augmentation?* | A hospital scan can arrive at any rotation. TTA averages 3 geometric views via `np.mean(axis=0)` to produce a consensus result |
| *Why Grad-CAM on `top_conv`?* | `top_conv` is the last spatial feature map before GlobalAveragePooling. Gradients here show exactly which spatial regions caused the prediction |
| *Why Focal Loss?* | `CategoricalFocalCrossentropy` penalises hard examples more — specifically helps with Glioma's irregular boundary |
| *Why class_weight glioma=1.5?* | Glioma had the lowest recall (83%). Increasing its penalty forces the model to take Glioma misclassifications more seriously |
| *Why Phase A then Phase B?* | Phase A trains only the custom head to prevent Catastrophic Forgetting of ImageNet edge detectors. Phase B unfreezes last 30 layers to adapt them to MRI data |
| *Why Dropout 0.4 + 0.3?* | Forces redundant neural pathways — kills memorization of training data patterns |

> **Viva Statement:** *"We validated the model on a completely unseen test set of 1600 MRI images (400 per class). The model achieved 94.88% accuracy, and Grad-CAM confirms that predictions are based on tumor regions — not skull, background, or artifacts."*

---

## ⚠️ Model Limitations

> Honest acknowledgement of limitations is a hallmark of rigorous research.

| Limitation | Detail |
|:---|:---|
| **Glioma recall is lower (83%)** | Glioma tumors have highly irregular morphology and variable boundaries — the hardest class in MRI classification literature |
| **2D slice classification only** | The model classifies individual 2D MRI slices, not full 3D volumetric scans (DICOM/NIfTI). Tumors span multiple slices — a single slice may be ambiguous |
| **No DICOM metadata** | Patient metadata (age, symptoms, prior scans) is not incorporated — a real clinical system would use multimodal inputs |
| **Dataset domain** | Trained on the Kaggle Brain Tumor MRI dataset — performance on MRI scans from different hospital scanners or imaging protocols may vary |
| **Decision-support only** | This system must not replace a qualified radiologist or neuro-oncologist diagnosis |

---

### 🔍 Failure Case Analysis

Glioma misclassifications were observed primarily in cases with:
- **Diffuse tumor boundaries** — irregular margins that blend with surrounding tissue
- **Low contrast regions** — tumors with T1/T2 intensity similar to adjacent brain matter
- **Overlap with normal tissue intensity** — particularly in early-stage or infiltrative gliomas

This aligns with known challenges in MRI-based tumor classification literature and motivates the use of:
- Class weighting (`glioma: 1.5`) to increase sensitivity
- Focal Loss to focus training on hard examples
- The uncertainty flag (`confidence < 85%`) to alert clinicians

---

## 🔮 Future Improvements

| Enhancement | Description |
|:---|:---|
| Vision Transformers (ViT) | Global self-attention across brain patches for long-range spatial reasoning |
| 3D CNN on DICOM volumes | Volumetric tumor analysis from full NIfTI MRI stacks in mm³ |
| Model Ensembling | EfficientNetB4 + DenseNet201 vote consensus for higher accuracy |
| DICOM Import | Direct integration with hospital PACS/RIS systems |

---

<div align="center">

*This system is intended as a decision-support tool, not a diagnostic replacement.*

<br/>

Built for the **NeuroScan Medical AI Project** &nbsp;·&nbsp; EfficientNetB4 · OpenCV · Flask · Grad-CAM · TTA

[![GitHub](https://img.shields.io/badge/GitHub-rajakumar123--commit/NeuroScan-181717?style=for-the-badge&logo=github)](https://github.com/rajakumar123-commit/NeuroScan)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a3a5c,100:0f0e0d&height=100&section=footer" width="100%"/>

</div>
