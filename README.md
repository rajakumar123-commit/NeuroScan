# 🧠 NeuroScan AI — Brain Tumor Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-97.74%25-22c55e?style=for-the-badge)
![Classes](https://img.shields.io/badge/Classification-4_Classes-f59e0b?style=for-the-badge)

**A production-grade, hybrid deep learning pipeline for MRI-based brain tumor classification.**

[🚀 Quick Start](#-quick-start) · [📐 Architecture](#-complete-under-the-hood-pipeline) · [📊 Results](#-results--performance) · [🛡️ Defense Notes](#️-viva-defense-notes)

</div>

---

## 📌 Project Overview

| Feature | Detail |
|---|---|
| **Task** | 4-Class Medical Classification |
| **Classes** | Glioma · Meningioma · Pituitary · No Tumor |
| **Architecture** | EfficientNetB4 + Custom Decision Head |
| **Preprocessing** | 6-Stage OpenCV Hybrid Pipeline |
| **Inference** | 3-Angle Test-Time Augmentation (TTA) |
| **Explainability** | Grad-CAM Thermal Heatmaps |
| **Backend** | Python Flask REST API |
| **Test Accuracy** | **97.74%** on 840 unseen MRI images |

---

## 🔬 Complete Under-The-Hood Pipeline

> This section shows **exactly** what happens from the millisecond a doctor uploads an MRI to the moment the final answer appears on the screen.

---

### 📍 STAGE 0 — Doctor Uploads the Image (Browser → Flask)

```mermaid
sequenceDiagram
    actor Doctor as 👨‍⚕️ Doctor
    participant UI as 🖥️ Web Dashboard
    participant Flask as 🐍 Flask /predict

    Doctor->>UI: Drag & Drop MRI scan (.jpg/.png)
    UI->>UI: FileReader → shows image preview
    Doctor->>UI: Clicks "Run Diagnosis" button
    UI->>Flask: POST /predict (multipart/form-data)
    Flask->>Flask: Validates file extension
    Flask->>Flask: uuid.uuid4() → saves to static/uploads/
    Flask-->>UI: ✅ File received, pipeline starts
```

---

### 📍 STAGE 1 — OpenCV 6-Stage Diagnostic Filtration

> The raw MRI is intercepted and cleaned **before** the AI ever sees it. If the brain cannot be detected, the image is rejected.

```mermaid
flowchart TD
    A["🩻 Raw MRI Image\nJPG / PNG / BMP\nAny brightness, any scanner"]
    A --> B

    subgraph STAGE1["🔬 STAGE 1 — OpenCV Preprocessing Pipeline"]
        B["Step 1: Median Blur\ncv2.medianBlur kernel=3\n👉 Removes salt-and-pepper scanner noise"]
        B --> C["Step 2: Adaptive Gamma Correction\n👉 Finds average brightness\n👉 Recalculates gamma so all scanners look equal\n👉 Prevents dark/bright bias"]
        C --> D["Step 3: CLAHE\nContrast Limited Adaptive Histogram Equalization\n👉 Divides image into 8×8 grids\n👉 Boosts contrast in each grid independently\n👉 White/gray brain matter becomes sharper"]
        D --> E["Step 4: Otsu Thresholding\ncv2.threshold THRESH_OTSU\n👉 Auto-calculates the perfect pixel cutoff\n👉 Skull = white (255) | Background = black (0)"]
        E --> F["Step 5: Skull Stripping\ncv2.morphologyEx MORPH_CLOSE\n👉 Fills tiny black holes in the skull mask\n👉 Applies mask → skull pixels become 0\n👉 Only soft brain tissue remains"]
        F --> G["Step 6: Contour Crop + Gray Padding\ncv2.findContours → selects largest blob\n👉 Crops tight bounding box around brain\n👉 Pads short side with neutral gray (not black)\n👉 Output: Perfect square — no squishing"]
    end

    G --> H["✅ Clean Brain Image\n260×260×3 numpy array\nZero skull · Zero artifacts · Balanced brightness"]

    subgraph GATE["🛑 Quality Gate"]
        QG["If no brain contour found →\nReturn Error 422 to UI\nDo NOT send garbage to AI"]
    end

    F --> GATE

    style A fill:#374151,color:#fff
    style H fill:#065f46,color:#fff
    style GATE fill:#7f1d1d,color:#fff
    style STAGE1 fill:#1e293b,color:#fff
```

---

### 📍 STAGE 2 — Test-Time Augmentation (The 3-Angle Double-Tap)

> The cleaned image is **never** evaluated just once. It is forked into 3 geometric variants to eliminate the risk of a badly-angled hospital scan breaking the AI.

```mermaid
flowchart TD
    A["✅ Clean Brain Image\n260×260×3"]

    A --> B["View 1: Normal\nOriginal cleaned image\n→ img_normal"]
    A --> C["View 2: Horizontal Mirror\ncv2.flip img 1\n→ img_hf"]
    A --> D["View 3: Vertical Flip\ncv2.flip img 0\n→ img_vf"]

    subgraph PREPROCESS["⚙️ EfficientNet Preprocessing"]
        B --> E["efficientnet.preprocess_input\nScales pixels to EfficientNet's\nexpected internal range"]
        C --> F["efficientnet.preprocess_input"]
        D --> G["efficientnet.preprocess_input"]
    end

    subgraph BATCH["📦 Batch Assembly"]
        E --> H["numpy array shape\n3 × 260 × 260 × 3\nAll 3 views bundled together"]
        F --> H
        G --> H
    end

    H --> I["🧠 EfficientNetB4\nmodel.predict batch\nAll 3 evaluated simultaneously"]

    style A fill:#1e3a5f,color:#fff
    style I fill:#6d28d9,color:#fff
    style BATCH fill:#1e293b,color:#fff
    style PREPROCESS fill:#1e293b,color:#fff
```

---

### 📍 STAGE 3 — EfficientNetB4 Under The Hood

> This shows what physically happens inside the 482-layer neural network when our image batch enters it.

```mermaid
flowchart TD
    A["📦 TTA Batch\n3 × 260 × 260 × 3"] --> B

    subgraph BASE["🏗️ EfficientNetB4 Base Network — 19M Parameters"]
        B["Stem Convolution\n3×3 Conv, Stride 2\nExtracts basic edges and gradients"]
        B --> C["MBConv Blocks × 32\n Mobile Inverted Bottleneck Blocks\nDepth-wise Separable Convolutions\nExtracts textures, curves, and soft tissue patterns"]
        C --> D["Squeeze-and-Excitation Blocks\nReweights important feature channels\nGives more attention to relevant tumor features"]
        D --> E["Top Convolution Layer\ntop_conv\n👁️ Grad-CAM hooks here\nFinal spatial feature map before pooling"]
    end

    subgraph HEAD["🎯 Custom Decision Head — Trained by Us"]
        E --> F["GlobalAveragePooling2D\nCollapses 2D feature map → 1D vector\nRemoves spatial information"]
        F --> G["Dense 512 neurons\nReLU activation + L2 Regularization\nLearns complex tumor representations"]
        G --> H["Dropout 0.4\nRandomly kills 40% of neurons during training\nForces redundant pathways — prevents memorization"]
        H --> I["BatchNormalization\nStabilizes neuron output distribution\nSpeeds convergence"]
        I --> J["Dense 256 neurons\nReLU + L2 Regularization\nRefines the representation further"]
        J --> K["Dropout 0.3"]
        K --> L["Dense 4 neurons\nSoftmax Activation\nOutputs probability for each class"]
    end

    subgraph OUTPUT["📊 Output per View — Shape 3 × 4"]
        L --> M["View 1: Glioma=0.94 Mening=0.03 NoTumor=0.02 Pituit=0.01"]
        L --> N["View 2: Glioma=0.91 Mening=0.05 NoTumor=0.02 Pituit=0.02"]
        L --> O["View 3: Glioma=0.96 Mening=0.02 NoTumor=0.01 Pituit=0.01"]
    end

    style A fill:#1e3a5f,color:#fff
    style L fill:#6d28d9,color:#fff
    style BASE fill:#1e293b,color:#fff
    style HEAD fill:#14532d,color:#fff
    style OUTPUT fill:#1c1917,color:#fff
```

---

### 📍 STAGE 4 — Statistical Averaging & Final Classification

```mermaid
flowchart TD
    A["📊 3 Probability Arrays\nShape: 3 × 4"]

    A --> B["np.mean predictions axis=0\nMathematically averages the 3 arrays\ninto a single consensus array"]

    B --> C["Final Probability Array\nGlioma: 0.9367\nMeningioma: 0.0333\nNo Tumor: 0.0167\nPituitary: 0.0133"]

    C --> D["np.argmax confidences\nFinds the index of the highest value"]

    D --> E["🏆 FINAL DIAGNOSIS\npred_class = GLIOMA\nconfidence = 93.67%"]

    C --> F["Full Breakdown Dictionary\nAll 4 class % values\nSent to frontend bar chart"]

    style A fill:#1e293b,color:#fff
    style E fill:#065f46,color:#fff
    style F fill:#1e3a5f,color:#fff
```

---

### 📍 STAGE 5 — Grad-CAM Heatmap Generation

> Using calculus, we reverse-engineer the model to prove exactly *where* it was looking.

```mermaid
flowchart TD
    A["🧠 EfficientNetB4 Model\n+ Preprocessed Image\n+ Predicted Class Index"]

    A --> B["Build Grad Model\ntf.keras.Model\nInputs: original input\nOutputs: top_conv layer + final predictions"]

    B --> C["tf.GradientTape\nRecord all mathematical operations\nas image passes through model"]

    C --> D["Extract Gradients\nGrad of GLIOMA output neuron\nw.r.t. top_conv feature map activations\n👉 Which pixels most changed the Glioma score?"]

    D --> E["Global Average Pool the Gradients\nnp.mean grads axis 0 1\nOne importance weight per feature channel"]

    E --> F["Weighted Feature Map\nMultiply each feature map channel\nby its importance weight\nSum all weighted channels together"]

    F --> G["ReLU Filter\nnp.maximum heatmap 0\nOnly keep positive activations\nNegative = not contributing to this class"]

    G --> H["Resize Heatmap\ncv2.resize to 260×260\nMatches original image dimensions"]

    H --> I["Normalize 0-255\nScales activation values\nto full visible color range"]

    I --> J["Apply Color Map\ncv2.applyColorMap COLORMAP_JET\nBlue = Low Activation\nRed = High Activation = Tumor Focus"]

    J --> K["Overlay on Original\ncv2.addWeighted\nHeatmap blended onto clean brain image\n0.4 opacity"]

    K --> L["🌡️ Final Thermal Heatmap\nSaved to static/heatmaps/\nURL returned in JSON response"]

    style A fill:#1e293b,color:#fff
    style D fill:#6d28d9,color:#fff
    style J fill:#7f1d1d,color:#fff
    style L fill:#065f46,color:#fff
```

---

### 📍 STAGE 6 — Flask Packages & Returns the Final Response

```mermaid
flowchart TD
    A["✅ All Stages Complete\nDiagnosis · Confidence · Breakdown · Heatmap URL"]

    A --> B["Build JSON Response\n{\n  diagnosis: GLIOMA\n  confidence: 93.67\n  color: #ef4444\n  icon: 🔴\n  breakdown: {glioma:93.67, mening:3.33...}\n  heatmap_url: /static/heatmaps/abc123.jpg\n  description: Clinical description text\n  severity: HIGH\n}"]

    B --> C["Flask jsonify response\nHTTP 200 OK\nContent-Type: application/json"]

    C --> D["🖥️ JavaScript Frontend Receives JSON"]

    D --> E["Updates Diagnosis Badge\nColor + Icon + Label"]
    D --> F["Animates Confidence Bar\n93.67% fills smoothly"]
    D --> G["Renders 4-Class Breakdown\nAnimated bar charts per class"]
    D --> H["Loads Heatmap Image\nFrom URL with cache-bust ?t=timestamp"]
    D --> I["Highlights Pipeline Steps\nAll 8 steps glow green"]
    D --> J["Smooth scrolls to Results\nscrollIntoView behavior smooth"]

    style A fill:#065f46,color:#fff
    style B fill:#1e293b,color:#fff
    style D fill:#1e3a5f,color:#fff
```

---

### 📍 COMPLETE END-TO-END SUMMARY FLOW

```mermaid
flowchart TD
    START["👨‍⚕️ Doctor Uploads MRI"] --> S0

    subgraph S0["Stage 0 — Web Layer"]
        W1["fetch POST /predict\nFormData with image file"]
        W1 --> W2["Flask saves file\nuuid filename"]
    end

    S0 --> S1

    subgraph S1["Stage 1 — OpenCV Preprocessing"]
        P1["Median Blur"] --> P2["Adaptive Gamma"]
        P2 --> P3["CLAHE Enhancement"]
        P3 --> P4["Otsu Threshold"]
        P4 --> P5["Skull Stripping"]
        P5 --> P6["Contour Crop + Padding\n→ 260×260×3 clean array"]
    end

    S1 -->|"❌ Brain not found"| ERR["Return Error 422\nto UI"]
    S1 -->|"✅ Brain detected"| S2

    subgraph S2["Stage 2 — TTA Generation"]
        T1["Normal View"]
        T2["Horizontal Flip"]
        T3["Vertical Flip"]
        T1 & T2 & T3 --> T4["efficientnet.preprocess_input\n3 × 260 × 260 × 3 batch"]
    end

    S2 --> S3

    subgraph S3["Stage 3 — EfficientNetB4 Inference"]
        N1["482 Layers\nMBConv Blocks\nSqueeze-and-Excitation"]
        N1 --> N2["Custom Head\nDense 512 → 256 → 4"]
        N2 --> N3["Softmax Output\n3 × 4 probability arrays"]
    end

    S3 --> S4

    subgraph S4["Stage 4 — Statistical Consensus"]
        M1["np.mean axis=0\nAverage 3 probability arrays"]
        M1 --> M2["np.argmax\nSelect winning class"]
        M2 --> M3["FINAL DIAGNOSIS\n+ Confidence Score"]
    end

    S4 --> S5

    subgraph S5["Stage 5 — Grad-CAM"]
        G1["GradientTape\nGradients of top_conv"]
        G1 --> G2["Weighted Feature Map\n+ ReLU Filter"]
        G2 --> G3["COLORMAP_JET Overlay\nThermal Heatmap Image"]
    end

    S5 --> S6

    subgraph S6["Stage 6 — Response Assembly"]
        R1["JSON: diagnosis + confidence\n+ breakdown + heatmap_url"]
        R1 --> R2["HTTP 200 → JavaScript\nAnimates Dashboard UI"]
    end

    S6 --> END["✅ Doctor sees:\nDiagnosis · Confidence\nClass Breakdown · Heatmap"]

    style START fill:#1e3a5f,color:#fff
    style END fill:#065f46,color:#fff
    style ERR fill:#7f1d1d,color:#fff
    style S0 fill:#1e293b,color:#aaa
    style S1 fill:#1e293b,color:#aaa
    style S2 fill:#1e293b,color:#aaa
    style S3 fill:#1e293b,color:#aaa
    style S4 fill:#1e293b,color:#aaa
    style S5 fill:#1e293b,color:#aaa
    style S6 fill:#1e293b,color:#aaa
```

---

## 📁 Project Structure

```
NeuroScan/
│
├── app/
│   ├── app.py                           # Flask server + /predict route + TTA logic
│   ├── templates/index.html             # Clinical dark-mode dashboard
│   └── static/
│       ├── uploads/                     # Incoming MRI files
│       └── heatmaps/                    # Grad-CAM outputs
│
├── src/
│   ├── preprocess.py                    # 6-Stage OpenCV Preprocessor
│   ├── predict.py                       # CLI inference with TTA
│   ├── grad_cam.py                      # GradientTape heatmap engine
│   ├── train_efficientnet.py            # Phase A + B local training
│   ├── train_colab_v2.py                # Colab GPU training (final run)
│   └── split_data.py                    # Dataset split utility
│
├── models/
│   └── neuroscan_efficientnet_final.keras   # Trained model — 97.74%
│
├── dataset_cropped/
│   ├── train/                           # 6,790 augmented MRIs
│   ├── val/                             # 840 MRIs
│   └── test/                           # 840 unseen MRIs (never touched)
│
└── README.md
```

---

## 📊 Results & Performance

### Classification Report — 840 Unseen Test Images

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Glioma** | 0.98 | 0.97 | **0.97** | 210 |
| **Meningioma** | 0.94 | 0.96 | **0.95** | 210 |
| **No Tumor** | 1.00 | 1.00 | **1.00** | 210 |
| **Pituitary** | 0.99 | 0.98 | **0.99** | 210 |
| **Overall** | | | **0.98** | 840 |

### Architecture Progression

| Model | Val Accuracy | Parameters |
|---|---|---|
| VGG16 Baseline | 89.40% | 138 Million |
| VGG16 Phase C | 92.50% | 138 Million |
| EfficientNetB4 Phase A | 88.81% | 19 Million |
| **EfficientNetB4 Phase B (Final)** | **97.74%** | **19 Million** |

> **Result: 7× fewer parameters, 8.34% higher accuracy than VGG16**

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/your-username/NeuroScan.git
cd NeuroScan
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install
pip install tensorflow opencv-python flask numpy scikit-learn matplotlib seaborn

# 3. Download model → place in models/neuroscan_efficientnet_final.keras

# 4. Launch
cd app && python app.py
# Open: http://127.0.0.1:5000
```

### CLI Inference

```bash
python src/predict.py path/to/mri.jpg
```

```
=============================================
 NEUROSCAN — MRI ANALYSIS RESULTS
=============================================
  Diagnosis  : GLIOMA
  Confidence : 93.67%
---------------------------------------------
 Breakdown:
  - glioma       :  93.67%
  - meningioma   :   3.33%
  - notumor      :   1.67%
  - pituitary    :   1.33%
=============================================
```

---

## 🛡️ Viva Defense Notes

| Question | Answer Summary |
|---|---|
| *Why Hybrid Pipeline?* | OpenCV forces AI to analyze tumor tissue, not skull/artifacts |
| *Why EfficientNet over VGG16?* | Compound Scaling: 97% accuracy with 7× fewer parameters |
| *Why TTA?* | Eliminates Geometric Bias — consensus from 3 angles |
| *Why Grad-CAM?* | Clinical proof the AI looks at tumor mass, not background |
| *Why 97% not 99%?* | Honest 4-class score vs. binary Data-Leaked 99% |
| *Why Phase Training?* | Prevents Catastrophic Forgetting of ImageNet edge detectors |
| *Why Label Smoothing 0.1?* | Prevents overconfidence; improves generalization |
| *Why Dropout 0.4?* | Forces redundant pathways — kills memorization |

---

## 🔮 Future Improvements

| Enhancement | Description |
|---|---|
| **Vision Transformers (ViT)** | Global attention across brain patches |
| **3D CNNs on DICOM volumes** | Full NIfTI volumetric tumor mass in mm³ |
| **Focal Loss** | Penalty focused on hard Meningioma/Glioma confusion |
| **Model Ensembling** | EfficientNetB4 + DenseNet201 vote consensus |

---

<div align="center">

Built for the **NeuroScan Medical AI Thesis Project**  
EfficientNetB4 · OpenCV · Flask · Grad-CAM · TTA · 97.74%

</div>
