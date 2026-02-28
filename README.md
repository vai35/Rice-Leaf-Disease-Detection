# 🌾 Rice Leaf Disease Classification

> A deep learning project to automatically classify rice plant diseases using CNN and Transfer Learning (EfficientNetV2B3), achieving **95.65% validation accuracy**.

---

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Results & Comparison](#results--comparison)
- [Key Techniques](#key-techniques)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Conclusion](#conclusion)

---

## 🧩 Problem Statement

Rice is one of the world's most important food crops. Leaf diseases, if undetected, can cause massive yield losses. Manual identification is time-consuming and error-prone. This project builds an automated image classification system to detect **3 types of rice leaf diseases** from photos, enabling early intervention.

**Target Classes:**
| Class | Description |
|-------|-------------|
| 🦠 Bacterial Leaf Blight | Causes water-soaked lesions on leaf edges |
| 🟤 Brown Spot | Produces small brown circular lesions |
| ⚫ Leaf Smut | Creates black, raised spots on leaf surfaces |

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| Total Images | 119 |
| Train Split | 96 images (80%) |
| Validation Split | 23 images (20%) |
| Image Size | 224 × 224 pixels |
| Batch Size | 16 |
| Classes | 3 |
| Source | DataMites Capstone Project |

> ⚠️ **Challenge:** The small dataset size (119 images) makes generalization difficult, requiring heavy use of data augmentation and transfer learning.

---

## 🔄 Project Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROJECT PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
  │  Data        │     │  Data        │     │  Baseline CNN    │
  │  Loading &   │────▶│  Augmentation│────▶│  (No Transfer    │
  │  Splitting   │     │              │     │   Learning)      │
  └──────────────┘     └──────────────┘     └────────┬─────────┘
                                                      │
                                                      │ Overfitting
                                                      │ ~35% Accuracy
                                                      ▼
  ┌──────────────────────────────────────────────────────────────┐
  │               TRANSFER LEARNING PHASE                        │
  │                                                              │
  │  MobileNetV2 ──┐                                             │
  │  MobileNetV3   │                                             │
  │  EfficientNetB0├──▶  Benchmark All Models ──▶ Select Best   │
  │  ResNet50      │                                             │
  │  InceptionV3   │                                             │
  │  DenseNet121   │                                             │
  └────────────────┘                                             │
                                                      ▼
  ┌──────────────────────────────────────────────────────────────┐
  │              FINE-TUNING PHASE (EfficientNetV2B3)            │
  │                                                              │
  │  • Unfreeze top 25% of layers                                │
  │  • Adam optimizer (lr = 1e-4)                                │
  │  • Early Stopping (patience=10)                              │
  │  • BatchNormalization + Dropout                              │
  └──────────────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
                                          ┌─────────────────────┐
                                          │  Final Model        │
                                          │  95.65% Val Acc     │
                                          └─────────────────────┘
```

---

## 🏗️ Model Architecture

### Baseline CNN (Custom)

```
Input (224×224×3)
       │
       ▼
┌─────────────────┐
│  Rescaling      │  (1/255 normalization)
│  (Data Aug.)    │  RandomFlip, RandomRotation, RandomZoom, RandomContrast
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Conv2D(32)     │  3×3, ReLU
│  MaxPooling2D   │  2×2
└────────┬────────┘
         ▼
┌─────────────────┐
│  Conv2D(64)     │  3×3, ReLU
│  MaxPooling2D   │  2×2
└────────┬────────┘
         ▼
┌─────────────────┐
│  Conv2D(128)    │  3×3, ReLU
│  MaxPooling2D   │  2×2
└────────┬────────┘
         ▼
┌─────────────────┐
│  Flatten        │
│  Dense(128)     │  ReLU
│  Dropout        │
│  Dense(3)       │  Softmax
└─────────────────┘
```

### Transfer Learning Architecture (EfficientNetV2B3)

```
Input (224×224×3)
       │
       ▼
┌─────────────────────┐
│  Data Augmentation  │  RandomFlip, RandomRotation, RandomZoom, RandomContrast
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Preprocessing      │  EfficientNet-specific normalization
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  EfficientNetV2B3   │  Pretrained on ImageNet
│  (Base Model)       │  Top 25% layers unfrozen for fine-tuning
│  Frozen: 75%        │
│  Trainable: 25%     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ GlobalAveragePool2D │
│ BatchNormalization  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Dense(256, ReLU)   │
│  Dropout(0.2)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Dense(128, ReLU)   │
│  Dropout(0.2)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Dense(3, Softmax)  │  Output: 3 disease classes
└─────────────────────┘
```

---

## 📊 Results & Comparison

### Phase 1 — Baseline & Augmentation

| Model | Val Accuracy | Notes |
|-------|-------------|-------|
| Baseline CNN (No Aug) | ~72–82% | Overfitting, unstable |
| CNN + Data Augmentation | ~39% | Slow convergence |
| CNN + Aug + ReduceLROnPlateau | ~22% | Struggling with small data |

### Phase 2 — Transfer Learning Benchmark (All Models)

| Model | Best Val Accuracy | Training Time |
|-------|------------------|---------------|
| MobileNetV3Large | 91.30% | ~0.64 mins |
| **EfficientNetB0** | **91.30%** | **~0.45 mins** ⚡ |
| ResNet50 | 91.30% | ~0.67 mins |
| InceptionV3 | 60.87% | ~0.59 mins |
| DenseNet121 | 78.26% | ~0.76 mins |

> 💡 **EfficientNetB0** selected as the best base — highest accuracy with lowest training time.

### Phase 3 — EfficientNet Family Fine-Tuning

| Model | Best Val Accuracy | Training Time |
|-------|------------------|---------------|
| EfficientNetV2B0 | 86.96% | ~0.89 mins |
| EfficientNetV2B1 | 82.61% | ~0.78 mins |
| EfficientNetV2B2 | 91.30% | ~0.83 mins |
| **EfficientNetV2B3** | **95.65%** | **~0.98 mins** 🏆 |

### Final Model — EfficientNetV2B3 (Fine-Tuned)

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **95.65%** |
| Trainable Parameters | 6,239,801 |
| Non-Trainable Parameters | 7,123,720 |
| Total Parameters | 13,363,521 |
| Optimizer | Adam (lr = 1e-4) |
| Fine-Tuned Layers | Top 25% |

---

## 🔧 Key Techniques

### Data Augmentation

```python
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.10)
])
```

| Technique | Value | Purpose |
|-----------|-------|---------|
| Random Horizontal Flip | Enabled | Simulate different orientations |
| Random Rotation | ±12% | Rotational invariance |
| Random Zoom | ±15% | Scale invariance |
| Random Contrast | ±10% | Lighting variation |

### Transfer Learning Strategy

```
ImageNet Pretrained Weights
           │
           ▼
    Feature Extraction          ← Freeze all layers first
           │
           ▼
    Fine-Tuning                 ← Unfreeze top 25% of layers
           │
           ▼
    Domain Adaptation           ← Train on rice disease images
```

### Training Callbacks

| Callback | Configuration | Purpose |
|----------|--------------|---------|
| EarlyStopping | patience=10, monitor=val_accuracy | Prevent overfitting |
| ReduceLROnPlateau | patience=5, factor=0.5 | Adaptive learning rate |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/vai35/rice-leaf-disease-classification.git
cd rice-leaf-disease-classification

# Install dependencies
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
```

### Requirements

| Library | Version |
|---------|---------|
| TensorFlow | ≥ 2.10 |
| Keras | ≥ 2.10 |
| NumPy | ≥ 1.21 |
| Matplotlib | ≥ 3.5 |
| Seaborn | ≥ 0.12 |
| scikit-learn | ≥ 1.0 |

---

## 🚀 Usage

```python
# Load and preprocess image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

img = load_img("leaf_image.jpg", target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
model = tf.keras.models.load_model("efficientnetv2b3_rice.h5")
predictions = model.predict(img_array)
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']
predicted_class = class_names[np.argmax(predictions)]
print(f"Predicted Disease: {predicted_class}")
```

---

## 📁 Project Structure

```
rice-leaf-disease-classification/
│
├── 📓 PRCP_1001_RiceLeaf.ipynb       # Main notebook
├── 📄 README.md                       # Project documentation
│
├── 📂 data/
│   ├── Bacterial leaf blight/         # Class 1 images
│   ├── Brown spot/                    # Class 2 images
│   └── Leaf smut/                     # Class 3 images
│
└── 📂 models/
    └── efficientnetv2b3_rice.h5       # Saved final model
```

---

## 🔍 Why EfficientNetV2B3 Won

```
Model Selection Decision Tree
           │
           ▼
    High Accuracy?
    ┌────────┴────────┐
   YES               NO
    │                 │
    ▼                 ▼
Fast Training?    Eliminated
    │
    ├── EfficientNetB0 (91.30%, fastest)
    │
    ▼
Best in Family?
    │
    └── EfficientNetV2B3 (95.65%) ✅ WINNER
        • Largest receptive field
        • Best feature extraction
        • Compound scaling advantage
```

---

## 📈 Conclusion

| Approach | Val Accuracy | Verdict |
|----------|-------------|---------|
| Baseline CNN | ~35% | ❌ Insufficient |
| CNN + Augmentation | ~39% | ❌ Still weak |
| MobileNetV2 (Transfer) | ~86% | ✅ Good |
| EfficientNetB0 (Transfer) | 91.30% | ✅ Better |
| **EfficientNetV2B3 (Fine-Tuned)** | **95.65%** | 🏆 **Best** |

Transfer learning with EfficientNetV2B3, combined with data augmentation and fine-tuning of the top 25% layers, proved to be the most effective strategy for this small-dataset image classification problem. The model successfully learned discriminative features for all three rice leaf diseases.

---

## 🔮 Future Scope

- 📊 **Expand Dataset** — Collect more images per class to improve generalization
- 🔍 **Grad-CAM Visualization** — Highlight which leaf regions drive predictions
- 📱 **Mobile Deployment** — Convert to TensorFlow Lite for on-field use
- 🌐 **Web App** — Deploy via Flask/FastAPI for farmer-facing diagnostics
- 🤖 **More Classes** — Extend to detect additional rice diseases

---

## 👤 Author

**[Vaishnavi Shidling]**
- 🔗 LinkedIn: [https://www.linkedin.com/in/vaishnavi-shidling/]
- 💻 GitHub: [https://github.com/vai35/]
- 📧 Email: [vaishnavishidling74@gmail.com]

---

*Built as part of the DataMites Capstone Project — PRCP-1001*
