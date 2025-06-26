# 🪲 Pest Detection & Pesticide Recommendation System

A **PyTorch-based deep learning system** for automated pest classification and pesticide recommendation using **EfficientNet-B0**.  
Built for agricultural diagnostics and intelligent pest management.

---

## 🚀 Features

- **Multi-class Classification**: Identifies **102 pest categories** with **67.75% accuracy**
- **Smart Recommendations**: Provides **pesticide suggestions and dosages** via CSV mapping
- **Robust Training**: Includes **data augmentation**, **dropout regularization**, and **early stopping**
- **Performance Analytics**: Generates **confusion matrices**, **ROC curves**, and **precision-recall analysis**

---

## 🧠 Model Architecture

- **Base Model**: `EfficientNet-B0` (pretrained on ImageNet)
- **Custom Head**:
  - Dropout: `p = 0.5`
  - Fully Connected Layer: `output = 102 classes`
- **Optimizer**: `Adam (lr=5e-4, weight_decay=1e-4)`
- **Scheduler**: `ReduceLROnPlateau` with early stopping mechanism

---

## 📊 Performance Metrics

| Metric     | Score   |
|------------|---------|
| Accuracy   | 67.75%  |
| Precision  | 63.30%  |
| Recall     | 58.49%  |
| F1 Score   | 59.86%  |

---

## 🧪 Data Processing

- **Augmentation**:
  - Resize: `224 × 224`
  - Random horizontal/vertical flip
  - Random rotation and color jitter
- **Normalization**:
  - `mean = [0.5, 0.5, 0.5]`
  - `std = [0.5, 0.5, 0.5]`
- **Split**:
  - `80%` training
  - `20%` validation

---

## 🛠️ Usage Examples

### 📈 Training Progress

- `Epoch 1`: 47.01% accuracy  
- `Epoch 10`: 81.31% accuracy (on training set)
- **Early stopping** to prevent overfitting
- **Model checkpointing** for best model saving

### 🧾 Classification Output

- **Predicted Pest**: Name with confidence %
- **Pesticide Recommendation**:
  - Name of recommended pesticide
  - Dosage information from CSV mapping
- **Visual Output**:
  - Original image display
  - Overlaid predictions and suggestions

---

## 🌾 Purpose

This system is developed to support **automated pest identification** and **precision pesticide usage**, enhancing productivity and reducing excessive pesticide use in agriculture.

---

## 📂 Project Structure (Suggested)

