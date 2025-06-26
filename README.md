**🪲 Pest Detection & Pesticide Recommendation System**

A PyTorch-based deep learning system for automated pest classification and pesticide recommendation using EfficientNet-B0.

**Features**
Multi-class Classification: Identifies 102 pest categories with 67.75% accuracy
Smart Recommendations: Provides pesticide suggestions and dosages via CSV mapping
Robust Training: Includes data augmentation, dropout regularization, and early stopping
Performance Analytics: Confusion matrices, ROC curves, and precision-recall analysis

**Model Architecture**
Base: EfficientNet-B0 (ImageNet pretrained)
Custom Head: Dropout (0.5) + FC layer for 102 classes
Optimizer: Adam (lr=5e-4, weight_decay=1e-4)
Scheduler: ReduceLROnPlateau with early stopping

**Performance Metrics**
Metric	   Score
Accuracy	67.75%
Precision	63.30%
Recall	    58.49%
F1 Score	59.86%

**Data Processing**
Augmentation: Resize (224×224), random flip/rotation, color jitter
Normalization: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
Split: 80% train, 20% validation

**Usage Examples**
**Training Progress**
Epoch 1: 47.01% → Epoch 10: 81.31% accuracy
Early stopping prevents overfitting
Automatic model saving

**Classification Output**
Pest name with confidence percentage
Recommended pesticide and dosage
Visual display with original image

Built for agricultural diagnostics and pest management automation.
