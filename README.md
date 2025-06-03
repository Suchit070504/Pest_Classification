This project evaluates the performance of a trained PyTorch image classification model on a test dataset. It computes key classification metrics and visualizes model behavior across the top 10 most confidently predicted categories using ROC and Precision-Recall curves.

📁 Dataset
The test dataset is expected to be organized in the following structure:

/classification/test/
    ├── category_1/
    │   ├── image1.jpg
    │   └── ...
    ├── category_2/
    └── ...
Update the dataset path if needed:

TEST_DATA_PATH = "/kaggle/input/ip02-dataset/classification/test"
⚙️ Requirements
Python 3.8+

PyTorch

scikit-learn

torchvision

matplotlib

seaborn

PIL

Install via pip:

🧩 Project Components
1. Test DataLoader
Custom Dataset class loads and transforms test images.

Labels are automatically inferred from folder names.

2. Model Evaluation
Evaluates the model using:

Accuracy

Precision (macro average)

Recall (macro average)

F1 Score (macro average)

3. Top 10 Class Selection
Selects top 10 most confident classes based on simulated prediction match counts. (This can be replaced with actual values.)

4. ROC Curve Plotting
Plots ROC curves and AUC scores for each of the selected 10 categories.

5. Precision-Recall Curve
Plots Precision-Recall curves with Average Precision (AP) for each top class.

🚀 Running the Evaluation
Make sure you have a trained model and define the correct model architecture beforehand.

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
Run the script to:

Load test data

Evaluate performance metrics

Plot ROC and PR curves

📊 Output
✅ Model Evaluation Metrics on test data.

📈 ROC Curve plot with AUC for each top 10 class.

📉 Precision-Recall Curve plot with AP scores.

📌 Future Improvements
Use actual model predictions to select top 10 categories dynamically.

Add confusion matrix and misclassified sample visualization.

Save plots and metrics to files for report generation.

📎 License
This project is intended for academic and research purposes.
