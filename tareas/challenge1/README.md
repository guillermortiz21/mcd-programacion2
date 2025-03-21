This is a challenge to predict if a breast cancer is malignant or benign.

The data comes from a Breast Cancer Wisconsin (Diagnostic) dataset:
<a href="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic" target="_blank">Dataset</a>

I solved the challenge using a Logistic Regression, getting the following results:

Confusion Matrix:

106 | 2

1   | 62

Precision: 0.9688

Recall: 0.9841

F1 Score: 0.9764

AUC: 0.9981

The python code also logs the model and the results using an MLflow pipeline.

To run the code you need these dependencies:
- mlflow==2.21.0
- cloudpickle==3.1.1
- numpy==1.26.4
- pandas==2.2.3
- psutil==6.1.1
- scikit-learn==1.6.1
- scipy==1.15.1
- matplotlib


You can download from here, and run it with the command python challenge1.py
