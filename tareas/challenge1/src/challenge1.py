import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri('http://localhost:5000')

def get_dataset():
    file_path = "../data/breast-cancer-wisconsin.data.csv"
    df_breast_cancer = pd.read_csv(file_path)
    return df_breast_cancer

def clean_data(df_breast_cancer):
    # All data in the "Unnamed: 32" column is null, remove the column.
    # Also remove the id column, it won't give much information about the target.
    df_breast_cancer = df_breast_cancer.drop(['id','Unnamed: 32'], axis=1)
    return df_breast_cancer

def x_y_split(df_breast_cancer):
    # Diagnosis is the target value, its value is either 'B' or 'M'
    # B is benign
    # M is malignant
    # Separate independent and target values
    X = df_breast_cancer.drop('diagnosis', axis=1)
    y = df_breast_cancer['diagnosis']
    return X,y

def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def get_trained_model(X_train, y_train):
    linear_regression_model = LogisticRegression()
    linear_regression_model.fit(X_train, y_train)
    mlflow.sklearn.log_model(linear_regression_model, "logistic_regression_model")
    return linear_regression_model

def evaluate_model(y_test, y_pred, y_probs):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n{cm}")
    display = ConfusionMatrixDisplay(cm)
    display.plot().figure_.savefig("confusionMatrix.png")
    mlflow.log_artifact("confusionMatrix.png")

    # Precision
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision:.4f}")
    mlflow.log_metric("Precision", precision)

    # Recall
    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall:.4f}")
    mlflow.log_metric("Recall", recall)

    # F1-Score
    f1 = f1_score(y_test, y_pred)
    print(f"F1 score: {f1:.4f}")
    mlflow.log_metric("F1Score", f1)

    # ROC curve and AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_probs[:,1])
    plt.cla()
    plt.plot(fpr, tpr, marker="o", linestyle="--")
    # Draw straight line
    x = [i*0.01 for i in range(100)]
    y = [i*0.01 for i in range(100)]
    plt.plot(x,y, color="r")

    # Calculate auc
    auc = metrics.roc_auc_score(y_test, y_probs[:,1])
    print(f"AUC: {auc:.4f}")
    mlflow.log_metric("AUC", auc)

    # Show ROC curve
    plt.title(f"Roc Curve. Auc: {auc:.4f}")
    plt.savefig("rocCurve.png")
    mlflow.log_artifact("rocCurve.png")

def main():
    # Stage 1: Get dataset
    df_breast_cancer = get_dataset()
    # Stage 2 create model
    # Clean data, remove not needed columns
    df_breast_cancer = clean_data(df_breast_cancer)
    # Split dataset in independent and target.
    X, y = x_y_split(df_breast_cancer)
    X = normalize_data(X)
    # Convert 'M' and 'B' to 1 and 0
    y = LabelEncoder().fit_transform(y)
    # Test-Train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3, shuffle=True
    )
    # Start mlflow experiment
    with mlflow.start_run(): # Stage 3: MLflow
        # Build model and train it
        model = get_trained_model(X_train, y_train)
        # Get predicted categories
        y_pred = model.predict(X_test)
        # Get predicted probabilities
        y_probs = model.predict_proba(X_test)
        # Evaluate model
        evaluate_model(y_test, y_pred, y_probs)

main()
