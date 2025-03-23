import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from train_ml import X_test, y_test

# Load models
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# Predictions
svm_preds = svm_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Evaluation
print("SVM Model Performance:")
print(classification_report(y_test, svm_preds))

print("Random Forest Model Performance:")
print(classification_report(y_test, rf_preds))

print("Confusion Matrix (SVM):")
print(confusion_matrix(y_test, svm_preds))
