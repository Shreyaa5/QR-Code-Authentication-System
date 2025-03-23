import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from feature_extraction import extract_features
from data_preprocessing import load_image, original_path, counterfeit_path

# Load dataset
X, y = [], []


if not os.path.exists(original_path):
    raise FileNotFoundError(f"Path not found: {original_path}")
for filename in os.listdir(original_path):
    img = load_image(os.path.join(original_path, filename))
    X.append(extract_features(img))
    y.append(0)  # Label for original

for filename in os.listdir(counterfeit_path):
    img = load_image(os.path.join(counterfeit_path, filename))
    X.append(extract_features(img))
    y.append(1)  # Label for counterfeit

X = np.array(X, dtype=np.float32)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, "svm_model.pkl")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "rf_model.pkl")

print("Models trained and saved successfully!")
