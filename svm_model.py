import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- robust parser for numeric data in 'img' strings ---
def extract_numeric_array(s: str) -> np.ndarray:
    if not isinstance(s, str):
        return np.array([], dtype=np.float32)
    s = s.replace("...", " ").replace("\\n", " ")
    tokens = re.findall(r"-?\d+\.?\d*", s)
    return np.array(tokens, dtype=np.float32) if tokens else np.array([], dtype=np.float32)

# --- load dataset ---
df = pd.read_csv("finalProcessed.csv")
print("Dataset shape:", df.shape)
print(df.head())

if not {"img", "labels"}.issubset(df.columns):
    raise ValueError("Expected columns: 'img', 'labels' in finalProcessed.csv")

# --- parse images into arrays ---
arrays = []
max_len = 0
for s in df["img"]:
    arr = extract_numeric_array(s)
    arrays.append(arr)
    if arr.size > max_len:
        max_len = arr.size

print(f"\nMax image vector length: {max_len}")

# --- pad arrays so all have equal length (for matrix training) ---
def pad_to_length(a, length):
    if a.size < length:
        return np.pad(a, (0, length - a.size), mode="constant")
    elif a.size > length:
        return a[:length]
    else:
        return a

X = np.vstack([pad_to_length(a, max_len) for a in arrays])
y = df["labels"].round().astype(int)

print("\nFinal X shape:", X.shape)
print("Labels:", np.unique(y, return_counts=True))

# --- train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- scale + PCA ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# PCA for dimensionality reduction (retain 95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)

print(f"After PCA: {X_train_p.shape[1]} components retained (95% variance)")

# --- SVM model with tuning ---
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto"],
    "class_weight": [None, "balanced"]
}

grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_p, y_train)
best_svm = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

# --- evaluate ---
y_pred = best_svm.predict(X_test_p)
acc = accuracy_score(y_test, y_pred)
print("\n🎯 SVM Accuracy:", round(acc, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- save model ---
joblib.dump(best_svm, "svm_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")
joblib.dump(pca, "svm_pca.pkl")
print("\nSaved: svm_model.pkl, svm_scaler.pkl, svm_pca.pkl")
