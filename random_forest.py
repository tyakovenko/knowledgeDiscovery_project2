import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- robust parser for numeric data in 'img' strings ---
def extract_numeric_array(s: str) -> np.ndarray:
    """
    Extract numeric tokens from a stringified image array.
    Handles artifacts like '...', '\\n', brackets.
    """
    if not isinstance(s, str):
        return np.array([], dtype=np.float32)
    s = s.replace("...", " ").replace("\\n", " ")
    tokens = re.findall(r"-?\d+\.?\d*", s)
    return np.array(tokens, dtype=np.float32) if tokens else np.array([], dtype=np.float32)


# --- Load dataset ---
df = pd.read_csv("finalProcessed.csv")
print("Dataset shape:", df.shape)
print(df.head())

if not {"img", "labels"}.issubset(df.columns):
    raise ValueError("Expected columns: 'img', 'labels' in finalProcessed.csv")

# --- Parse 'img' column into numeric arrays ---
arrays = []
max_len = 0
for s in df["img"]:
    arr = extract_numeric_array(s)
    arrays.append(arr)
    if arr.size > max_len:
        max_len = arr.size

print(f"\nMax image vector length: {max_len}")

# --- Pad or truncate arrays to uniform length ---
def pad_to_length(a, length):
    if a.size < length:
        return np.pad(a, (0, length - a.size), mode="constant")
    elif a.size > length:
        return a[:length]
    else:
        return a

X = np.vstack([pad_to_length(a, max_len) for a in arrays])
y = df["labels"].round().astype(int)

print("\nFinal feature matrix shape:", X.shape)
print("Label distribution:", np.unique(y, return_counts=True))

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Optional: scale + PCA to reduce input size ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# PCA (retain 95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)

print(f"After PCA: {X_train_p.shape[1]} components retained (95% variance)")

# --- Random Forest with hyperparameter tuning ---
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
    "class_weight": [None, "balanced"]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

grid = GridSearchCV(
    rf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_p, y_train)
best_rf = grid.best_estimator_

print("\nBest Random Forest Parameters:")
print(grid.best_params_)

# --- Evaluate ---
y_pred = best_rf.predict(X_test_p)
acc = accuracy_score(y_test, y_pred)
print("\nRandom Forest Accuracy:", round(acc, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save model + preprocessors ---
joblib.dump(best_rf, "random_forest.pkl")
joblib.dump(scaler, "rf_scaler.pkl")
joblib.dump(pca, "rf_pca.pkl")

print("\n Saved: random_forest.pkl, rf_scaler.pkl, rf_pca.pkl")
