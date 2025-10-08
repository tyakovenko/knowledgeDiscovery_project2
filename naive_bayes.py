# naive_bayes_img_only.py
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- Robust numeric parser for 'img' strings (same style as other scripts) ---
def extract_numeric_array(s: str) -> np.ndarray:
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

# --- Parse images into numeric arrays ---
arrays = []
max_len = 0
for s in df["img"]:
    arr = extract_numeric_array(s)
    arrays.append(arr)
    if arr.size > max_len:
        max_len = arr.size

print(f"\nMax image vector length: {max_len}")

# --- Pad/truncate to uniform length ---
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

# --- Standardize + PCA (NB benefits from denoising/lower-dim) ---
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)

print(f"After PCA: {X_train_p.shape[1]} components retained (95% variance)")

# --- Gaussian Naive Bayes + simple tuning (var_smoothing) ---
param_grid = {"var_smoothing": np.logspace(-12, -6, 7)}  # 1e-12 ... 1e-6
grid = GridSearchCV(
    GaussianNB(),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train_p, y_train)

best_nb = grid.best_estimator_
print("\nBest GaussianNB Parameters:", grid.best_params_)

# --- Evaluate ---
y_pred = best_nb.predict(X_test_p)
acc = accuracy_score(y_test, y_pred)
print("\nNaive Bayes Accuracy:", round(acc, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save artifacts ---
joblib.dump(best_nb, "naive_bayes.pkl")
joblib.dump(scaler, "nb_scaler.pkl")
joblib.dump(pca, "nb_pca.pkl")
print("\nSaved: naive_bayes.pkl, nb_scaler.pkl, nb_pca.pkl")
