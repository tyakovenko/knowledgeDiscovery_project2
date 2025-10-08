
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- robust parser for numeric data in 'img' strings (no sizes, img-only) ---
def extract_numeric_array(s: str) -> np.ndarray:
    """
    Extract numeric tokens from a stringified image array.
    Handles artifacts like '...', '\\n', and brackets.
    """
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

# --- parse 'img' into numeric arrays ---
arrays = []
max_len = 0
for s in df["img"]:
    arr = extract_numeric_array(s)
    arrays.append(arr)
    if arr.size > max_len:
        max_len = arr.size

print(f"\nMax image vector length: {max_len}")

# --- pad/truncate to uniform length ---
def pad_to_length(a: np.ndarray, length: int) -> np.ndarray:
    if a.size < length:
        return np.pad(a, (0, length - a.size), mode="constant")
    elif a.size > length:
        return a[:length]
    return a

X = np.vstack([pad_to_length(a, max_len) for a in arrays])
y = df["labels"].round().astype(int)

print("\nFinal X shape:", X.shape)
print("Label distribution:", np.unique(y, return_counts=True))

# --- train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- scale + PCA (keeps it fast & denoised even with large vectors) ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)

print(f"After PCA: {X_train_p.shape[1]} components retained (95% variance)")

# --- Decision Tree with hyperparameter tuning ---
param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "class_weight": [None, "balanced"],
    "ccp_alpha": [0.0, 0.0001, 0.001]  # minimal cost-complexity pruning
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_p, y_train)
best_dt = grid.best_estimator_
print("\nBest Decision Tree Parameters:")
print(grid.best_params_)

# --- evaluate ---
y_pred = best_dt.predict(X_test_p)
acc = accuracy_score(y_test, y_pred)
print("\nDecision Tree Accuracy:", round(acc, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- save model + preprocessors ---
joblib.dump(best_dt, "decision.pkl")
joblib.dump(scaler, "dt_scaler.pkl")
joblib.dump(pca, "dt_pca.pkl")

print("\n Saved: decision_tree.pkl, dt_scaler.pkl, dt_pca.pkl")
