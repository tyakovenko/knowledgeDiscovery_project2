from pathlib import Path
import pandas as pd

# Use current directory where this script lives
DATA_ROOT = Path(__file__).parent

# Folder → label mapping
FOLDER_TO_LABEL = {
    "fogsmog": "fog",
    "rime": "snow",
}

# Allowed image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

rows = []

for folder, label in FOLDER_TO_LABEL.items():
    cls_dir = DATA_ROOT / folder
    if not cls_dir.exists():
        print(f"WARNING: {cls_dir} not found, skipping.")
        continue

    for p in cls_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            # Add prefix for CSV only (not renaming file)
            prefix = "f" if label == "fog" else "s"
            prefixed_name = f"{prefix}{p.name}"

            rows.append({
                "Images": prefixed_name,  # prefixed name shown in CSV
                "Class": label
            })

# Build DataFrame
df = pd.DataFrame(rows, columns=["Images", "Class"])
df = df.sort_values(["Class", "Images"]).reset_index(drop=True)

# Display preview
print(df.head())
print(f"\nTotal rows: {len(df)}")
print("\nPer-class counts:")
print(df["Class"].value_counts())

# Save to CSV
df.to_csv("all_images_labels.csv", index=False)
print("\n✅ all_images_labels.csv created successfully!")
