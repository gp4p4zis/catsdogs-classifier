# STEP 2: Load, clean and pre-process the dataset provided using Pandas and NumPy

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = r"C:\Users\gpoke\catsdogs\dataset"
IMG_DIR = os.path.join(BASE_DIR, "images")
LABELS_PATH = os.path.join(BASE_DIR, "labels.csv")

# 1️⃣ Load labels
df = pd.read_csv(LABELS_PATH)

# 2️⃣ Clean column names just in case
df.columns = df.columns.str.strip().str.lower()

# 3️⃣ Normalize label names (make them lowercase)
df['label'] = df['label'].str.lower().str.strip()

# 4️⃣ Add full file paths
df['filepath'] = df['image_name'].apply(lambda x: os.path.join(IMG_DIR, x))

# 5️⃣ Check for missing or invalid paths
missing = df[~df['filepath'].apply(os.path.exists)]
if len(missing):
    print(f"⚠️ Missing {len(missing)} images:")
    print(missing.head())
else:
    print("✅ All image paths valid.")

# 6️⃣ Encode labels (cat=0, dog=1)
label_map = {'cat': 0, 'dog': 1}
df = df[df['label'].isin(label_map.keys())]  # remove any weird entries
df['label_id'] = df['label'].map(label_map)

# 7️⃣ Split into train/val/test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# 8️⃣ Save CSVs for later use
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("✅ Dataset prepared successfully.")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(df['label'].value_counts())
