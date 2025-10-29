# STEP 3: Create and train a deep machine learning model using either Tensorflow or PyTorch

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler     # 🟩 for mixed precision
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os

# ===========================
# 🧩 CONFIGURATION
# ===========================
model_name = "nateraw/vit-base-cats-vs-dogs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8                # 🟩 smaller batches often faster on limited GPUs
EPOCHS = 2 # Faster 
LR = 2e-5

# ===========================
# 🧩 LOAD MODEL + PROCESSOR
# ===========================
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)  # 🟩 fast image processor
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# ===========================
# 🧩 LOAD DATA
# ===========================
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')

# 🟩 optional sampling for faster experimentation
train_df = train_df.sample(5000, random_state=42)
val_df = val_df.sample(1000, random_state=42)

label2id = {'cat': 0, 'dog': 1}
id2label = {0: 'cat', 1: 'dog'}

# ===========================
# 🧩 CACHED DATASET (loads once into memory)
# ===========================
class CachedDataset(Dataset):
    def __init__(self, df, processor):
        self.samples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching dataset"):
            image = Image.open(row.filepath).convert("RGB").resize((224,224))
            inputs = processor(images=image, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in inputs.items()}
            item["labels"] = torch.tensor(label2id[row.label])
            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

train_ds = CachedDataset(train_df, processor)
val_ds = CachedDataset(val_df, processor)

# 🟩 For Windows, set num_workers=0 to avoid crashes
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)

# ===========================
# 🧩 TRAINING PREP
# ===========================
optimizer = AdamW(model.parameters(), lr=LR)
scaler = GradScaler()  # 🟩 for mixed precision
best_val_acc = 0.0

# ===========================
# 🧩 TRAINING LOOP
# ===========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        # 🟩 mixed precision speeds up GPU compute
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

    # ===========================
    # 🧩 VALIDATION
    # ===========================
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device)
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with autocast():
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.3f}")

    # 🟩 Save best model checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("trained_model", exist_ok=True)
        model.save_pretrained("trained_model")
        processor.save_pretrained("trained_model")
        print(f"✅ New best model saved (val_acc={val_acc:.3f})")

print(f"🏁 Training complete. Best validation accuracy: {best_val_acc:.3f}")
