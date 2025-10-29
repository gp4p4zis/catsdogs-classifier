# STEP 1: Download a use a pre-trained model from Hugging Face

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os

# 1️⃣ Load model + processor
model_name = "nateraw/vit-base-cats-vs-dogs"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# 2️⃣ Load one test image
img_path = r"C:\Users\gpoke\catsdogs\dataset\images\16249.jpg"  # <-- replace with a path from your 25k images
image = Image.open(img_path).convert("RGB")

# 3️⃣ Preprocess and predict
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred_id = probs.argmax().item()
    confidence = probs[0][pred_id].item()

label = model.config.id2label[pred_id]
print(f"Prediction: {label} (confidence: {confidence:.2f})")
