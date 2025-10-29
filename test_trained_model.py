# This is to test MY model and differs from test_pretrained.py

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load your trained model
model_path = "trained_model"
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)
model.eval()

# Load an example image
image = Image.open(r"C:\Users\gpoke\catsdogs\dataset\images\1345.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_id = probs.argmax().item()
    confidence = probs[pred_id].item()

label = model.config.id2label[pred_id]
print(f"Prediction: {label} (confidence: {confidence:.2f})")
