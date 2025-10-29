# 🐱🐶 Cats vs Dogs Classifier

A simple deep learning application that classifies images of cats and dogs using a fine-tuned Vision Transformer (ViT) model.

---

## 🚀 Features
- Pretrained ViT model from Hugging Face ("nateraw/vit-base-cats-vs-dogs")
- Fine-tuned on custom cats vs dogs dataset
- FastAPI REST API for image classification
- Minimal HTML upload page for quick testing

---

## 📦 Installation

```bash
git clone https://github.com/gp4p4zis/catsdogs-classifier.git
cd catsdogs-classifier
python -m venv venv
venv\Scripts\activate      # on Windows
pip install -r requirements.txt
```

---

## 🚀 Running the App
```bash
# Start the FastAPI server
uvicorn app:app --reload
```
* The app will run at http://127.0.0.1:8000
* Open this URL in your browser to see a simple image upload page.

---
## 🖼️ Uploading an Image / Predicting

1. Click **“Choose File”** and select an image of a cat or dog.
2. Click **“Predict”** to see the prediction.

> You should see a prediction like:  
> **“Predicted label:”** `Cat`  
> **“Confidence:”** `92%`

