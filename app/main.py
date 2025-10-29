# STEP 4: Create endpoints using FastAPI that will accept an image and return the class

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch, io

# Load model and processor once at startup
MODEL_PATH = "trained_model"
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
model.eval()

app = FastAPI(title="Cats vs Dogs Classifier")


# Adding HTML because default UI is confusing

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Cats vs Dogs Classifier 🐱🐶</title>
            <style>
                body { font-family: sans-serif; text-align: center; margin-top: 100px; }
                input[type=file] { margin: 10px; }
                button { padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 6px; }
                #result { margin-top: 20px; font-size: 20px; }
            </style>
        </head>
        <body>
            <h1>Cats vs Dogs Classifier 🐱🐶</h1>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Predict</button>
            </form>
            <div id="result"></div>

            <script>
                const form = document.getElementById('uploadForm');
                form.onsubmit = async (e) => {
                    e.preventDefault();
                    const fileInput = form.querySelector('input[type=file]');
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    const response = await fetch('/predict', { method: 'POST', body: formData });
                    const result = await response.json();
                    document.getElementById('result').innerText =
                        result.error ? result.error :
                        `Prediction: ${result.prediction} (Confidence: ${(result.confidence * 100).toFixed(1)}%)`;
                };
            </script>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_id = probs.argmax().item()
            confidence = probs[pred_id].item()
        label = model.config.id2label[pred_id]
        return {"prediction": label, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}
