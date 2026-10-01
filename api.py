from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = load_model("pneumonia_model.keras")
img_size = 150

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((img_size, img_size))
    array = np.array(image) / 255.0
    array = np.expand_dims(array, axis=0)

    prob = model.predict(array)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    return {
        "label": label,
        "confidence": float(prob if label == "PNEUMONIA" else 1 - prob)
    }
