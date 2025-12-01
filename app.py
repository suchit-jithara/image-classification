import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from transformers import CLIPProcessor, CLIPModel
import json

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load categories
with open("categories.json") as f:
    categories = json.load(f)

app = FastAPI()

def classify(image):
    image = Image.open(image)

    # Flatten category list
    text_labels = []
    map_to_category = {}

    for cat, subs in categories.items():
        for s in subs:
            text_labels.append(s)
            map_to_category[s] = cat

    # Encode
    inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)
    best_idx = probs.argmax().item()

    subcategory = text_labels[best_idx]
    category = map_to_category[subcategory]
    confidence = float(probs[0][best_idx])

    return {
        "category": category,
        "subcategory": subcategory,
        "confidence": round(confidence, 4)
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    result = classify(file.file)
    return result
