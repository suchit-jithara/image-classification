# app.py
import io
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

# --------- Load models once at startup ----------
# YOLOv8 (ultralytics) will automatically download weights (yolov8n.pt) on first run.
yolo = YOLO("yolov8n.pt")  # small and fast model; switch to yolov8s.pt or custom model if needed

# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load categories
with open("categories.json", "r") as f:
    categories = json.load(f)

# Flatten subcategory list and maintain mapping to category
text_labels = []
label_to_category = {}
for cat, subs in categories.items():
    for s in subs:
        text_labels.append(s)
        label_to_category[s] = cat

app = FastAPI(title="SwiftBid: YOLO + CLIP Classifier")

# Helper: run CLIP on a PIL image and return best subcategory + score
def classify_with_clip(pil_image: Image.Image):
    inputs = clip_processor(text=text_labels, images=pil_image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # shape (1, N_text_labels)
    probs = logits_per_image.softmax(dim=1)
    best_idx = int(probs.argmax().item())
    best_sub = text_labels[best_idx]
    best_cat = label_to_category[best_sub]
    confidence = float(probs[0][best_idx].item())
    return {"category": best_cat, "subcategory": best_sub, "confidence": round(confidence, 4)}

# Helper: convert ultralytics result boxes to list of dicts
def boxes_to_list(results):
    boxes = []
    for box, conf, cls in zip(results.boxes.xyxy.tolist(), results.boxes.conf.tolist(), results.boxes.cls.tolist()):
        x1, y1, x2, y2 = map(float, box)
        boxes.append({"xyxy": [x1, y1, x2, y2], "conf": float(conf), "cls": int(cls)})
    return boxes

@app.post("/classify")
async def classify_image(
    file: UploadFile = File(...),
    use_detection: bool = Query(True, description="Use YOLO detection before CLIP (true/false). If false, CLIP runs on full image."),
    pick_strategy: str = Query("largest", description="Which detected box to pick: 'largest' or 'highest_conf' or 'all'"),
):
    """
    Upload image (form field 'file'). 
    Query params:
      - use_detection (default true) : run YOLO before CLIP
      - pick_strategy : 'largest' | 'highest_conf' | 'all'
    """
    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Invalid image or upload failed", "detail": str(e)})

    det_info = []
    clip_results = []

    if use_detection:
        # ultralytics expects numpy array (H,W,3) in RGB
        np_img = np.array(pil_image)
        results = yolo(np_img)  # returns list of Results; we use first
        r = results[0]
        if not hasattr(r, "boxes") or len(r.boxes) == 0:
            # No detections: fallback to whole image
            clip_res = classify_with_clip(pil_image)
            clip_results.append({"clip": clip_res, "box": None})
            det_info = []
        else:
            boxes = boxes_to_list(r)
            det_info = boxes

            # Choose boxes per strategy
            if pick_strategy == "largest":
                # pick box with largest area
                best = max(boxes, key=lambda b: (b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1]))
                chosen = [best]
            elif pick_strategy == "highest_conf":
                best = max(boxes, key=lambda b: b["conf"])
                chosen = [best]
            elif pick_strategy == "all":
                chosen = boxes
            else:
                chosen = boxes

            # For each chosen box: crop and classify
            for b in chosen:
                x1, y1, x2, y2 = map(int, b["xyxy"])
                # Ensure crop within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(pil_image.width, x2)
                y2 = min(pil_image.height, y2)
                try:
                    crop = pil_image.crop((x1, y1, x2, y2))
                    clip_res = classify_with_clip(crop)
                    clip_results.append({"clip": clip_res, "box": b})
                except Exception as e:
                    # fallback to whole image if crop fails
                    clip_res = classify_with_clip(pil_image)
                    clip_results.append({"clip": clip_res, "box": None})

    else:
        # No detection: direct CLIP on full image
        clip_res = classify_with_clip(pil_image)
        clip_results.append({"clip": clip_res, "box": None})

    # Choose a final result — return the clip result with highest confidence
    final = max(clip_results, key=lambda item: item["clip"]["confidence"])
    response = {
        "final": final["clip"],
        "detection_chosen_box": final["box"],
        "all_clip_results": clip_results,
        "all_detections": det_info
    }
    return response
