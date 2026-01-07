import os
import cv2
from ultralytics import YOLO

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/person_and_bib.pt")
INPUT_FOLDER = os.environ.get("INPUT_FOLDER", "input")
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", "output")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO bib-detection model
model = YOLO(MODEL_PATH)

# ---------------------------------------------------------
# PROCESS IMAGES
# ---------------------------------------------------------
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(INPUT_FOLDER, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ Failed to open image: {filename}")
            continue

        # Run detection
        results = model(img)[0]

        # If detections exist
        if len(results.boxes) == 0:
            print(f"⚠ No bib detected in image: {filename}")
            continue

        # Assume bib region = class 0 OR the highest-confidence box
        bib_box = results.boxes[0]     # first box (already sorted by confidence)
        x1, y1, x2, y2 = map(int, bib_box.xyxy[0])

        # Crop safely (avoid negative values)
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped = img[y1:y2, x1:x2]

        # Save cropped result
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(out_path, cropped)

        print(f"✔ Saved cropped bib region: {out_path}")

print("🎯 All done!")