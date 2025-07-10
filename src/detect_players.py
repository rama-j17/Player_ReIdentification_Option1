import os
import cv2
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
VIDEOS = {
    "broadcast": "../data/broadcast.mp4",
    "tacticam": "../data/tacticam.mp4"
}
MODEL_PATH = "../models/best.pt"
OUTPUT_DIR = "../outputs"
TARGET_CLASS_NAME = "player"  # Only keep 'player' detections

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Optional: Get class names from model (only works if class names are stored in model metadata)
try:
    class_names = model.names
except:
    class_names = {0: "player"}  # fallback, assuming only one class

def process_video(video_path, tag):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detections = []

    print(f"[INFO] Processing video: {tag} ({frame_count} frames)")

    for frame_idx in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on frame
        results = model(frame)[0]  # first result batch
        for r in results.boxes:
            cls_id = int(r.cls.item())
            class_name = class_names.get(cls_id, str(cls_id))

            if class_name != TARGET_CLASS_NAME:
                continue

            conf = float(r.conf.item())
            x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
            detections.append({
                "frame": frame_idx,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": round(conf, 4),
                "class": class_name
            })

    cap.release()

    # Save CSV
    df = pd.DataFrame(detections)
    out_path = os.path.join(OUTPUT_DIR, f"{tag}_detections.csv")
    df.to_csv(out_path, index=False)
    print(f"[DONE] Saved: {out_path}")

if __name__ == "__main__":
    for tag, path in VIDEOS.items():
        process_video(path, tag)
