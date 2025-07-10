import os
import cv2
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import models, transforms

# Paths
DATA_DIR = "../data"
DETECTIONS_DIR = "../outputs"
OUTPUT_DIR = "../features"
VIDEOS = {
    "broadcast": os.path.join(DATA_DIR, "broadcast.mp4"),
    "tacticam": os.path.join(DATA_DIR, "tacticam.mp4")
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load pretrained ResNet50 (ImageNet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()  # remove classification layer
resnet.eval().to(device)

# Image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_embedding(image):
    image = transform(image).unsqueeze(0).to(device)
    embedding = resnet(image)
    return embedding.squeeze().cpu().numpy()

def process_video(video_path, detections_path, tag):
    cap = cv2.VideoCapture(video_path)
    detections = pd.read_csv(detections_path)

    results = []
    grouped = detections.groupby("frame")

    print(f"[INFO] Extracting embeddings for: {tag}")

    for frame_idx, group in tqdm(grouped):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        for _, row in group.iterrows():
            x1, y1, x2, y2 = map(int, [row.x1, row.y1, row.x2, row.y2])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            try:
                emb = extract_embedding(crop)
                results.append({
                    "frame": int(frame_idx),
                    "bbox": [x1, y1, x2, y2],
                    "embedding": emb
                })
            except Exception as e:
                print(f"[WARN] Skipping one crop: {e}")

    cap.release()

    # Save as pickle
    out_path = os.path.join(OUTPUT_DIR, f"{tag}_embeddings.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[DONE] Saved embeddings: {out_path}")

if __name__ == "__main__":
    for tag in ["broadcast", "tacticam"]:
        video_path = VIDEOS[tag]
        det_path = os.path.join(DETECTIONS_DIR, f"{tag}_detections.csv")
        process_video(video_path, det_path, tag)
