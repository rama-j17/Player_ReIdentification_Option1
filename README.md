# Cross-Camera Soccer Player Re-Identification (Broadcast ↔ Tacticam)

This project implements a full pipeline for re-identifying soccer players across two distinct video views: a broadcast view and a wide-angle tacticam. Given the challenges of viewpoint, resolution, and player motion, we utilize object detection, appearance-based feature embeddings, and cosine similarity to establish identity matches.

---

## 🛠 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/rama-j17/Player_ReIdentification_Option1.git
cd Player_ReIdentification_Option1
```

### 2. Install Dependencies

We recommend creating a virtual environment first:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required libraries:

```bash
pip install -r requirements.txt
```

### 3. Download Large Files (Videos, Model)

- Download `broadcast.mp4`, `tacticam.mp4`, and `best.pt`.
- Place them in the following structure:

```
data/
  ├─ broadcast.mp4
  └─ tacticam.mp4
models/
  └─ best.pt
```

### 4. Run the Pipeline

```bash
# Step 1: Detect Players
python src/detect_players.py

# Step 2: Extract Features & Embeddings
python src/extract_features.py

# Step 3: Match Players Across Views
python src/match_players.py

# Step 4: Visualize Results
python src/visualize_matches.py
```

---

## 📦 Project Structure

```
Player_ReIdentification_Option1/
├── data/                  # Videos and intermediate detections
├── models/               # Pretrained YOLOv11 model
├── features/             # Extracted player embeddings
├── outputs/              # CSV matches and annotated videos
├── src/                  # All pipeline scripts
├── requirements.txt      # Dependencies
├── README.md             # You are here
└── report.pdf            # Final report 
```

---
## Result
Annotated Videos are stored in `outputs` folder under `visualized` sub-folder.
