import os
import cv2
import pandas as pd
from tqdm import tqdm

# Paths
DATA_DIR = "../data"
MATCHES_PATH = "../outputs/player_matches.csv"
OUT_DIR = "../outputs/visualized"
os.makedirs(OUT_DIR, exist_ok=True)

VIDEOS = {
    "broadcast": os.path.join(DATA_DIR, "broadcast.mp4"),
    "tacticam": os.path.join(DATA_DIR, "tacticam.mp4")
}

def draw_boxes(video_path, view_tag, matches_df):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out_path = os.path.join(OUT_DIR, f"{view_tag}_matched.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"[INFO] Rendering: {view_tag}")

    for frame_idx in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections for this frame & view
        frame_matches = matches_df[
            (matches_df['frame'] == frame_idx) &
            (matches_df['view'] == view_tag)
        ]

        for _, row in frame_matches.iterrows():
            x1, y1, x2, y2 = map(int, [row.x1, row.y1, row.x2, row.y2])
            player_id = int(row.player_id)

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {player_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_writer.write(frame)

    cap.release()
    out_writer.release()
    print(f"[DONE] Saved visualization: {out_path}")

if __name__ == "__main__":
    matches = pd.read_csv(MATCHES_PATH)

    for view in ["broadcast", "tacticam"]:
        draw_boxes(VIDEOS[view], view, matches)
