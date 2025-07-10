import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Config
FEATURE_DIR = "../features"
OUTPUT_PATH = "../outputs/player_matches.csv"
SIMILARITY_THRESHOLD = 0.85  # Tune based on testing

def load_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def match_players(broadcast, tacticam):
    player_id_counter = 1
    assigned_ids = {}

    matched_rows = []

    print("[INFO] Matching players between views...")

    for tac_det in tqdm(tacticam):
        tac_emb = tac_det["embedding"]
        tac_frame = tac_det["frame"]
        best_match_id = None
        best_score = 0

        for broad_det in broadcast:
            if abs(broad_det["frame"] - tac_frame) > 2:  # temporal window of ±2
                continue

            score = cosine_similarity(
                [tac_emb], [broad_det["embedding"]])[0][0]

            if score > best_score and score >= SIMILARITY_THRESHOLD:
                best_score = score
                best_match_id = assigned_ids.get(
                    id(broad_det), None)

        if best_match_id is not None:
            pid = best_match_id
        else:
            pid = player_id_counter
            player_id_counter += 1

        # Save ID assignments
        assigned_ids[id(tac_det)] = pid

        # Log match
        matched_rows.append({
            "player_id": pid,
            "frame": tac_det["frame"],
            "view": "tacticam",
            "x1": tac_det["bbox"][0],
            "y1": tac_det["bbox"][1],
            "x2": tac_det["bbox"][2],
            "y2": tac_det["bbox"][3]
        })

        # If matched, also add the corresponding broadcast detection
        if pid not in [row["player_id"] for row in matched_rows if row["view"] == "broadcast"]:
            for bd in broadcast:
                if abs(bd["frame"] - tac_det["frame"]) <= 2:
                    emb_sim = cosine_similarity(
                        [bd["embedding"]], [tac_emb])[0][0]
                    if emb_sim >= SIMILARITY_THRESHOLD:
                        matched_rows.append({
                            "player_id": pid,
                            "frame": bd["frame"],
                            "view": "broadcast",
                            "x1": bd["bbox"][0],
                            "y1": bd["bbox"][1],
                            "x2": bd["bbox"][2],
                            "y2": bd["bbox"][3]
                        })
                        assigned_ids[id(bd)] = pid
                        break

    return matched_rows

if __name__ == "__main__":
    tac_path = os.path.join(FEATURE_DIR, "tacticam_embeddings.pkl")
    broad_path = os.path.join(FEATURE_DIR, "broadcast_embeddings.pkl")

    tacticam_data = load_embeddings(tac_path)
    broadcast_data = load_embeddings(broad_path)

    results = match_players(broadcast_data, tacticam_data)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[DONE] Matches saved to: {OUTPUT_PATH}")
