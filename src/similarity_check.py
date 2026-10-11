import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_embedding(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    return np.load(path)

def compare_faces(id_embedding_file, live_embedding_file, threshold=0.6):
    # Load embeddings
    id_embedding = load_embedding(id_embedding_file)
    live_embedding = load_embedding(live_embedding_file)

    # Compute Euclidean distance
    distance = np.linalg.norm(id_embedding - live_embedding)

    print(f"[INFO] Distance: {distance:.4f}")

    if distance <= threshold:
        print("[RESULT] ✅ Face Matched: ID proof and Live face belong to the same person.")
        return True
    else:
        print("[RESULT] ❌ Face Not Matched: ID proof and Live face are different.")
        return False

if __name__ == "__main__":
    compare_faces("student id card example_face1.npy", "live_face1.npy")
