# src/face_embeddings.py

import os
import numpy as np
import face_recognition

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_INPUT_PATH = os.path.join(BASE_DIR, "data", "embeddings")   # cropped faces
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "models")                   # store .npy embeddings

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

def generate_embeddings():
    """
    Generate embeddings for all cropped face images in data/embeddings/
    and save them as .npy files in models/
    """
    for file in os.listdir(EMBEDDINGS_INPUT_PATH):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(EMBEDDINGS_INPUT_PATH, file)
            print(f"[INFO] Processing {file}...")

            # Load image
            image = face_recognition.load_image_file(image_path)  
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                embedding = encodings[0]   # take the first detected face
                save_path = os.path.join(MODEL_OUTPUT_PATH, f"{os.path.splitext(file)[0]}.npy")

                # Save embedding
                np.save(save_path, embedding)
                print(f"[SAVED] Embedding stored at {save_path}")
            else:
                print(f"[WARNING] No face found in {file}")

if __name__ == "__main__":
    generate_embeddings()
