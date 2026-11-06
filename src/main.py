import os
import cv2
import numpy as np
import face_recognition

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ID_PROOFS_DIR = os.path.join(DATA_DIR, "id_proofs")
LIVE_FACES_DIR = os.path.join(DATA_DIR, "live_faces")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "models")

# Utility function to save embedding
def save_embedding(embedding, file_path):
    np.save(file_path, embedding)
    print(f"[SAVED] Embedding stored at {file_path}")

# Step 1: Detect and crop face from ID proof
def detect_face_from_id(image_name):
    img_path = os.path.join(ID_PROOFS_DIR, image_name)
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"[ERROR] Could not read image: {img_path}")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_img)

    if len(boxes) == 0:
        raise Exception("[ERROR] No face detected in ID proof.")

    top, right, bottom, left = boxes[0]
    face = img[top:bottom, left:right]
    cropped_path = os.path.join(DATA_DIR, "embeddings", "id_face.jpg")
    cv2.imwrite(cropped_path, face)
    print(f"[SAVED] Cropped ID face at {cropped_path}")
    return cropped_path

# Step 2: Generate embeddings
def generate_embedding(image_path, save_as):
    img = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(img)

    if len(encodings) == 0:
        raise Exception("[ERROR] No face encoding found.")

    embedding = encodings[0]
    save_embedding(embedding, os.path.join(EMBEDDINGS_DIR, save_as))
    return embedding

# Step 3: Capture live face
def capture_live_face():
    cam = cv2.VideoCapture(0)
    print("[INFO] Press SPACE to capture, ESC to quit.")

    live_face_path = os.path.join(LIVE_FACES_DIR, "live_face.jpg")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        cv2.imshow("Live Capture", frame)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC
            print("[INFO] Exiting without capture.")
            cam.release()
            cv2.destroyAllWindows()
            return None
        elif key % 256 == 32:  # SPACE
            cv2.imwrite(live_face_path, frame)
            print(f"[SAVED] Live face captured at {live_face_path}")
            cam.release()
            cv2.destroyAllWindows()
            return live_face_path

# Step 4: Compare embeddings
def compare_embeddings(id_file, live_file, threshold=0.6):
    id_embedding = np.load(os.path.join(EMBEDDINGS_DIR, id_file))
    live_embedding = np.load(os.path.join(EMBEDDINGS_DIR, live_file))

    distance = np.linalg.norm(id_embedding - live_embedding)
    print(f"[INFO] Distance: {distance:.4f}")

    if distance <= threshold:
        print("[RESULT] ✅ Face Matched: Same person.")
    else:
        print("[RESULT] ❌ Face Not Matched: Different person.")

# Main workflow
if __name__ == "__main__":
    # Step 1: Detect face from ID proof
    id_face_path = detect_face_from_id("student id card example.jpg")

    # Step 2: Create embedding for ID proof face
    generate_embedding(id_face_path, "id_face.npy")

    # Step 3: Capture live face
    live_face_path = capture_live_face()
    if live_face_path:
        generate_embedding(live_face_path, "live_face.npy")

        # Step 4: Compare embeddings
        compare_embeddings("id_face.npy", "live_face.npy")

