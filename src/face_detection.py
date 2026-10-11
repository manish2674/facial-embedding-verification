import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ID_PROOF_PATH = os.path.join(BASE_DIR, "data", "id_proofs")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "embeddings")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Haar cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_and_save_faces(image_name):
    image_path = os.path.join(ID_PROOF_PATH, image_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"[INFO] No faces found in {image_name}")
        return

    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y+h, x:x+w]
        face_filename = f"{os.path.splitext(image_name)[0]}_face{i+1}.jpg"
        save_path = os.path.join(OUTPUT_PATH, face_filename)
        cv2.imwrite(save_path, face)
        print(f"[SAVED] {save_path}")


if __name__ == "__main__":
    detect_and_save_faces("student id card example.jpg")

