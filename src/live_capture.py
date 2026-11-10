import cv2
import face_recognition
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DIR = os.path.join(BASE_DIR, "data", "live_faces")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(LIVE_DIR, exist_ok=True)

def capture_and_save_embedding(output_name="live_face1"):
    cap = cv2.VideoCapture(0)

    print("[INFO] Press SPACE to capture face, ESC to quit.")

    while True :
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to access camera.")
            break

        cv2.imshow("Live Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:
            face_locations = face_recognition.face_locations(frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                embedding = face_encodings[0]

                # Save image
                image_path = os.path.join(LIVE_DIR, f"{output_name}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"[SAVED] Image saved at {image_path}")

                # Save embedding
                npy_path = os.path.join(MODEL_DIR, f"{output_name}.npy")
                np.save(npy_path, embedding)
                print(f"[SAVED] Embedding saved at {npy_path}")
                break
            else:
                print("[WARN] No face detected, try again.")

        # ESC → Exit
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_save_embedding("live_face1")
 
