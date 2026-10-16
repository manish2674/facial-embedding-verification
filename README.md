# Face Recognition & Verification Pipeline

A Python-based computer vision project that verifies whether a face captured through a webcam matches the face extracted from an identity document. The system uses pretrained face-recognition capabilities, facial embeddings, and Euclidean distance to compare two faces.

## Overview

Face verification is a computer vision task that determines whether two face images are likely to belong to the same person.

This project demonstrates an end-to-end face verification pipeline, starting with an image of an identity document and ending with a match or mismatch result based on facial similarity.

The system extracts a face from the document image, converts it into a numerical embedding, captures a face through a webcam, generates another embedding, and compares the two representations.

## Key Features

- **Face Detection:** Locates a face in an identity document image.
- **Face Cropping:** Extracts the detected face for further processing.
- **Facial Embeddings:** Converts face images into numerical representations using a pretrained face-recognition model.
- **Live Face Capture:** Captures a face using a webcam.
- **Similarity Comparison:** Calculates Euclidean distance between two facial embeddings.
- **Verification Result:** Determines whether the measured distance falls within a predefined threshold.
- **Embedding Storage:** Saves facial embeddings as NumPy `.npy` files for subsequent use.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image processing, face detection in the standalone detection script, and webcam capture |
| face_recognition | Face location and facial encoding generation |
| dlib | Underlying face-recognition model implementation used by `face_recognition` |
| NumPy | Numerical operations, embedding storage, and distance calculation |

## How It Works

The main workflow is implemented in `src/main.py`.

### Step 1: Provide an Identity Document Image

The system loads an identity document image from:

`data/id_proofs/`

The current implementation expects the file:

`student id card example.jpg`

### Step 2: Detect and Crop the Face

The `detect_face_from_id()` function uses `face_recognition.face_locations()` to locate faces in the document image.

The first detected face is cropped and saved to:

`data/embeddings/id_face.jpg`

If no face is detected, the function raises an error.

### Step 3: Generate the ID Face Embedding

The `generate_embedding()` function processes the cropped face and generates a facial encoding using the pretrained face-recognition model.

The resulting numerical vector is saved as:

`models/id_face.npy`

### Step 4: Capture the Live Face

The `capture_live_face()` function opens the webcam using OpenCV.

- Press **SPACE** to capture the current camera frame.
- Press **ESC** to exit without capturing.

The captured frame is saved to:

`data/live_faces/live_face.jpg`

The system then generates a facial embedding from the captured image and saves it as:

`models/live_face.npy`

### Step 5: Compare the Facial Embeddings

The `compare_embeddings()` function loads both saved embeddings and calculates their Euclidean distance:

`distance = np.linalg.norm(id_embedding - live_embedding)`

A smaller distance indicates that the two embeddings are closer in the embedding space.

The current decision threshold is `0.6`:

- If distance is less than or equal to `0.6`, the system reports a face match.
- If distance is greater than `0.6`, the system reports a face mismatch.

The threshold is configurable and should be evaluated against representative matching and non-matching face pairs before real-world use.

## Project Structure

```text
face-recognition-pipeline/
│
├── data/
│   ├── id_proofs/
│   │   └── student id card example.jpg
│   │
│   ├── embeddings/
│   │   └── id_face.jpg
│   │
│   └── live_faces/
│       └── live_face.jpg
│
├── models/
│   ├── id_face.npy
│   └── live_face.npy
│
├── src/
│   ├── main.py
│   ├── face_detection.py
│   ├── face_embeddings.py
│   ├── live_capture.py
│   └── similarity_check.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

*Note: The image and embedding files shown above are generated or example files. They may not all be present before the first run.*

### Source Code Files

- **`main.py`** — Runs the complete verification pipeline: face extraction, embedding generation, webcam capture, and comparison.
- **`face_detection.py`** — Provides a standalone face-detection and cropping utility using OpenCV's Haar Cascade classifier.
- **`face_embeddings.py`** — Generates and saves embeddings for supported face images found in `data/embeddings/`.
- **`live_capture.py`** — Provides a standalone webcam capture utility that generates and saves a live face embedding.
- **`similarity_check.py`** — Loads existing embedding files and compares them using Euclidean distance.

The standalone scripts provide separate utilities. The complete workflow described in this README is executed through `main.py`.

## Installation and Setup

### Prerequisites

- Python installed on your computer.
- A working webcam.
- Git, if cloning the repository.
- A compatible installation of the project's Python dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/manish2674/facial-embedding-verification.git
cd facial-embedding-verification
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `face_recognition` package depends on dlib, so installation requirements may vary by operating system and Python environment. Use compatible package versions if installation fails.

### 4. Prepare the Input Image

Place your test identity document image in:

`data/id_proofs/`

Ensure its filename matches the one configured in `src/main.py`, or update the filename in the code.

### 5. Run the Project

From the repository root, execute:

```bash
python src/main.py
```

Follow the webcam instructions in the terminal. Press SPACE to capture a face or ESC to exit.

## Example Output

A successful execution can produce output similar to:

```text
[SAVED] Cropped ID face at .../data/embeddings/id_face.jpg
[SAVED] Embedding stored at .../models/id_face.npy
[INFO] Press SPACE to capture, ESC to quit.
[SAVED] Live face captured at .../data/live_faces/live_face.jpg
[SAVED] Embedding stored at .../models/live_face.npy
[INFO] Distance: 0.7170
[RESULT] Face Not Matched: Different person.
```

The distance shown is an example from a previous execution, not a guaranteed result. Actual results depend on the input images and capture conditions.

## Important Implementation Details

- The project uses a pretrained face-recognition model rather than training a new face-recognition model from scratch.
- Facial embeddings are numerical representations of detected faces.
- NumPy files store the generated embeddings; they are not model weights.
- The current main workflow compares the first detected ID face with the first available live-face encoding.
- The Euclidean distance threshold is a configurable decision rule, not a guarantee of identity.

## Limitations

- **Image Quality:** Lighting, camera angle, resolution, and face alignment may affect verification results.
- **Threshold Selection:** The threshold has not been established as universally reliable. It requires evaluation on representative data.
- **Multiple Faces:** The current implementation uses the first detected face rather than resolving multiple candidates.
- **Liveness Detection:** The project does not include dedicated liveness or anti-spoofing checks.
- **Security:** Face matching alone does not establish that a person is physically present or that an identity document is authentic.

This project is intended for learning and demonstration. It should not be used as the sole basis for high-stakes identity decisions.

## Future Improvements

- Add a graphical interface for easier interaction.
- Improve error handling and input validation.
- Evaluate the verification threshold on matching and non-matching face pairs.
- Add liveness detection and anti-spoofing safeguards.
- Improve support for multiple detected faces.
- Add automated tests and clearer installation instructions.

## Privacy Considerations

Facial images and facial embeddings are biometric data. Use authorized test images, obtain appropriate consent, and handle stored images and embeddings responsibly.

Do not publish another person's identity documents or biometric data without appropriate authorization.

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.

## Author

**Manish Patil**

GitHub: [@manish2674](https://github.com/manish2674)

---

If you find this project useful, feel free to explore the code and learn from the implementation.
