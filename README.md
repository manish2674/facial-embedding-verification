# Student ID Verification using Face Recognition

# Project Overview
This project implements a **Student ID Verification System** using Python and the `face_recognition` library.  
The goal is to verify whether the person presenting a **live face capture** matches the **face on their ID card**.

The workflow:
1. **Extract face from ID card image** (face cropping).
2. **Generate face embeddings** for ID face and live captured face.
3. **Compare embeddings** to calculate similarity (Euclidean distance).
4. **Decision**: Match ✅ or Not Match ❌

