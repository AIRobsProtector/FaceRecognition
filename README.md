
# Face Recognition System with Dlib and OpenCV

This project is a complete face recognition pipeline using Python, Dlib, and OpenCV. It includes:

- Real-time face image collection via webcam
- Feature extraction and CSV storage
- Live face recognition with optional blink detection

## Project Structure

```
.
├── get_faces_from_camera.py             # Capture and save face images from camera
├── features_extraction_to_csv.py        # Extract features and save to features_all.csv
├── face_reco_from_camera.py             # Real-time face recognition (with optional blink detection)
├── data/
│   ├── data_faces_from_camera/          # Stores face images of each person
│   ├── data_dlib/                       # Contains dlib model files
│   └── features_all.csv                 # CSV of registered face features
```

## Usage Steps

### 1. Capture Face Images

Run:

```bash
python get_faces_from_camera.py
```

Key instructions:

- Press `n` to create a new user folder
- Press `s` to save the current detected face
- Press `q` to quit

Images are saved in `./data/data_faces_from_camera/person_X/`.

---

### 2. Extract Face Features

Run:

```bash
python features_extraction_to_csv.py
```

This reads images from `data_faces_from_camera`, extracts 128D features, averages them, and writes to `features_all.csv`.

---

### 3. Real-Time Face Recognition

Run:

```bash
python face_reco_from_camera.py
```

This loads the webcam stream, detects faces, and compares them with those in `features_all.csv`. Recognition results are shown live.

Optional:

- Press `b` to enable blink detection (only works when one face is in the frame and blink count > 5).

## Required Dlib Models

Place the following pre-trained models under `./data/data_dlib/`:

- `shape_predictor_68_face_landmarks.dat`
- `dlib_face_recognition_resnet_model_v1.dat`

Download them from the official Dlib GitHub or other trusted sources.

## Dependencies

```bash
pip install opencv-python dlib imutils numpy pandas pillow scipy
```

Recommended: Python 3.6+ and using a virtual environment.

## Notes

- All faces should be collected using the same camera for better consistency.
- System does not support multi-face recognition in blink mode.
- If `features_all.csv` is missing, you'll be prompted to run the first two scripts.

## Acknowledgements

This project uses models and tools provided by [Dlib](http://dlib.net/).
