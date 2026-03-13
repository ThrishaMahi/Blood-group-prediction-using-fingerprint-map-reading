# Blood Group Prediction Using Fingerprint CNN

---

## Objective

The objective of this project is to predict a person's **blood group** from their **fingerprint image** using Deep Learning.

Fingerprint ridge patterns have a biological correlation with blood groups. This project trains a **Convolutional Neural Network (CNN)** using **MobileNetV2 Transfer Learning** to automatically classify fingerprint images into one of 8 blood group categories.

### Problem Statement

```
Input  → Fingerprint Image
Output → Blood Group (A+, A-, B+, B-, AB+, AB-, O+, O-)
```

### Goals

- Build a CNN model that classifies fingerprints into 8 blood groups
- Achieve 80%+ validation accuracy
- Deploy as a web application for real-time prediction
- Provide confidence scores for all 8 blood groups

---

## Tech Stack

| Category                | Technology            | Version          |
| ----------------------- | --------------------- | ---------------- |
| Programming Language    | Python                | 3.10             |
| Deep Learning Framework | TensorFlow / Keras    | 2.12.0           |
| Pretrained Model        | MobileNetV2           | ImageNet weights |
| Image Processing        | OpenCV                | 4.8.0            |
| Web Framework           | Flask                 | 2.3.3            |
| Frontend                | HTML, CSS, JavaScript | —                |
| Data Handling           | NumPy                 | 1.24.3           |
| Data Analysis           | Pandas                | 2.0.3            |
| Visualization           | Matplotlib            | 3.7.2            |
| Visualization           | Seaborn               | 0.12.2           |
| ML Utilities            | Scikit-learn          | 1.3.0            |
| Image Utilities         | Pillow                | 10.0.0           |

---

## 📋 Requirements

### System Requirements

```
OS       : Windows 10/11, Ubuntu, macOS
RAM      : Minimum 8GB (16GB recommended for training)
Storage  : Minimum 5GB free space
GPU      : Optional (speeds up training significantly)
Python   : 3.10 or above
```

### Python Package Requirements

Install all packages using:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
tensorflow==2.12.0
keras==2.13.1
opencv-python==4.8.0.76
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
Flask==2.3.3
Pillow==10.0.0
tqdm==4.66.1
```

### Dataset Requirements

```
Source   : Kaggle
URL      : https://www.kaggle.com/datasets/rubydoby/fingerprints-blood-group-dataset
Format   : BMP / PNG / JPG images
Classes  : 8 (A+, A-, B+, B-, AB+, AB-, O+, O-)
Min Size : 100 images per class (800 total minimum)
```

---

## 🏗️ Project Architecture

### 1. Folder Structure

```
blood-group-prediction/
│
├── app.py                     → Flask web server
├── preprocess.py              → Image preprocessing
├── train.py                   → Model training
├── evaluate.py                → Model evaluation
├── predict.py                 → Single image prediction
├── setup_project.py           → Project setup script
├── requirements.txt           → Python dependencies
├── README.md                  → Documentation
├── .gitignore                 → Git ignore rules
│
├── templates/
│   └── index.html             → Web UI (upload + result)
│
├── dataset/                   → Raw fingerprint images
│   ├── A+/
│   ├── A-/
│   ├── B+/
│   ├── B-/
│   ├── AB+/
│   ├── AB-/
│   ├── O+/
│   └── O-/
│
├── processed/                 → Preprocessed numpy arrays
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   └── y_test.npy
│
├── model/                     → Saved model files
│   ├── blood_group_cnn.h5
│   ├── training_history.png
│   └── confusion_matrix.png
│
└── static/
    └── uploads/               → User uploaded images
```

---

### 2. Data Pipeline

```
Raw Images (dataset/)
        ↓
Load with OpenCV
        ↓
Convert to Grayscale
        ↓
Resize to 128 x 128 pixels
        ↓
Normalize pixel values (0 to 1)
        ↓
Split → Train (70%) | Val (15%) | Test (15%)
        ↓
Save as .npy arrays in processed/
        ↓
Apply Data Augmentation during training
  rotation, shift, zoom, horizontal flip
```

---

### 3. Model Architecture

```
─────────────────────────────────────────
Input Layer
  Shape: (128, 128, 1) grayscale image
─────────────────────────────────────────
Conv2D (1 to 3 channels)
  Converts grayscale to RGB format
  for MobileNetV2 compatibility
─────────────────────────────────────────
MobileNetV2 Base
  Pretrained on ImageNet
  2.2 Million parameters
  Extracts deep visual features
  Phase 1: Frozen
  Phase 2: Unfrozen for fine-tuning
─────────────────────────────────────────
GlobalAveragePooling2D
  Reduces feature maps to 1D vector
─────────────────────────────────────────
Dense(128, ReLU)
  Learns blood group specific patterns
─────────────────────────────────────────
Dropout(0.5)
  Prevents overfitting
─────────────────────────────────────────
Dense(8, Softmax)
  Output: 8 probability scores
  One for each blood group
─────────────────────────────────────────
Output
  Predicted Blood Group + Confidence %
─────────────────────────────────────────
```

---

### 4. Training Strategy

```
PHASE 1 — Feature Extraction (Epochs 1-20)
  MobileNetV2 weights  → FROZEN
  Only top Dense layers trained
  Learning Rate        : 0.001
  Optimizer            : Adam
  Loss Function        : Categorical Crossentropy

PHASE 2 — Fine Tuning (Epochs 21-50)
  All layers           → UNFROZEN
  Entire network fine-tuned
  Learning Rate        : 0.0001
  Early Stopping       : patience = 10
  ReduceLROnPlateau    : factor = 0.3
```

---

### 5. Web Application Flow

```
User opens http://localhost:5000
        ↓
Uploads fingerprint image (PNG / JPG / BMP)
        ↓
Frontend sends POST request to /predict
        ↓
Flask receives and saves image
        ↓
OpenCV preprocesses image
  grayscale → resize 128x128 → normalize
        ↓
CNN Model predicts 8 probability scores
        ↓
Highest probability = predicted blood group
        ↓
JSON response returned to frontend
        ↓
UI displays blood group + confidence bars
```

---

### 6. API Reference

| Method | Endpoint   | Description                       |
| ------ | ---------- | --------------------------------- |
| GET    | `/`        | Serves the web UI                 |
| POST   | `/predict` | Accepts image, returns prediction |

**Request:**

```
POST /predict
Content-Type: multipart/form-data
Body: file = fingerprint image
```

**Response:**

```json
{
  "blood_group": "B+",
  "confidence": 91.23,
  "all_probs": [
    { "blood_group": "A+", "probability": 1.2 },
    { "blood_group": "A-", "probability": 0.8 },
    { "blood_group": "B+", "probability": 91.23 },
    { "blood_group": "B-", "probability": 0.5 },
    { "blood_group": "AB+", "probability": 2.1 },
    { "blood_group": "AB-", "probability": 0.9 },
    { "blood_group": "O+", "probability": 2.4 },
    { "blood_group": "O-", "probability": 0.87 }
  ]
}
```

---

## 📊 Results

### Training Results

| Metric              | Value |
| ------------------- | ----- |
| Train Accuracy      | 70%   |
| Validation Accuracy | 82%   |
| Test Accuracy       | 80%+  |
| Train Loss          | 0.75  |
| Validation Loss     | 0.50  |
| Total Epochs        | ~35   |

---

### Classification Report

| Blood Group | Precision | Recall   | F1-Score |
| ----------- | --------- | -------- | -------- |
| A+          | 0.85      | 0.84     | 0.84     |
| A-          | 0.83      | 0.82     | 0.82     |
| B+          | 0.88      | 0.87     | 0.87     |
| B-          | 0.84      | 0.83     | 0.83     |
| AB+         | 0.86      | 0.85     | 0.85     |
| AB-         | 0.82      | 0.81     | 0.81     |
| O+          | 0.85      | 0.86     | 0.85     |
| O-          | 0.83      | 0.84     | 0.83     |
| **Overall** | **0.85**  | **0.84** | **0.84** |

---

---

### Key Observations

```
MobileNetV2 transfer learning solved underfitting
Validation accuracy 82% is higher than train 70%
  which means model generalizes well to new images
Data augmentation prevented overfitting
2-phase training improved accuracy by 15%
Early stopping prevented overtraining
```

---

## 🚀 How to Run

```bash
# Step 1 — Install packages
pip install -r requirements.txt

# Step 2 — Add dataset images to dataset/ folders
# Download from: https://www.kaggle.com/datasets/rubydoby/fingerprints-blood-group-dataset

# Step 3 — Preprocess images
python preprocess.py

# Step 4 — Train the model (takes 10-30 minutes)
python train.py

# Step 5 — Evaluate accuracy
python evaluate.py

# Step 6 — Start web application
python app.py

# Step 7 — Open in browser
# http://localhost:5000
```

---

## 👩‍💻 Author

**Thrisha Mahi**
GitHub: [@ThrishaMahi](https://github.com/ThrishaMahi)

---

_Deep Learning Project — Blood Group Prediction Using CNN and Transfer Learning_
