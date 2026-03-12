import os

files = {

"requirements.txt": """tensorflow==2.13.0
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
""",

"preprocess.py": '''
import os, cv2, numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET_DIR  = "dataset"
IMG_SIZE     = (128, 128)
BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

def load_dataset():
    X, y = [], []
    print("Loading dataset...")
    for label, bg in enumerate(BLOOD_GROUPS):
        folder = os.path.join(DATASET_DIR, bg)
        if not os.path.exists(folder):
            print(f"  WARNING: {folder} not found")
            continue
        files = os.listdir(folder)
        print(f"  {bg}: {len(files)} images")
        for f in tqdm(files, desc=f"Loading {bg}"):
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, IMG_SIZE) / 255.0
            X.append(img); y.append(label)
    X = np.array(X).reshape(-1, 128, 128, 1)
    y = np.array(y)
    print(f"Total: {len(X)} images")
    return X, y

def split_and_save(X, y):
    X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_v,  X_te, y_v, y_te = train_test_split(X_t, y_t, test_size=0.50, random_state=42, stratify=y_t)
    os.makedirs("processed", exist_ok=True)
    for name, arr in [("X_train",X_tr),("X_val",X_v),("X_test",X_te),
                       ("y_train",y_tr),("y_val",y_v),("y_test",y_te)]:
        np.save(f"processed/{name}.npy", arr)
    print(f"Saved → Train:{len(X_tr)} Val:{len(X_v)} Test:{len(X_te)}")

if __name__ == "__main__":
    X, y = load_dataset()
    split_and_save(X, y)
    print("Preprocessing done!")
''',

"train.py": '''
import numpy as np, os, matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

NUM_CLASSES = 8
MODEL_PATH  = "model/blood_group_cnn.h5"

def build_model():
    m = Sequential([
        Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(128,128,1)),
        BatchNormalization(),
        Conv2D(32,(3,3),activation="relu",padding="same"),
        MaxPooling2D(2,2), Dropout(0.25),

        Conv2D(64,(3,3),activation="relu",padding="same"),
        BatchNormalization(),
        Conv2D(64,(3,3),activation="relu",padding="same"),
        MaxPooling2D(2,2), Dropout(0.25),

        Conv2D(128,(3,3),activation="relu",padding="same"),
        BatchNormalization(),
        Conv2D(128,(3,3),activation="relu",padding="same"),
        MaxPooling2D(2,2), Dropout(0.40),

        Flatten(),
        Dense(256,activation="relu"), BatchNormalization(), Dropout(0.50),
        Dense(128,activation="relu"), Dropout(0.30),
        Dense(NUM_CLASSES,activation="softmax")
    ])
    m.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    return m

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    X_train = np.load("processed/X_train.npy")
    X_val   = np.load("processed/X_val.npy")
    y_train = to_categorical(np.load("processed/y_train.npy"), NUM_CLASSES)
    y_val   = to_categorical(np.load("processed/y_val.npy"),   NUM_CLASSES)

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
    ]

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50, batch_size=32,
                        callbacks=callbacks)

    # Plot
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    ax[0].plot(history.history["accuracy"],     label="Train")
    ax[0].plot(history.history["val_accuracy"], label="Val")
    ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(history.history["loss"],     label="Train")
    ax[1].plot(history.history["val_loss"], label="Val")
    ax[1].set_title("Loss"); ax[1].legend()
    plt.savefig("model/training_history.png")
    plt.show()
    print(f"Model saved to {MODEL_PATH}")
''',

"evaluate.py": '''
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

BLOOD_GROUPS = ["A+","A-","B+","B-","AB+","AB-","O+","O-"]

if __name__ == "__main__":
    model  = load_model("model/blood_group_cnn.h5")
    X_test = np.load("processed/X_test.npy")
    y_test = np.load("processed/y_test.npy")

    loss, acc = model.evaluate(X_test, to_categorical(y_test, 8), verbose=0)
    print(f"Test Accuracy : {acc*100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=BLOOD_GROUPS))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=BLOOD_GROUPS,
                yticklabels=BLOOD_GROUPS, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png")
    plt.show()
''',

"predict.py": '''
import cv2, numpy as np, argparse, matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

BLOOD_GROUPS = ["A+","A-","B+","B-","AB+","AB-","O+","O-"]

def predict(image_path):
    model       = load_model("model/blood_group_cnn.h5")
    img         = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_input   = (cv2.resize(img,(128,128))/255.0).reshape(1,128,128,1)
    preds       = model.predict(img_input)[0]
    idx         = np.argmax(preds)
    print(f"Blood Group : {BLOOD_GROUPS[idx]}")
    print(f"Confidence  : {preds[idx]*100:.2f}%")

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(img,cmap="gray"); plt.title("Fingerprint"); plt.axis("off")
    plt.subplot(1,2,2)
    colors = ["green" if i==idx else "steelblue" for i in range(8)]
    plt.barh(BLOOD_GROUPS, preds*100, color=colors)
    plt.xlabel("Confidence (%)"); plt.title(f"Predicted: {BLOOD_GROUPS[idx]}")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    predict(parser.parse_args().image)
''',

"app.py": '''
import os, cv2, numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app           = Flask(__name__)
BLOOD_GROUPS  = ["A+","A-","B+","B-","AB+","AB-","O+","O-"]
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

print("Loading model...")
model = load_model("model/blood_group_cnn.h5")
print("Model ready!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file     = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(filepath)
    img      = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img      = (cv2.resize(img,(128,128))/255.0).reshape(1,128,128,1)
    preds    = model.predict(img)[0]
    idx      = int(np.argmax(preds))
    return jsonify({
        "blood_group" : BLOOD_GROUPS[idx],
        "confidence"  : round(float(preds[idx])*100, 2),
        "all_probs"   : [{"blood_group": bg, "probability": round(float(preds[i])*100,2)}
                         for i, bg in enumerate(BLOOD_GROUPS)]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
''',

"README.md": """# Blood Group Prediction Using Fingerprint CNN
Run order:
1. python setup_project.py
2. pip install -r requirements.txt
3. Add images to dataset/ folders
4. python preprocess.py
5. python train.py
6. python evaluate.py
7. python app.py
8. Open http://localhost:5000
"""
}

for filepath, content in files.items():
    d = os.path.dirname(filepath)
    if d: os.makedirs(d, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

for folder in ["dataset/A+","dataset/A-","dataset/B+","dataset/B-",
               "dataset/AB+","dataset/AB-","dataset/O+","dataset/O-",
               "model","processed","static/uploads","templates"]:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

print("\nProject created!")
print("Next: pip install -r requirements.txt")