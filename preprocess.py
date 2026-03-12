
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
