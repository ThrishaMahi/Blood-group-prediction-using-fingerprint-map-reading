
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
