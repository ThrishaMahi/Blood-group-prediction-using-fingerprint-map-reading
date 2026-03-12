
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
