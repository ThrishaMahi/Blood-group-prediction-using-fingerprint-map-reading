
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
