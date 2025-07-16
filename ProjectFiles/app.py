from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = tf.keras.models.load_model("model/blood_model.h5")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load class names in correct order
with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)
class_names = [None] * len(class_indices)
for name, index in class_indices.items():
    class_names[index] = name

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        pred = model.predict(img_array)
        class_index = np.argmax(pred)
        prediction = class_names[class_index]

        return render_template("prediction.html", prediction=prediction, img_path=filepath)

@app.route('/logout')
def logout():
    return render_template("logout.html")

if __name__ == "__main__":
    app.run(debug=True)
