from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load both models
ct_model = load_model('models/ct_model.keras')
mri_model = load_model('models/mri_model.keras')

# Class labels
class_labels = ['Healthy', 'Tumor']

# Function to predict an image
def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize

    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    return class_labels[predicted_class_index]

# Homepage route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# CT scan upload route
@app.route("/ct", methods=["GET", "POST"])
def ct_scan():
    ct_prediction = None
    ct_image = None

    if request.method == "POST":
        if "ct_file" in request.files and request.files["ct_file"].filename != "":
            ct_file = request.files["ct_file"]
            ct_filepath = os.path.join("static", ct_file.filename)
            ct_file.save(ct_filepath)
            ct_prediction = predict_image(ct_filepath, ct_model)
            ct_image = ct_filepath

    return render_template("ct.html", ct_prediction=ct_prediction, ct_image=ct_image)

# MRI scan upload route
@app.route("/mri", methods=["GET", "POST"])
def mri_scan():
    mri_prediction = None
    mri_image = None

    if request.method == "POST":
        if "mri_file" in request.files and request.files["mri_file"].filename != "":
            mri_file = request.files["mri_file"]
            mri_filepath = os.path.join("static", mri_file.filename)
            mri_file.save(mri_filepath)
            mri_prediction = predict_image(mri_filepath, mri_model)
            mri_image = mri_filepath

    return render_template("mri.html", mri_prediction=mri_prediction, mri_image=mri_image)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
