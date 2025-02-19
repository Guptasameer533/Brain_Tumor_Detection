# Brain Tumor Detection

## 📌 Overview

This project is a **Brain Tumor Detection System** using **Convolutional Neural Networks (CNN)**. It supports **both CT scan and MRI scan** images to predict whether a brain tumor is present or not. The model is trained using **deep learning** techniques and integrated into a **Flask web application** for user-friendly predictions.

## 🏗️ Features

- **Supports both CT and MRI scans** for tumor prediction.
- **Pre-trained CNN model** for accurate classification.
- **Flask-based web app** with an interactive frontend.
- **Drag-and-drop functionality** for easy image upload.

## 📂 Project Structure

```
brain-tumor-detection/
│── notebooks/
│   ├── brain_tumor.ipynb  # ipynb notebook for model training
|
│── models/
│   ├── ct-tumor.keras     # Trained model for CT scans
│   ├── mri-tumor.keras    # Trained model for MRI scans
│
│── static/
│   ├── images/            # Sample Images
│
│── templates/
│   ├── index.html         # Home page of the Web App
│   ├── ct.html            # Webpage for uploading and predicting CT scans
│   ├── mri.html           # Webpage for uploading and predicting MRI scans
│
│── app.py                 # Flask backend for handling requests
│── requirements.txt       # Required Python dependencies
│── README.md              # Project documentation
```

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sarthak-436/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2️⃣ Install Dependencies

Ensure you have Python installed (recommended: Python 3.8+), then run:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App

```bash
python app.py
```

Open [**http://127.0.0.1:5000/**](http://127.0.0.1:5000/) in your browser.


⭐ **If you found this project useful, consider giving it a star!** ⭐

