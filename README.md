# RetinaScan — Diabetic Retinopathy Detection Web App

A web application that uses a Convolutional Neural Network (CNN) to classify retinal fundus images into one of five diabetic retinopathy (DR) severity grades, built as a faculty-supervised group internship project.

---

## Overview

Diabetic Retinopathy is a progressive eye disease caused by prolonged high blood sugar levels, and one of the leading causes of preventable blindness worldwide. This project provides an automated, web-based screening tool that classifies a retinal image into one of five severity grades, giving an instant preliminary assessment.

**Severity grades detected:**

| Grade | Name | Description |
|---|---|---|
| 0 | No DR | No signs of diabetic retinopathy |
| 1 | Mild | Microaneurysms present — earliest detectable stage |
| 2 | Moderate | More blocked vessels, visible lesions |
| 3 | Severe | Many blocked vessels, high risk of progression |
| 4 | Proliferative DR | Abnormal new vessel growth — most advanced stage |

---

## Tech Stack

- **Language:** Python 3.13
- **ML Framework:** TensorFlow / Keras
- **Web Framework:** Flask
- **Data Handling:** NumPy, Pandas, scikit-learn
- **Image Processing:** Pillow (PIL)
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3, JavaScript, Three.js (r128)

---

## Dataset

This project uses a 224x224 pre-resized variant of the **APTOS 2019 Blindness Detection** dataset, sourced via Kaggle:

🔗 [diabetic-retinopathy-224x224-2019-data](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data)

- **Total images:** 3,644
- **Format:** PNG, already resized to 224x224
- **Structure:** folder-per-class (`No_DR/`, `Mild/`, `Moderate/`, `Severe/`, `Proliferate_DR/`)

---

## Folder Structure

```
DR-Detection-Project/
├── preprocess.py              # Data loading, preprocessing, visualization
├── train_model.py             # CNN architecture, training, evaluation
├── model.py                   # Loads trained model, runs predictions
├── app.py                     # Flask application and routes
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── dr_model_best.keras        # Trained model (generated after training)
├── predictions_history.json   # Auto-generated prediction history
├── templates/
│   ├── index.html             # Home page (upload interface)
│   └── result.html            # Prediction results page
├── static/
│   ├── css/
│   │   └── style.css          # Application stylesheet
│   └── uploads/                # Uploaded images stored here
└── train_images/               # Dataset (not included — see Dataset section)
```

---

## Installation

1. **Clone or download this repository** into a local folder.

2. **Install Python 3.13** if not already installed.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset** from the Kaggle link above and place it in a `train_images/` folder in the project root, organized by class (see Folder Structure above).

---

## How to Run

### 1. Train the model (first time only)

```bash
python train_model.py
```

This will preprocess the dataset, train the CNN, and save the best-performing model as `dr_model_best.keras`. Training takes approximately 20-40 minutes on a CPU-only machine, depending on hardware.

> **Note:** TensorFlow does not support native GPU acceleration on Windows for versions 2.11+. Training runs on CPU; this is expected behavior, not an error.

### 2. Run the web application

```bash
python app.py
```

### 3. Open the app

Navigate to **http://127.0.0.1:5000** in your browser.

### 4. Use the app

- Drag and drop a retinal image (PNG/JPEG) into the circular upload area, or click to browse
- Click **Analyze image**
- View the predicted severity grade, confidence breakdown, and clinical recommendation

---

## Model Details

- **Architecture:** Custom CNN built from scratch (no transfer learning) — 3 convolutional blocks (32→64→128 filters) with max pooling, followed by a dense classification head with dropout regularization
- **Input:** 224×224×3 RGB images
- **Output:** 5-class softmax (one probability per severity grade)
- **Training enhancements:** Light data augmentation (rotation, shift, zoom, flip) and class weighting to address dataset imbalance

**Known limitation:** The dataset has significant class imbalance (No_DR images substantially outnumber Severe and Proliferative DR images). This affects the model's ability to detect rarer severity grades as reliably as common ones — an inherent challenge of the available data rather than the modeling approach. See the full Project Documentation report for detailed metrics.

---

## Disclaimer

This application is built for **educational and research purposes only**. It is not a certified medical device and must not be used as a substitute for professional diagnosis by a qualified ophthalmologist.

---

