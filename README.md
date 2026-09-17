# RetinaScan — Diabetic Retinopathy Detection

> A faculty-supervised educational project using a custom CNN and Flask web application to classify retinal fundus images into five diabetic-retinopathy severity grades.

## Overview

RetinaScan explores an end-to-end computer-vision workflow: image preprocessing, CNN training, model evaluation, inference, and deployment through a web interface.

The application accepts a retinal fundus image and returns a predicted severity grade with a confidence breakdown. It is intended for educational and research use only.

## Severity Classes

| Grade | Classification |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Tech Stack

- **Language:** Python 3.13
- **Deep Learning:** TensorFlow / Keras
- **Web:** Flask
- **Data:** NumPy, Pandas, scikit-learn
- **Image Processing:** Pillow
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3, JavaScript, Three.js

## Dataset

The project uses a 224×224 variant of the APTOS 2019 Blindness Detection dataset containing 3,644 images across five classes. The dataset itself is not included in this repository.

## Model

The current implementation uses a custom CNN rather than transfer learning:

```text
224×224 RGB image
        ↓
Convolutional Block × 3
32 → 64 → 128 filters
        ↓
Max Pooling + Dropout
        ↓
Dense Classification Head
        ↓
5-class Softmax
```

Training includes light image augmentation and class weighting to address class imbalance.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Start the web application:

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## Limitations

The dataset is imbalanced, with common classes substantially better represented than rarer severity grades. This can affect classification reliability. Model results should therefore be interpreted in the context of the dataset and evaluation methodology.

## Disclaimer

This application is **not a medical device** and must not be used for clinical diagnosis or treatment decisions. It was developed for educational and research purposes.

## Portfolio Focus

This project demonstrates computer vision, deep learning, image preprocessing, model training, class-imbalance handling, Flask inference APIs, and web-based ML deployment.

## Author

**Arihant Gupta** — BCA student at JIIT Noida

[GitHub](https://github.com/arihxnt786)
