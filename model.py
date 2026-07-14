from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'dr_model_best.keras'
IMG_SIZE = (224, 224)

CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        from tensorflow.keras.models import load_model

        _model = load_model(str(MODEL_PATH), compile=False)
    return _model

def preprocess_image(image_path, size=IMG_SIZE):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(size)

    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_grade(image_path):
    img_array = preprocess_image(image_path)

    predictions = get_model().predict(img_array)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    all_confidences = {
        CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        'predicted_class': predicted_class,
        'predicted_index': int(predicted_index),
        'confidence': round(confidence, 2),
        'all_confidences': all_confidences
    }
