from flask import Flask, render_template, request
import os
import json
import time
from pathlib import Path
import model

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
HISTORY_FILE = BASE_DIR / 'predictions_history.json'
MAX_HISTORY = 6
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

GRADE_INFO = {
    0: {
        'name': 'No_DR',
        'color': '#3DDC97',
        'description': 'No signs of diabetic retinopathy were detected in this image.',
        'recommendation': 'Continue regular annual eye exams to monitor for any future changes.'
    },
    1: {
        'name': 'Mild',
        'color': '#A8D86B',
        'description': 'Microaneurysms are present — this is the earliest detectable stage of diabetic retinopathy.',
        'recommendation': 'Schedule a follow-up with an ophthalmologist within the next 12 months.'
    },
    2: {
        'name': 'Moderate',
        'color': '#F2B84B',
        'description': 'More blood vessels are blocked and some lesions are visible in the retina.',
        'recommendation': 'An ophthalmologist review is recommended within the next 6 months.'
    },
    3: {
        'name': 'Severe',
        'color': '#F2864B',
        'description': 'Many blood vessels are blocked, indicating a high risk of disease progression.',
        'recommendation': 'Urgent referral to an ophthalmologist is required.'
    },
    4: {
        'name': 'Proliferate_DR',
        'color': '#FF6B5B',
        'description': 'Abnormal new blood vessel growth detected — this is the most advanced and vision-threatening stage.',
        'recommendation': 'Immediate specialist consultation is strongly advised.'
    }
}

CLASS_COLORS = {
    'No_DR': '#3DDC97',
    'Mild': '#A8D86B',
    'Moderate': '#F2B84B',
    'Severe': '#F2864B',
    'Proliferate_DR': '#FF6B5B'
}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_to_history(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:MAX_HISTORY]
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except IOError:
        pass
    return history

@app.errorhandler(413)
def file_too_large(e):
    return (
        "The uploaded file is too large. Please upload an image under 10 MB.",
        413
    )

@app.route('/')
def home():
    recent_predictions = load_history()
    return render_template('index.html', recent_predictions=recent_predictions)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']

    if file.filename == '':
        return "No file selected", 400

    if not allowed_file(file.filename):
        return (
            "Invalid file type. Please upload a PNG or JPEG retinal image.",
            400
        )

    try:
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_name = f"{int(time.time() * 1000)}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

        from PIL import Image
        try:
            with Image.open(filepath) as test_img:
                test_img.verify()
        except Exception:
            os.remove(filepath)
            return (
                "The uploaded file could not be read as a valid image. "
                "Please try a different file.",
                400
            )

        result = model.predict_grade(filepath)
        grade_info = GRADE_INFO[result['predicted_index']]

        save_to_history({
            'filename': unique_name,
            'predicted_class': result['predicted_class'],
            'predicted_index': result['predicted_index'],
            'confidence': result['confidence'],
            'grade_color': grade_info['color'],
            'timestamp': time.strftime('%d %b %Y, %H:%M')
        })

        return render_template(
            'result.html',
            filename=unique_name,
            predicted_class=result['predicted_class'],
            predicted_index=result['predicted_index'],
            confidence=result['confidence'],
            all_confidences=result['all_confidences'],
            description=grade_info['description'],
            recommendation=grade_info['recommendation'],
            grade_color=grade_info['color'],
            class_colors=CLASS_COLORS
        )

    except Exception as e:
        return f"An error occurred while processing the image: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
