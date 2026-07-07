"""
preprocess.py
Diabetic Retinopathy Detection Web App - Data Preprocessing Module
Week 1 (Days 1-7)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ---------- Configuration ----------
IMAGE_DIR = 'train_images/'
IMG_SIZE = (224, 224)

CLASS_MAP = {
    'No_DR': 0,
    'Mild': 1,
    'Moderate': 2,
    'Severe': 3,
    'Proliferate_DR': 4
}

CLASS_NAMES = list(CLASS_MAP.keys())
CLASS_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8b0000']


def load_image(img_path, size=IMG_SIZE):
    """Load a single image, convert to RGB, resize if needed."""
    img = Image.open(img_path)
    img = img.convert('RGB')
    if img.size != size:
        img = img.resize(size)
    return np.array(img)


def normalize_image(img_array):
    """Scale pixel values from 0-255 to 0-1."""
    return img_array.astype('float32') / 255.0


def preprocess_dataset(base_dir=IMAGE_DIR, size=IMG_SIZE, limit_per_class=None):
    """
    Walk through all class folders, load + preprocess images,
    and return (images, labels, class_counts) as NumPy arrays + dict.
    """
    images = []
    labels = []
    class_counts = {}

    for folder_name, label in CLASS_MAP.items():
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f"Warning: folder not found - {folder_path}")
            continue

        filenames = os.listdir(folder_path)
        if limit_per_class is not None:
            filenames = filenames[:limit_per_class]

        for fname in filenames:
            try:
                img_path = os.path.join(folder_path, fname)
                img = load_image(img_path, size)
                img = normalize_image(img)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Skipped {fname}: {e}")

        class_counts[folder_name] = len(filenames)
        print(f"Loaded {len(filenames)} images from {folder_name} (label {label})")

    return np.array(images), np.array(labels), class_counts


def split_and_encode(X, y, test_size=0.2, num_classes=5):
    """Split into train/test sets and one-hot encode labels."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
    y_test_encoded = to_categorical(y_test, num_classes=num_classes)
    return X_train, X_test, y_train_encoded, y_test_encoded, y_train, y_test


def plot_class_distribution(class_counts, save_path='grade_distribution.png'):
    """Bar chart of image count per DR grade."""
    counts = [class_counts.get(name, 0) for name in CLASS_NAMES]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(CLASS_NAMES, counts, color=CLASS_COLORS)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                  str(count), ha='center', fontsize=11, fontweight='bold')

    plt.title('Diabetic Retinopathy Dataset - Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('DR Severity Grade')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_sample_images(base_dir=IMAGE_DIR, save_path='sample_images.png'):
    """Display one sample image from each class side by side."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for idx, (folder_name, label) in enumerate(CLASS_MAP.items()):
        folder_path = os.path.join(base_dir, folder_name)
        sample_file = os.listdir(folder_path)[0]
        img_path = os.path.join(folder_path, sample_file)

        img = mpimg.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(f'{folder_name}\n(Grade {label})', fontsize=12, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle('Sample Retinal Images by DR Severity Grade', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    print("Starting preprocessing pipeline...\n")

    X, y, class_counts = preprocess_dataset()
    print("\nImages shape:", X.shape)
    print("Labels shape:", y.shape)

    X_train, X_test, y_train_enc, y_test_enc, y_train, y_test = split_and_encode(X, y)
    print("\nTrain set:", X_train.shape, y_train_enc.shape)
    print("Test set:", X_test.shape, y_test_enc.shape)

    print("\nGenerating visualizations...")
    plot_class_distribution(class_counts)
    plot_sample_images()

    print("\nWeek 1 preprocessing complete!")