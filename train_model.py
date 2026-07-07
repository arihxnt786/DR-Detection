"""
train_model.py
Diabetic Retinopathy Detection - CNN Model Building & Training (Final v4)
Week 2 (Days 8-14, final rebuild)

PROGRESS SO FAR:
v3 (capped class weights, max 2.5x): 75.0% accuracy, 2.1% gap - excellent
overall numbers, but Proliferate_DR recall was still 0%, with PDR cases
being absorbed almost entirely into the much larger Moderate class (999
training images vs PDR's 295). Severe was also weak (15% recall).

DIAGNOSIS: capping ALL class weights at 2.5x under-corrected specifically
for PDR and Severe, since Moderate's 5x sample-size advantage over PDR
needed a stronger counterweight than a uniform cap allowed.

CHANGES IN THIS VERSION (targeted, not a rebuild):
1. Selective class weighting - No_DR/Mild/Moderate keep their natural
   balanced weights (these already work well, don't disturb them), while
   Severe and Proliferate_DR get their FULL uncapped balanced weight,
   specifically countering the Moderate-class gravitational pull seen
   in the v3 confusion matrix.
2. Label smoothing (0.1) added to the loss function - a standard,
   architecture-safe technique that prevents the model from being
   overconfident on majority classes, giving minority classes more room
   in the probability distribution. Does not touch model layers, so it
   carries none of the instability risk BatchNorm caused earlier.
3. Everything else (architecture, augmentation, EarlyStopping, epoch
   ceiling) is UNCHANGED from v3, since those parts are proven stable
   and produced healthy, clean training curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import preprocess


CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']


# ---------- Model Architecture (unchanged from v3 - proven stable) ----------
def build_cnn_model(input_shape=(224, 224, 3), num_classes=5):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])
    return model


def compile_model(model):
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model


def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, acc, 'b-', label='Training Accuracy')
    axes[0].plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
    axes[0].set_title('Training vs Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, loss, 'b-', label='Training Loss')
    axes[1].plot(epochs_range, val_loss, 'r-', label='Validation Loss')
    axes[1].set_title('Training vs Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def analyze_overfitting(history):
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    gap = final_train_acc - final_val_acc

    print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Accuracy Gap: {gap:.4f}")

    if abs(gap) > 0.10:
        print("Significant gap detected (>10%)")
    else:
        print("Gap within acceptable range")

    best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f} at epoch {best_epoch}")


def evaluate_model(model, X_test, y_test_enc, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test_enc, verbose=1)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - DR Severity Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    return test_accuracy, cm, report


# ---------- Data Augmentation (unchanged, proven settings) ----------
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()


def compute_weights_selective(y_train, boost_classes=(3, 4), cap_others=2.5):
    """
    Selective weighting strategy:
    - Classes in boost_classes (default: Severe=3, Proliferate_DR=4) get
      their full, uncapped 'balanced' weight - these are the classes
      being absorbed into Moderate and need the strongest correction.
    - All other classes (No_DR, Mild, Moderate) get balanced weights
      capped at `cap_others` times the minimum weight, since these
      already perform well and over-weighting them further would only
      hurt overall accuracy without benefit.
    """
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    min_w = raw_weights.min()

    final_weights = {}
    for i, w in enumerate(raw_weights):
        if i in boost_classes:
            final_weights[i] = w  # full, uncapped balanced weight
        else:
            final_weights[i] = min(w, min_w * cap_others)

    return final_weights


# ---------- Training Configuration (unchanged from v3) ----------
BATCH_SIZE = 32
EPOCHS = 50

checkpoint_callback = ModelCheckpoint(
    'dr_model_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)


if __name__ == "__main__":
    # ---------- Step 1: Load and preprocess data ----------
    print("Loading and preprocessing dataset...\n")
    X, y, class_counts = preprocess.preprocess_dataset()
    X_train, X_test, y_train_enc, y_test_enc, y_train, y_test = preprocess.split_and_encode(X, y)

    print("\nTrain set:", X_train.shape, y_train_enc.shape)
    print("Test set:", X_test.shape, y_test_enc.shape)

    # ---------- Step 2: Build and compile model ----------
    model = build_cnn_model()
    model = compile_model(model)
    model.summary()

    # ---------- Step 3: Augmentation + selective class weights ----------
    train_generator = train_datagen.flow(X_train, y_train_enc, batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow(X_test, y_test_enc, batch_size=BATCH_SIZE)

    class_weights = compute_weights_selective(y_train, boost_classes=(3, 4), cap_others=2.5)
    print("\nClass weights (selective - Severe & PDR uncapped, others capped 2.5x):", class_weights)

    # ---------- Step 4: Train ----------
    print("\nStarting training with light augmentation + selective class weights + label smoothing...\n")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    print("\nTraining complete!")
    analyze_overfitting(history)
    plot_training_history(history)

    # ---------- Step 5: Evaluate ----------
    evaluate_model(model, X_test, y_test_enc, y_test)