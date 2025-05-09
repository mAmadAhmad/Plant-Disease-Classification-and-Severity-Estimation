import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Load the model
model = None


def load_model():
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model('model_disease_classifier.keras')
            print("Model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    return model


def preprocess_image(img, img_size=(224, 224)):
    """Preprocess image for model prediction"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return img


def predict_disease(img):
    """
    Predict plant disease from image
    Returns: (class_index, confidence_percentage, top_classes)
    """
    # Load model if not already loaded
    model = load_model()

    # Preprocess image
    processed_img = preprocess_image(img)

    # Prepare for prediction
    X_test = np.array([processed_img])

    # Get prediction
    y_pred_probs = model.predict(X_test)

    # Get predicted class
    predicted_class = np.argmax(y_pred_probs[0])
    confidence = float(y_pred_probs[0][predicted_class] * 100)

    # Get top classes with probabilities
    top_indices = np.argsort(y_pred_probs[0])[::-1][:5]  # Get top 5 indices
    top_classes = [(int(idx), float(y_pred_probs[0][idx] * 100)) for idx in top_indices]

    return predicted_class, confidence, top_classes


def get_disease_severity(img):
    """
    Estimate disease severity from image
    Returns: (severity_percentage, visualization_image)
    """
    # Resize and convert to HSV
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Step 1: Leaf Segmentation (exclude background) ---
    leaf_lower = np.array([10, 40, 40])
    leaf_upper = np.array([85, 255, 255])
    leaf_mask = cv2.inRange(hsv, leaf_lower, leaf_upper)

    # Optional: Smooth leaf mask
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # --- Step 2: Disease Region Masking ---
    disease_lower = np.array([10, 50, 50])
    disease_upper = np.array([30, 255, 255])
    disease_mask = cv2.inRange(hsv, disease_lower, disease_upper)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # --- Step 3: Infection within Leaf Area Only ---
    infected_region = cv2.bitwise_and(disease_mask, disease_mask, mask=leaf_mask)
    infected_pixels = np.sum(infected_region > 0)
    leaf_pixels = np.sum(leaf_mask > 0)

    # Calculate severity
    severity = (infected_pixels / leaf_pixels) * 100 if leaf_pixels > 0 else 0

    return severity, infected_region