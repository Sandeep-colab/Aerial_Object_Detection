import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import os
# Optional: Import YOLO if detection is included
# from ultralytics import YOLO 

# --- Configuration ---
CLASSIFICATION_MODEL_PATH = 'models/transfer_learning_best.keras' 
# Assuming Transfer Learning was the best model. ADJUST THIS PATH!
# YOLO_MODEL_PATH = 'yolov8/best.pt' # Path to your trained YOLOv8 model
CLASS_NAMES = ['Bird', 'Drone']
IMG_SIZE = (224, 224)

# --- Load Models (Cached for Performance) ---
@st.cache_resource
def load_classification_model(path):
    """Loads the Keras classification model."""
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None

# @st.cache_resource
# def load_detection_model(path):
#     """Loads the YOLOv8 object detection model."""
#     try:
#         model = YOLO(path)
#         return model
#     except Exception as e:
#         st.error(f"Error loading YOLOv8 model: {e}")
#         return None

clf_model = load_classification_model(CLASSIFICATION_MODEL_PATH)
# det_model = load_detection_model(YOLO_MODEL_PATH)

# --- Helper Functions ---
def run_classification(image: Image.Image, model: tf.keras.Model):
    """Preprocesses the image and runs the classification model."""
    # Resize and convert to array
    img_resized = image.resize(IMG_SIZE)
    # The normalization layer is built into the model (in transfer.py), so we don't normalize here
    img_array = np.array(img_resized) 
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0) 

    # Predict
    prediction = model.predict(img_array, verbose=0)[0]
    
    # For binary classification (sigmoid output)
    confidence = prediction[0]
    predicted_class_idx = 1 if confidence >= 0.5 else 0
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    # Format confidence
    final_confidence = confidence if predicted_class_idx == 1 else 1.0 - confidence
    
    return predicted_class, final_confidence

# def run_detection(image: Image.Image, model: YOLO):
#     """Runs the YOLOv8 object detection model."""
#     results = model(image, verbose=False)
#     return results[0].plot() # Returns the annotated image array

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Aerial Object Detection & Classification", layout="wide")
    st.title("🦅 Bird vs. Drone Aerial Object Classification")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.header("1. Upload Image")
        uploaded_file = st.file_uploader(
            "Upload an aerial image (jpg/png):", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is None:
            st.info("Awaiting image upload...")
            return

        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
    if clf_model:
        with col2:
            st.header("2. Analysis Results")
            st.subheader("Classification (Bird or Drone)")
            
            # Run Classification
            predicted_class, confidence = run_classification(image, clf_model)
            
            if predicted_class == 'Drone':
                st.error(f"🚨 Predicted Object: **DRONE**")
            else:
                st.success(f"🕊️ Predicted Object: **BIRD**")
            
            st.metric(label=f"Confidence in {predicted_class}", value=f"{confidence*100:.2f}%")

            st.markdown("---")
            
            # --- Optional: Object Detection ---
            # if det_model and st.checkbox("Show Object Detection (YOLOv8)"):
            #     st.subheader("Object Detection Results")
            #     with st.spinner('Running YOLOv8 Detection...'):
            #         annotated_img = run_detection(image, det_model)
            #         st.image(annotated_img, caption='Object Detection with Bounding Boxes', use_column_width=True)


if __name__ == "__main__":
    main()