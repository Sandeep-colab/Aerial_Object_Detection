import tensorflow as tf
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import glob
# Updated Import
from src.data_utils import load_classification_datasets

def evaluate_model(model_path, test_ds, class_names):
    """Loads a model and evaluates it on the test dataset."""
    
    model_name = os.path.basename(model_path).split('_best')[0]
    print(f"\n--- Evaluating Model: {model_name} ---")

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # 1. Get predictions
    y_pred_probs = model.predict(test_ds)
    y_pred = np.round(y_pred_probs).astype(int)

    # 2. Extract true labels
    # NOTE: Reloading test_ds_raw to ensure deterministic label order
    _, _, test_ds_raw, _ = load_classification_datasets()
    y_true = np.concatenate([y for x, y in test_ds_raw], axis=0)
    y_true = y_true[:len(y_pred)] # Align lengths
    
    # 3. Generate Classification Report and Confusion Matrix
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    return test_acc

def main():
    # Load test data and class names once
    _, _, test_ds, class_names = load_classification_datasets()

    model_files = glob.glob('models/*_best.keras')
    
    if not model_files:
        print("No models found in the 'models/' directory. Please run training first.")
        return

    for model_path in model_files:
        evaluate_model(model_path, test_ds, class_names)

if __name__ == '__main__':
    # To run this script correctly with the package structure, you must 
    # run it from the root directory using:
    # python -m src.eval 
    main()