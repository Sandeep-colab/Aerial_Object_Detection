import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
# Updated Imports using package structure
from src.data_utils import load_classification_datasets, get_preprocessing_layers, IMG_SIZE
from src.models.cnn import build_custom_cnn_model
from src.models.transfer import build_transfer_learning_model
import matplotlib.pyplot as plt
import os

EPOCHS = 30
MODEL_DIR = 'models/'
os.makedirs(MODEL_DIR, exist_ok=True)

def compile_and_train(model, train_ds, val_ds, model_name):
    """Compiles and trains the given model."""
    
    print(f"\n--- Starting Training for {model_name} ---")

    # Define Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f'{model_name}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    return history, os.path.join(MODEL_DIR, f'{model_name}_best.keras')

def plot_history(history, model_name):
    """Plots the training and validation accuracy and loss."""
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} Training and Validation Loss')
    
    plt.savefig(f'{MODEL_DIR}/{model_name}_history.png')
    plt.show()

def main():
    # 1. Load Data and Layers
    train_ds, val_ds, _, _ = load_classification_datasets()
    normalization_layer, augmentation_layer = get_preprocessing_layers()
    input_shape = IMG_SIZE + (3,)

    # --- Train Custom CNN ---
    cnn_model = build_custom_cnn_model(input_shape, normalization_layer, augmentation_layer)
    cnn_history, cnn_path = compile_and_train(cnn_model, train_ds, val_ds, 'custom_cnn')
    plot_history(cnn_history, 'custom_cnn')

    # --- Train Transfer Learning Model ---
    tf_model = build_transfer_learning_model(input_shape, normalization_layer, augmentation_layer)
    tf_history, tf_path = compile_and_train(tf_model, train_ds, val_ds, 'transfer_learning')
    plot_history(tf_history, 'transfer_learning')

if __name__ == '__main__':
    # To run this script correctly with the package structure, you must 
    # run it from the root directory using:
    # python -m src.models.train 
    main()