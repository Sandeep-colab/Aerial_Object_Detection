import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn_model(input_shape, normalization_layer, augmentation_layer):
    """
    Builds a custom Convolutional Neural Network (CNN) model.
    """
    model = models.Sequential([
        # 1. Preprocessing and Augmentation Layers
        normalization_layer,
        augmentation_layer,

        # 2. Convolutional Blocks
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # 3. Dense Head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        # 4. Output Layer (Binary Classification)
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ], name='Custom_Bird_Drone_CNN')
    
    return model

if __name__ == '__main__':
    from data_utils import IMG_SIZE, get_preprocessing_layers
    norm, aug = get_preprocessing_layers()
    input_shape = IMG_SIZE + (3,)
    model = build_custom_cnn_model(input_shape, norm, aug)
    model.summary()