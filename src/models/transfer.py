import tensorflow as tf
from keras import layers
from keras.applications import MobileNetV2
from keras.models import Model
from src.data_utils import IMG_SIZE # Assuming src is in your path

def build_transfer_learning_model(input_shape, normalization_layer, augmentation_layer):
    """
    Builds a transfer learning model using MobileNetV2 base.
    """
    
    # 1. Load the pre-trained MobileNetV2 base
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False, # We'll add our own classification head
        weights='imagenet' # Use weights trained on ImageNet
    )

    # 2. Freeze the base model's weights during initial training
    base_model.trainable = False

    # 3. Create the input layer with preprocessing/augmentation
    inputs = tf.keras.Input(shape=input_shape)
    x = normalization_layer(inputs)
    x = augmentation_layer(x)
    
    # 4. Pass through the frozen base model
    x = base_model(x, training=False) # Important: set training=False when base is frozen

    # 5. Add the new classification head
    x = layers.GlobalAveragePooling2D()(x) # Reduces spatial dimensions
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output_layer')(x)

    # 6. Create the final Model
    model = Model(inputs=inputs, outputs=outputs, name='MobileNetV2_Transfer')
    
    return model

if __name__ == '__main__':
    from src.data_utils import get_preprocessing_layers
    norm, aug = get_preprocessing_layers()
    input_shape = IMG_SIZE + (3,)
    model = build_transfer_learning_model(input_shape, norm, aug)
    model.summary()