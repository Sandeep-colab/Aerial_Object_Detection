import tensorflow as tf
import os

# --- Configuration Constants ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# Assumes the script is run from the root directory ('aerial-bird-drone/')
DATA_ROOT = 'data/class_dataset' 
AUTOTUNE = tf.data.AUTOTUNE

def load_classification_datasets():
    """Loads and optimizes the Bird/Drone classification datasets."""
    
    def load_subset(subset_name):
        print(f"Loading {subset_name} Data...")
        return tf.keras.utils.image_dataset_from_directory(
            directory=os.path.join(DATA_ROOT, subset_name),
            labels='inferred',
            label_mode='binary',
            image_size=IMG_SIZE,
            interpolation='nearest',
            batch_size=BATCH_SIZE,
            shuffle=(subset_name == 'TRAIN'),
            seed=42
        )

    train_ds = load_subset('TRAIN')
    val_ds = load_subset('VALID')
    test_ds = load_subset('TEST')
    
    # Get class names
    class_names = train_ds.class_names

    # Optimize datasets
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("Data loading complete and pipelines optimized.")
    return train_ds, val_ds, test_ds, class_names

def get_preprocessing_layers():
    """Returns the normalization and data augmentation layers."""
    
    # 1. Normalization (Rescaling pixel values from [0, 255] to [0, 1])
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    # 2. Data Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ], name='data_augmentation')
    
    return normalization_layer, data_augmentation

if __name__ == '__main__':
    # Example usage and sanity check
    train_ds, _, _, class_names = load_classification_datasets()
    print(f"\nClass Names: {class_names}")
    
    norm, aug = get_preprocessing_layers()
    print("Preprocessing and Augmentation layers created successfully.")