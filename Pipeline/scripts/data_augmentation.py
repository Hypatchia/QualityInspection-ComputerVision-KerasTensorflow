import tensorflow as tf
# Create a function to augemnt a dataset
def augment_dataset(dataset, training=True):
    """
    Apply data augmentation to a dataset.

    Parameters:
    - dataset (tf.data.Dataset): Input training dataset.
    - training (bool, optional): Whether to apply data augmentation. Default is True.

    Returns:
    - augmented_dataset (tf.data.Dataset): Augmented training dataset if training is True, else the original dataset.
    """
    # Define the data augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2),
    ])

    # Apply data augmentation to the dataset if training is True
    augmented_dataset = dataset.map(lambda x, y: (data_augmentation(x, training=training), y))

    return augmented_dataset
