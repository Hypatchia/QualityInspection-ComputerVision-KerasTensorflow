
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
def create_data_generators(data_dir, target_size, batch_size, class_mode, split=0.2, test_dir=None):
    """
    Creates train, validation, and test tf data generators.

    Parameters:
    - data_dir (str): Directory path containing the dataset.
    - target_size (tuple): Tuple specifying the height and width to which all images will be resized.
    - batch_size (int): Size of the batches of data.
    - class_mode (str): One of "binary", "categorical", "sparse", or "input". Defines the type of label arrays that are returned.
    - split (float): Fraction of the dataset to allocate to validation.
    - test_dir (str): Directory path containing the test dataset.

    Returns:
    - train_generator (tf.keras.preprocessing.image.DirectoryIterator): Train data generator.
    - validation_generator (tf.keras.preprocessing.image.DirectoryIterator): Validation data generator.
    - test_generator (tf.keras.preprocessing.image.DirectoryIterator): Test data generator.
    """
    datagen = ImageDataGenerator(
        validation_split=split,
        rescale=1.0 / 255.0,
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='training',
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='validation',
    )

    if test_dir:
        test_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
        )

        test_generator = test_datagen.flow_from_directory(
            directory=test_dir,
            target_size=target_size,
            batch_size=1,
            class_mode='sparse',  # Adjust class_mode based on your requirement
            shuffle=False,
        )

        return train_generator, validation_generator, test_generator
    else:
        return train_generator, validation_generator
    

def get_labels_from_generator(generator):
    """
    Get labels from a data generator.

    Parameters:
    - generator: Data generator yielding batches of data.

    Returns:
    - labels (list): List of labels extracted from the generator.
    """
    labels = []

    # Iterate through the generator to extract batch labels
    for _ in range(len(generator)):
        _, batch_labels = next(generator)
        labels.extend(batch_labels)

    return labels



def dataset_from_gen(generator, target_size):
    """
    Create a TensorFlow dataset from a data generator.

    Parameters:
    - generator: Data generator providing batches of data.
    - target_size (tuple): Tuple specifying the height and width to which all images will be resized.

    Returns:
    - dataset (tf.data.Dataset): TensorFlow dataset created from the data generator.
    """
    # Define the output signature of the dataset
    output_signature = (
        tf.TensorSpec(shape=(None, *target_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )

    # Create a TensorFlow dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=output_signature
    )

    return dataset
