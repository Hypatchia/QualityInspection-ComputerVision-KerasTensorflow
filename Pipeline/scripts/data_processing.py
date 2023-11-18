
import tensorflow as tf
def preprocess_image(image, label):
    """
    Preprocess an image by converting it to grayscale.

    Parameters:
    - image (tf.Tensor): Input image tensor.
    - label: Label corresponding to the image.

    Returns:
    - preprocessed_image (tf.Tensor): Preprocessed image tensor.
    - label: Unchanged label.
    """
    # Convert the image to grayscale
    preprocessed_image = tf.image.rgb_to_grayscale(image)

    return preprocessed_image, label
