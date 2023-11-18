from tensorflow.keras import models, layers

def create_cnn_model(input_shape,num_classes):
    """
    Creates a custom Convolutional Neural Network (CNN) model.

    Parameters:
    - input_shape (tuple): The shape of the input images (height, width, channels).
    - num_classes (int): The number of classes in the classification task.

    Returns:
    - model (tensorflow.keras.models.Sequential): The created CNN model.
    """

    # Create a Sequential model
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Model summary
    model.summary()

    return model
