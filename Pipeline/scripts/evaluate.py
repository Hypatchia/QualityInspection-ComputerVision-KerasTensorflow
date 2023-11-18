import matplotlib.pyplot as plt

def evaluate_training(model_history):
    """
    Evaluate and visualize the training performance of a neural network model.

    Parameters:
    - model_history (tensorflow.keras.callbacks.History): The history object obtained from model training.

    Returns:
    - None: Displays plots of training accuracy and training/validation loss.
    """

    # Extract metrics from history
    acc = model_history.history['accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs_range = range(len(acc))

    # Create a figure for subplots
    plt.figure(figsize=(12, 16))

    # Plot accuracy metrics
    plt.subplot(3, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training Accuracy')

    # Plot loss metrics
    plt.subplot(3, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Adjust layout for better visualization
    plt.tight_layout()

    # Show the plots
    plt.show()
