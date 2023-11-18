import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.metrics import Precision, Recall, AUC

def apply_weight_clustering(model, num_of_clusters, clustering_learning_rate):
    """
    Apply weight clustering to the given model.

    Parameters:
    - model (tf.keras.Model): The input model to be clustered.
    - num_of_clusters (int): The number of clusters to use for weight clustering.
    - clustering_learning_rate (float): The learning rate for fine-tuning the clustered model.

    Returns:
    - tf.keras.Model: The clustered model.
    """

    # Import required modules from TensorFlow Model Optimization
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    # Set clustering parameters
    clustering_params = {
        'number_of_clusters': num_of_clusters,
        'cluster_centroids_init': CentroidInitialization.LINEAR
    }

    # Cluster the whole model
    clustered_model = cluster_weights(model, **clustering_params)

    # Use a smaller learning rate for fine-tuning the clustered model
    opt = tf.keras.optimizers.Adam(learning_rate=clustering_learning_rate)

    # Compile the clustered model with specified metrics
    clustered_model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc_roc'),
            AUC(name='auc_pr', curve='PR')
        ]
    )

    # Display a summary of the clustered model
    clustered_model.summary()

    return clustered_model

