import mlflow
from scripts.data_loading import extract_zip_dataset, load_from_local
from scripts.data_split_creation import create_data_generators, get_labels_from_generator, dataset_from_gen
from scripts.train import create_cnn_model
from scripts.test import make_predictions
from scripts.data_augmentation import augment_dataset
from scripts.evaluate import evaluate_training
from scripts.model_compression import apply_weight_clustering
import tensorflow_model_optimization as tfmot
import tensorflow as tf 



def main():
    
    # Set the paths to the dataset file and the extraction path
    print("Loading data...")
    dataset_file_path = '../Data/Products.zip'
    extraction_path = '../Data/'
    # Extract dataset
    success = extract_zip_dataset(dataset_file_path, extraction_path)
    if not success:
        print("Error during extraction.")
        return
    print("Dataset extracted successfully.")
    
    # Define the path to the training and test directory
    data_dir ='./Data/ProductsDataset'
    test_dir = "./Data/TestDataset"

    # Set datagenerator parameters
    # Set target size for images
    # Set image desired size
    image_size = (400, 400)

    # Set batch_size for tf dataset
    batch_size = 32
    # Set tf datagen parameters
    class_mode='sparse'
    split =0.2

    # Set the seed for random operations.
    seed = 42

    # Create train, val and test data generators from paths
    print("Creating data generators...")
    train_gen, val_gen, test_gen = create_data_generators(data_dir, image_size, batch_size, class_mode, test_dir=test_dir)
    # Get test labels
    test_labels = get_labels_from_generator(test_gen)
    # Get the class names and the corresponding indices.
    class_indices = train_gen.class_indices
    class_names = list(train_gen.class_indices.keys())
    # Get size of train and validation samples
    num_train_samples = train_gen.n
    num_val_samples = val_gen.n
    num_test_samples = test_gen.n

    # Create TensorFlow datasets from the generators
    train_dataset = dataset_from_gen(train_gen,image_size)
    validation_dataset = dataset_from_gen(val_gen,image_size)
    test_dataset=dataset_from_gen(test_gen,image_size)

    # Augment the training dataset
    # Prepare train dataset for training : Augmentation
    train_dataset = augment_dataset(train_dataset)
    # Set model hyperparameters
    dilation_rate = 1
    num_classes = 3
    # Set image shape for model input
    img_shape = image_size + (3,)  # Add channel dimension

    # Build CNN model
    
    model = create_cnn_model(img_shape,num_classes)

    # Compile and train the model
    # Set epochs
    n_epochs = 15
    # Set batch_size
    batch_size = 32
    # Set learning rate
    learning_rate = 0.0005
    # Set steps per epoch
    steps_per_epoch = (num_train_samples // batch_size) *3
    steps_per_epoch=steps_per_epoch
    validation_steps = (num_val_samples // batch_size)*3

    # Compile model using adam ,sparse categorical cross entropy and accuracy as a performance measure.   
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # Train Model
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        batch_size=batch_size,
        epochs = n_epochs )
    
    # View training metrics
    evaluate_training(history)

    # Make predictions on the test set
    print("Making predictions on test set...")
    predicted_labels = make_predictions(model, test_dataset)
    # Apply weight clustering for model
    num_of_clusters=16
    clustering_learning_rate=1e-5

    clustered_model = apply_weight_clustering(model,num_of_clusters,clustering_learning_rate)
    # Fine-tune model
    clustered_model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch*3,
            epochs=1,
            validation_data =validation_dataset,validation_steps=validation_steps*3)
    clustered_labels = make_predictions(clustered_model, test_dataset, num_test_samples)
    # Prepare model for serving by removing training-only variables.
    model_for_serving = tfmot.clustering.keras.strip_clustering(clustered_model)
    # Save the model
    print("Saving models...")
    model.save('MulticlassClassifierModel.h5')
    model_for_serving.save('Model_for_Serving.h5')



if __name__ == "__main__":
    main()
