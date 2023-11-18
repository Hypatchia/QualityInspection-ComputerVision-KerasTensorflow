import os 
import zipfile


# Extract Dataset from zip file from Directory
def extract_zip_dataset(dataset_file_path, extraction_path):
    """
    Extracts a zipped dataset to the specified extraction path.

    Parameters:
    - dataset_file_path (str): Path to the zipped dataset file.
    - extraction_path (str): Path where the dataset will be extracted.

    Returns:
    - success (bool): True if extraction was successful, False otherwise.
    """
    try:
        os.makedirs(extraction_path, exist_ok=True)
        
        with zipfile.ZipFile(dataset_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        
        return True  # Extraction successful
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False  # Extraction failed
    

from google.colab import drive
# Load dataset from Drive (Google Colab)
def load_from_drive(dataset_file_path,extraction_path):
  drive.mount('/content/drive')
  # Create the extraction directory if it doesn't exist
  os.makedirs(extraction_path, exist_ok=True)
  # Extract the zipped dataset
  with zipfile.ZipFile(dataset_file_path, 'r') as zip_ref:
      zip_ref.extractall(extraction_path)

# Load dataset from local machine
