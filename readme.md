
<h3 align="center" style="line-height: 1.3;">
  Quality Inspection Vision System with TensorFlow-Keras for Manufactured Products
</h3>

## Project Overview
  
  The project presents a comprehensive pipeline for developing a quality inspection vision system using Tensorflow-Keras for manufactured products. The system is designed to scale, allowing for the inspection of product quality on a conveyor belt in a production line. Tailored for integration into production processes, the system utilizes computer vision to categorize products as Broken, Flawed, or Good.
  
The Github Repo for the Django Project can be accessed at <a >https://github.com/Hypatchia/QualityInspection-Images-Django</a>


## Features
* **Scalability for Business Growth:** The system adapts to the evolving needs of growing businesses and expanding production lines, this Scalability translates into increased productivity, operational flexibility, and strategic advantage in responding to dynamic market demands.
* **TensorFlow-Keras Integration:** Advanced image analysis, ensuring accurate and adaptive product categorization enforces Technological leadership which enhances competitiveness, and this adds value to business operations.
 * **Real-time Inspection for Operational Efficiency:** The system ensures instantaneous quality assessment, allowing for timely identification and resolution of production issues. This Real-time inspection contributes to operational efficiency, reducing downtime, minimizing defects, and optimizing the overall production process.
 * **Product Categorization:** Categorized data insights enable informed decision-making, helping businesses address quality issues, optimize production, and improve overall financial performance.
## Built with:

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Latest-blue?style=flat&logo=tensorflow)](https://www.tensorflow.org/)


## Methodology:

- Load 3 classes dataset into train, validation & test
- Preprocess Images
- Augment Dataset
- Build, Train, and optimize a Multiclss CNN.
- Evaluate Quality of Training
- Optimize the model for size and speed using weight clustering.
- Save and integrate the resulting model into a Django WebApp.

## Dataset

The CNN model was trained on a dataset of product images belonging to 3 classes: **Broken {0} - Flawed {1} - Good {2}**


<h3 align="center">The Images in the Dataset 
</h3>
<p align="center">
  <img src="imgs/data.jpg" alt="Dataset Train & Validation" style="width:50%; height:auto;">
</p>

## Image Processing:
To prepare the images for training and prediction, the following processing steps were performed:

* Images were loaded into their respective training and validation directories using the TensorFlow data generator.
* Resizing: Images were resized to a consistent size to ensure uniform input for the neural network.
* Dataset Augmentation: Fundamental Augmentation was applied.

## Training a Convolutional Neural Network for Multiclass Image Classification

A Convolutional Neural Network (CNN) was designed to perform the multiclass classification. The CNN is trained on the labeled dataset, learning to distinguish between good, broken, and flawed products.

The CNN architecture includes convolutional layers, pooling layers, and fully connected layers.

<h3 align="center">Multiclass CNN Model Architecture
</h3>
<p align="center">
  <img src="imgs/cnn.jpg" alt="Dataset Train & Validation" style="width:50%; height:auto;">
</p>



## Model Evaluation:

To evaluate the performance of the trained model and ensure its accuracy and reliability in predicting product defects, the metrics used were the classification report: accuracy, recall, f1 score, and precision.


<p style="margin-top:2rem"> <li> The Training, Validation Loss, and Training Metrics are shown in the figure:</li></p>



<p align="center">
  <img src="imgs/training.png" alt="Dataset Train & Validation" style="width:50%; height:auto;">
</p>

<p style="margin-top:2rem"> <li> The Evaluation Metrics on Newly Unseen Data gave the Classification Report: </li></p>


<p align="center">
  <img src="imgs/report.png" alt="Dataset Train & Validation" style="width:50%; height:auto;">
</p>

<p style="margin-top:2rem"> <li> The Evaluation Metrics on Newly Unseen Data gave the Confusion Matrix </li></p>

<p align="center">
  <img src="imgs/cm.png" alt="Dataset Train & Validation" style="width:20%; height:auto;">
</p>

## Model Compression: Weight Clustering:

The weight Clustered & FineTuned Model resulted in the following metrics:

<p style="margin-top:2rem"> <li> The Evaluation Metrics on Newly Unseen Data gave the Classification Report: </li></p>


<h3 align="center">The Images in the Dataset 
</h3>
<p align="center">
  <img src="imgs/report_2.png" alt="Dataset Train & Validation" style="width:50%; height:auto;">
</p>


<p style="margin-top:2rem"> <li> The Evaluation Metrics on Newly Unseen Data gave the Confusion Matrix </li></p>
<h3 align="center">The Images in the Dataset 
</h3>
<p align="center">
  <img src="imgs/cm_2.jpg" alt="Dataset Train & Validation" style="width:20%; height:auto;">
</p>




## Web App

A defect assessment web system was built using Django that allows the upload of an image of a product, processing, and then prediction of its status with respective probabilities.

The process includes loading the pre-trained deep learning model from Azure Blob storage, preprocessing the image, and then making the final prediction of the product's status as "broken", "flawed", "good"

A screenshot of the app is available:

The app can be accessed at [https://productinspection.azurewebsites.net/](https://productinspection.azurewebsites.net/).


## Deployment

The final web application has been deployed on Azure App Services, ensuring scalability and reliability. Azure - Blob Containers are used for storing deep learning models and product images for seamless integration with the web application.

## Setup to run

To run the project, follow these steps:

* Clone the Repository
* Navigate to the project directory
* Create a Virtual Environment and Activate it
* Install requirements
* You can then either:
    * Navigate to /Pipeline/ & run 
    ~~~
    python main.py
    ~~~
    * navigate to /Notebooks/ & Run The Jupyter Notebook.

## Contact
 Feel free to reach out to me on LinkedIn or through email & don't forget to visit my portfolio.
 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/samiabelhaddad/)
[![Email](https://img.shields.io/badge/Email-Contact%20Me-brightgreen?style=flgat&logo=gmail)](mailto:samiamagbelhaddad@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit%20My%20Portfolio-white?style=flat&logo=website)](https://sambelh.azurewebsites.net/)




