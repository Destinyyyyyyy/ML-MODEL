# Project Title

## Overview
This project aims to build a model to predict the authenticity of LinkedIn profiles based on various features such as the number of connections, projects, skills, etc.

## Installation
To install the required dependencies, run the following command:

- pip install -r requirements.txt

## Usage
To use the project, follow these steps:
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run the `predict_linkedin.py` script with the command `python predict_linkedin.py`.
4. Follow the on-screen instructions to make predictions for LinkedIn profiles.

## Data Analysis
The data analysis process involved exploring the dataset, visualizing key features, and identifying correlations between variables. This helped gain insights into the dataset and understand its characteristics.

## Model Building
The model was built using TensorFlow and Keras. It consists of several dense layers with ReLU activation functions and dropout regularization to prevent overfitting. The model was trained using the Adam optimizer and categorical cross-entropy loss function.

## Prediction
To make predictions using the trained model, follow these steps:
1. Prepare a CSV file containing the LinkedIn profile data for which you want to make predictions.
2. Ensure that the CSV file has the same features as the training data but without the "real" column.
3. Run the `predict_linkedin.py` script and provide the path to the CSV file when prompted.
4. The script will output the predicted authenticity for each LinkedIn profile.

## Repository Structure
- `predict_linkedin.py`: Script for making predictions using the trained model.
- `your_model.h5`: Trained model saved in HDF5 format.
- `requirements.txt`: List of required Python packages and their versions.

## Additional Files
- `linkedin_train.csv`: CSV file containing the training dataset.
- `linkedin_test.csv`: CSV file containing the test dataset.
- `prediction.py`: Script containing functions for data preprocessing, model loading, and making predictions.

## References
- (https://www.semanticscholar.org/paper/Identifying-Fake-Profiles-in-LinkedIn-Adikari-Dutta/2bc753e410dd8c029576a06389c9208dc09907c5)



