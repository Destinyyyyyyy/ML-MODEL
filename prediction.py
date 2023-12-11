# predict_linkedin.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

def preprocess_data(data):
    if 'real' in data.columns:
        data = data.drop(columns=['real'])

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled


def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def make_predictions(model, input_data):
    predictions = model.predict(input_data)
    return predictions

def postprocess_predictions(predictions):
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

def main():
    input_data = pd.read_csv('testingpred.csv')

    preprocessed_data = preprocess_data(input_data)

    trained_model = load_trained_model('your_model.h5')

    scaler = StandardScaler()
    preprocessed_data_scaled = scaler.fit_transform(preprocessed_data)

    predictions = make_predictions(trained_model, preprocessed_data_scaled)

    predicted_classes = postprocess_predictions(predictions)

    for i, predicted_class in enumerate(predicted_classes):
        print(f"Profile {i+1}: {'Real' if predicted_class == 1 else 'Fake'}")

if __name__ == "__main__":
    main()