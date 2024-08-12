from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the model and scaler
model = load_model('/users/nik/lstm_weather_model.keras')
scaler = MinMaxScaler()
# Assuming you have a method to fit or load the scaler used during training
# scaler = ...

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        xs.append(x)
    return np.array(xs)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    
    # Feature extraction and scaling
    features = df[['Wind Speed (m/s)', 'Wind Direction (°)']]
    scaled_features = scaler.transform(features)
    
    # Create sequences
    SEQ_LENGTH = 24  # Adjust this if needed
    X = create_sequences(scaled_features, SEQ_LENGTH)
    
    # Make predictions
    predictions = model.predict(X)
    predictions_inv = scaler.inverse_transform(predictions)
    
    return jsonify(predictions_inv.tolist())

if __name__ == '__main__':
    app.run(debug=True)
