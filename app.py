import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from preprocess import preprocess_data
import pandas as pd
from google.cloud import storage

app = Flask(__name__)
CORS(app)  # Enable CORS for the web app

# Set the path to your service account JSON key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/HP/Desktop/BACKUP/folajimi/keylogger_project/thesis/implementation/bilstm-397009-d1590212f677.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "bilstm-397009"

# Initialize the storage client
client = storage.Client()

# Replace 'your-bucket-name' with the name of your GCS bucket
bucket_name = 'bi_lstm_model'
model_blob_name = 'model.h5'  # Path within the bucket where the model is located
model_local_path = 'saved_model.h5'  # Local path to save the downloaded model

# Get the bucket
bucket = client.bucket(bucket_name)

# Get the model blob
model_blob = bucket.blob(model_blob_name)

# Download the model from GCS to local path
print("Downloading model from GCS...")
model_blob.download_to_filename(model_local_path)
print("Model downloaded successfully!")

# Load the model
print("Loading model...")
loaded_model = load_model(model_local_path, compile=False)
print("Model loaded successfully!")

#loaded_model = load_model('saved_model.h5', compile=False)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read uploaded CSV file
    validation_data = pd.read_csv(file)
    
    # Preprocess the validation data
    preprocessed_data = preprocess_data(validation_data)
    
    # Make predictions
    predictions = loaded_model.predict(preprocessed_data)
    
    # Convert predicted probabilities to binary predictions using a threshold
    threshold = 0.7
    binary_predictions = (predictions > threshold).astype(int)

    # Map binary predictions to labels
    label_mapping = {0: 'Benign', 1: 'Keylogger'}
    labels = [label_mapping[pred[0]] for pred in binary_predictions]

    # Create a dictionary containing the labels
    result_dict = {
        'labels': labels
    }

    # Return the labels as JSON
    return jsonify(result_dict)

if __name__ == '__main__':
    app.run(debug=True)