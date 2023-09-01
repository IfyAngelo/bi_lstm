import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from preprocess import preprocess_data
import pandas as pd
from google.cloud import storage
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for the web app

# Initialize the storage client
client = storage.Client()

# Get the bucket name and model blob name from environment variables
bucket_name = os.environ.get('BUCKET_NAME')
model_blob_name = os.environ.get('MODEL_BLOB_NAME')

# Get the bucket
bucket = client.bucket(bucket_name)

def load_cached_model():
    global loaded_model
    if loaded_model is None:
        # Get the model blob
        model_blob = bucket.blob(model_blob_name)

        # Create a temporary file to load the model
        with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
            # Download the model from GCS to temporary file
            print("Downloading model from GCS...")
            model_blob.download_to_file(temp_model_file)
            print("Model downloaded successfully!")

            # Load the model directly from the temporary file
            temp_model_file.seek(0)
            loaded_model = load_model(temp_model_file.name, compile=False)
            print("Model loaded successfully!")

# Load the cached model when the app starts
loaded_model = None
load_cached_model()

@app.route('/predict', methods=['POST'])
def predict():
    # Load the cached model
    load_cached_model()
    
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
