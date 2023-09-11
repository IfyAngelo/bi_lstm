import requests
import json

url = "http://127.0.0.1:5000/predict"  # Update the URL

# Read the sample CSV file
files = {'file': open('validation_data.csv', 'rb')}

# Send the POST request
response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=2))
else:
    print("Error:", response.text)