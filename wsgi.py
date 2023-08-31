from app import app  # Import your Flask app instance
from waitress import serve

if __name__ == '__main__':
    # Serve the app using Waitress
    serve(app, host='0.0.0.0', port=8080)
