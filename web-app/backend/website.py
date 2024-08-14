import os
import pandas as pd
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Define the absolute path to the CSV file
predictions_path = ('/mnt/predictions/predictions.csv')

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Serve the Predicted Label column from validation_predictions.csv row by row."""
    # Load the CSV data
    df = pd.read_csv(predictions_path)
    
    # Extract the Predicted Label column
    df['Predicted Label'].tolist()
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
