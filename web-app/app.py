import os
import pandas as pd
from flask import Flask, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define the absolute path to the CSV file
predictions_path = '/mnt/predictions/predictions.csv'

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Serve the Predicted Label column from predictions.csv row by row."""
    try:
        # Load the CSV data
        df = pd.read_csv(predictions_path)
        
        # Extract the Predicted Label column
        predictions = df['Predicted Label'].tolist()
        
        # Return the predictions as JSON
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
