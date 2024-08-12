import os
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Define the absolute path to the CSV file
csv_path = os.path.join(os.getenv("SAVE_DIR", "./app_data"), 'prediction_outputs.csv')

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Serve the Predicted Label column from validation_predictions.csv row by row."""
    try:
        # Load the CSV data
        df = pd.read_csv(csv_path)
        
        # Extract the Predicted Label column
        predicted_labels = df['Predicted Label'].tolist()

        return jsonify(predicted_labels)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
