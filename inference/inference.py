import os
import pickle
import time
import pandas as pd

# Define all directories and file paths needed
predictions_dir = '/mnt/predictions'
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir, exist_ok=True)
predictions_path = '/mnt/predictions/predictions.csv'
model_path = '/mnt/model/model.pkl'
data_path = '/mnt/validation/validation_data.csv'

def load_model(model_path):
    """Load the saved model from the specified path."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_data(data_path):
    """Load the new data from the specified path."""
    return pd.read_csv(data_path)

def make_predictions(model, X):
    """Use the loaded model to make predictions on the input data."""
    return model.predict(X)

def save_predictions(predictions, output_path):
    """Save the predictions to a CSV file."""
    predictions_df = pd.DataFrame(predictions, columns=['Predicted Label'])
    predictions_df.to_csv(output_path)

def main():
    try:
        # Load the model
        model = load_model(model_path)
        print(f"Model loaded from {model_path}.")

        # Load validation dataset
        df_new_data = load_data(data_path)
        print(f"Data loaded from {data_path}.")

        # Get feature columns
        X_new = df_new_data.drop(columns=["label"], errors='ignore')
        X_new = X_new.loc[:, ~X_new.columns.str.contains('^Unnamed: 0.1')]

        # Make predictions
        predictions = make_predictions(model, X_new)
        print("Predictions made.")

        # Save predictions
        save_predictions(predictions, predictions_path)
        print(f"Predictions saved to {predictions_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    while True:
        time.sleep(100)

# Run the code
main()