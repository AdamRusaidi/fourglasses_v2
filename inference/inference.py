import os
import pickle
import pandas as pd

# Define directories for model and data
save_dir = os.getenv("SAVE_DIR", "/fourglasses/app_data")
model_path = os.path.join(save_dir, 'model.pkl')
new_data_path = os.path.join(save_dir, 'validation_predictions.csv')
predictions_path = os.path.join(save_dir, 'prediction_outputs.csv')

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
    predictions_df.to_csv(output_path, index=False)

def main():
    try:
        # Load the model
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")

        # Load new data
        df_new_data = load_data(new_data_path)
        print(f"Data loaded from {new_data_path}")

        # Ensure the feature columns match the model's training data
        X_new = df_new_data.drop(columns=["label"], errors='ignore')

        # Make predictions
        predictions = make_predictions(model, X_new)
        print("Predictions made")

        # Save predictions
        save_predictions(predictions, predictions_path)
        print(f"Predictions saved to {predictions_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()