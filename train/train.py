import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define directory to save model (EDIT THIS PLEASE)
save_dir = os.getenv("SAVE_DIR", "/fourglasses/app_data")
model_path = os.path.join(save_dir, 'model.pkl')
predictions_path = os.path.join(save_dir, 'validation_predictions.csv')

# Load dataset
df_reduced_model = pd.read_csv('/app_data/preprocessed_data.csv')

# Split the data into 80% training and 20% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    df_reduced_model.drop(columns=["label"]),
    df_reduced_model["label"],
    train_size=0.8,
    random_state=42,
    stratify=df_reduced_model["label"]
)

# Split the temporary set into 50% testing and 50% validation
X_test, X_val, y_test, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

def train_model(X_train, y_train, model_path):
    """Train the SVM model and save it."""
    SVM_model = svm.SVC(kernel='rbf', C=2400)
    SVM_model.fit(X_train, y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(SVM_model, f)

def predict(model_path, X):
    """Load the model and make predictions."""
    with open(model_path, 'rb') as f:
        SVM_model = pickle.load(f)
    return SVM_model.predict(X)

@app.route('/train', methods=['POST'])
def train():
    """Endpoint to train the model."""
    train_model(X_train, y_train, model_path)
    return jsonify({"message": "Model trained and saved!"})

@app.route('/predict', methods=['POST'])
def predict_validation():
    """Endpoint to predict on validation set and save predictions."""
    y_val_pred = predict(model_path, X_val)

    # Reverse mapping to decode the predictions and original validation values
    reverse_mapping = {2: 'POSITIVE', 1: 'NEUTRAL', 0: 'NEGATIVE'}
    y_val_pred_decoded = [reverse_mapping[label] for label in y_val_pred]
    y_val_decoded = [reverse_mapping[label] for label in y_val]

    valid_accuracy = (y_val_pred == y_val).sum() / len(y_val)
    predictions_df = pd.DataFrame({
        'True Label': y_val_decoded,
        'Predicted Label': y_val_pred_decoded
    })

    #Saving validation predictions
    predictions_df.to_csv(predictions_path, index=False)

    return jsonify({
        "message": "Validation predictions made and saved!",
        "validation_accuracy": valid_accuracy
    })

@app.route('/test-predict', methods=['POST'])
def test_predict():
    """Endpoint to predict on test set."""
    y_test_pred = predict(model_path, X_test)

    # Reverse mapping to decode the predictions and original test values
    reverse_mapping = {2: 'POSITIVE', 1: 'NEUTRAL', 0: 'NEGATIVE'}
    y_test_pred_decoded = [reverse_mapping[label] for label in y_test_pred]
    y_test_decoded = [reverse_mapping[label] for label in y_test]

    test_accuracy = (y_test_pred == y_test).sum() / len(y_test)
    return jsonify({
        "message": "Test predictions made!",
        "test_accuracy": test_accuracy,
        "classification_report": classification_report(y_test_decoded, y_test_pred_decoded, output_dict=True)
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
