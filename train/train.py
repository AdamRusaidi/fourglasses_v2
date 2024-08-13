import os
import pickle
import pandas as pd
import boto3
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split

# AWS S3 Configuration
s3_bucket = os.getenv("S3_BUCKET")
s3_client = boto3.client('s3')

def download_from_s3(bucket_name, object_name, file_name):
    try:
        print(f"Downloading {object_name} from S3 bucket {bucket_name}")
        s3_client.download_file(bucket_name, object_name, file_name)
        print(f"Successfully downloaded {object_name} to {file_name}")
    except Exception as e:
        print(f"Failed to download {object_name} from S3: {e}")

def upload_to_s3(bucket_name, file_name, object_name=None):
    try:
        if object_name is None:
            object_name = os.path.basename(file_name)
        print(f"Uploading {file_name} to S3 bucket {bucket_name} as {object_name}")
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"Successfully uploaded {file_name} as {object_name}")
    except Exception as e:
        print(f"Failed to upload {file_name} to S3: {e}")

# Define paths
model_path = "/tmp/model.pkl"
local_preprocessed_data = "/tmp/processed_emotions.csv"
local_validation_data = "/tmp/validation_data.csv"

# Download preprocessed data from S3
download_from_s3(s3_bucket, "processed_emotions.csv", local_preprocessed_data)

# Ensure the file exists before proceeding
if not os.path.exists(local_preprocessed_data):
    raise FileNotFoundError(f"{local_preprocessed_data} not found!")

# Load dataset
df_reduced_model = pd.read_csv(local_preprocessed_data)

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

# Save validation data to CSV
val_data = pd.concat([X_val, y_val], axis=1)
val_data.to_csv(local_validation_data, index=False)

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

def train():
    """Train the model and upload the model to S3."""
    train_model(X_train, y_train, model_path)
    upload_to_s3(s3_bucket, model_path, "model.pkl")
    upload_to_s3(s3_bucket, local_validation_data, "validation_data.csv")

def test_predict():
    """Predict on test set."""
    y_test_pred = predict(model_path, X_test)

    reverse_mapping = {2: 'POSITIVE', 1: 'NEUTRAL', 0: 'NEGATIVE'}
    y_test_pred_decoded = [reverse_mapping[label] for label in y_test_pred]
    y_test_decoded = [reverse_mapping[label] for label in y_test]

    test_accuracy = (y_test_pred == y_test).sum() / len(y_test)
    print(f"Test accuracy: {test_accuracy}")
    print(f"Classification report: {classification_report(y_test_decoded, y_test_pred_decoded)}")

    while True:
        time.sleep(100)

if __name__ == "__main__":
    train()
    test_predict()
