import os
import numpy as np
import pandas as pd
import time
from xgboost import XGBClassifier
import boto3

def download_from_s3(bucket_name, object_name, file_name):
    s3_client = boto3.client('s3')
    try:
        print(f"Attempting to download {object_name} from S3 bucket {bucket_name}")
        s3_client.download_file(bucket_name, object_name, file_name)
        print(f"Successfully downloaded {object_name} from S3 to {file_name}")
    except Exception as e:
        print(f"Failed to download {object_name} from S3: {e}")

def upload_to_s3(bucket_name, file_name, object_name=None):
    s3_client = boto3.client('s3')
    try:
        if object_name is None:
            object_name = os.path.basename(file_name)
        print(f"Attempting to upload {file_name} to S3 bucket {bucket_name} as {object_name}")
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"Successfully uploaded {file_name} to S3 as {object_name}")
    except Exception as e:
        print(f"Failed to upload {file_name} to S3: {e}")

def preprocess(input_path, output_path):
    try:
        print(f"Reading data from {input_path}")
        df = pd.read_csv(input_path)
        print(df.head())
        print('Read data successfully')

        # Preprocess data
        print("Starting preprocessing...")
        df = df.dropna()
        df["label"] = df["label"].map({'POSITIVE': 2, 'NEUTRAL': 1, 'NEGATIVE': 0})

        # Remove highly correlated features
        correlation_matrix = df.corr().abs()
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
        df = df.drop(columns=to_drop)

        # Train a model and use its feature importance to select the most informative features.
        X = df.drop(columns=['label'])
        y = df['label']

        xgb = XGBClassifier()
        xgb.fit(X, y)

        feature_importance = xgb.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        })

        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        selected_features = importance_df[importance_df['Importance'] > 0.01]['Feature'].tolist()

        df_reduced_model = df[selected_features + ['label']]

        df_reduced_model.to_csv(output_path)
        print(f"Preprocessing complete, data saved to {output_path}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    bucket_name = os.getenv('S3_BUCKET')
    input_object = 'emotions.csv'
    output_object = 'processed_emotions.csv'

    local_input_path = '/tmp/emotions.csv'
    local_output_path = '/tmp/processed_emotions.csv'

    print(f"Starting preprocessing for bucket: {bucket_name}")

    download_from_s3(bucket_name, input_object, local_input_path)
    preprocess(local_input_path, local_output_path)
    upload_to_s3(bucket_name, local_output_path, output_object)
