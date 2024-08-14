import os
import numpy as np
import pandas as pd
import time
from flask import Flask, jsonify
from xgboost import XGBClassifier

app = Flask(__name__)

# @app.route('/preprocess', methods=['POST'])
def preprocess():
    # Load data from the raw_data folder
    # os.chdir('/app')
    # with open('/app/dataset/emotions.csv', 'r') as file:
    #     df = file.read()
    df = pd.read_csv("/app/dataset/emotions.csv")
    print(df.head())
    print('Read data successfully')

    # Preprocess data
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

    # Train the model
    xgb = XGBClassifier()
    xgb.fit(X, y)

    # Get feature importances
    feature_importance = xgb.feature_importances_

    # Create a DataFrame to hold feature names and their importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    })

    # Sort the features by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Select features with importance above 0.01
    selected_features = importance_df[importance_df['Importance'] > 0.01]['Feature'].tolist()

    # Create a new DataFrame with the selected features
    df_reduced_model = df[selected_features + ['label']]

    # # Ensure the directory exists
    # os.makedirs('/data', exist_ok=True)

    # Save preprocessed data
    # df_reduced_model.to_csv('/mnt/dataset/preprocessed_data.csv')
    # with open('/mnt/dataset/preprocessed_data.csv', 'w') as file:
    #     file.write(df_reduced_model)
    df_reduced_model.to_csv('/mnt/preprocessed/preprocessed_data.csv')

    print('Preprocessing complete, data saved to /mnt/preprocessed/preprocessed_data.csv')

    # test = pd.read_csv("/mnt/preprocessed/preprocessed_data.csv")
    # print(test.head())

    while True:
        time.sleep(100)

if __name__ == "__main__":
    preprocess()
    app.run(host='0.0.0.0', port=5000)
