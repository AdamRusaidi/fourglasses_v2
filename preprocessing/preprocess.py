import os
import numpy as np
import pandas as pd
import time
from xgboost import XGBClassifier


def preprocess():
    output_dir = '/mnt/preprocessed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

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
    df_reduced_model.to_csv(f'{output_dir}/preprocessed_data.csv')

    print('Preprocessing complete, data saved to /mnt/preprocessed/preprocessed_data.csv')

    while True:
        time.sleep(100)

preprocess()