import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    # Load data from the raw_data folder
    df = pd.read_csv("./raw_data/emotions.csv")

    # Preprocess data
    df = df.dropna()
    df["label"] = df["label"].map({'POSITIVE': 2, 'NEUTRAL': 1, 'NEGATIVE': 0})

    # Remove highly correlated features
    correlation_matrix = df.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
    df = df.drop(columns=to_drop)

    # Feature selection with RandomForest
    X = df.drop(columns="label")
    y = df["label"]
    clf = RandomForestClassifier()
    clf.fit(X, y)
    feature_importance = clf.feature_importances_
    selected_features = X.columns[feature_importance > 0.0148]

    # Reduced dataset
    df_reduced_model = df[selected_features].copy()
    df_reduced_model["label"] = y

    # Ensure the directory exists
    os.makedirs('../app_data', exist_ok=True)

    # Save preprocessed data
    df_reduced_model.to_csv('../app_data/preprocessed_data.csv')

    return jsonify({"message": "Preprocessing complete, data saved to ../app_data/preprocessed_data.csv"})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
