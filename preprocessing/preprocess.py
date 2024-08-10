import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

def prepare_data():
    df = pd.read_csv("./raw_data/emotions.csv")
    df = df.dropna() 
    df["label"] = df["label"].map({'POSITIVE': 2, 'NEUTRAL': 1, 'NEGATIVE': 0})
    # Identify and remove highly correlated features, as they may carry redundant information.
    correlation_matrix = df.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
    data_len = len(df.columns)
    df = df.drop(columns=to_drop)
    # Train a model and use its feature importance to select the most informative features.
    X = df.drop(columns="label")
    y = df["label"]
    # Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X, y)
    # Get the feature importances
    feature_importance = clf.feature_importances_
    # Adjust the threshold as needed
    selected_features = X.columns[feature_importance > 0.0148]
    df_reduced_model = df[selected_features]
    # Ensure df_reduced_model is a copy of the DataFrame to avoid SettingWithCopyWarning
    df_reduced_model = df_reduced_model.copy()
    # make the variable y as the label feature
    df_reduced_model.loc[:, "label"] = y
    # Dataset to be used for inference part
    df_reduced_model.to_csv('/app/data/preprocessed_data.csv')

if __name__ == "__main__":
    prepare_data()
