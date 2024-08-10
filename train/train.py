import pickle
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm

df = pd.read_csv('/app/data/preprocessed_data.csv')

# Split the data into 80% training and 20% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    df.drop(columns=["label"]),
    df["label"],
    train_size=0.8,
    random_state=42,
    stratify=df["label"]
)

# Split the temporary set into 50% testing and 50% validation
X_test, X_val, y_test, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

def train():
    # define the non-linear SVM model
    SVM_model = svm.SVC(kernel='rbf', C=2400)
    # Train the SVM model
    SVM_model.fit(X_train, y_train)
    # Predict on the test set
    y_test_pred = SVM_model.predict(X_test)
    # Create a reverse map dictionary for decoding
    reverse_mapping = {2: 'POSITIVE', 1: 'NEUTRAL', 0: 'NEGATIVE'}
    # Decode the predicted test labels
    y_test_pred_decoded = [reverse_mapping[label] for label in y_test_pred]
    # Decode the correct test  labels
    y_test_decoded = [reverse_mapping[label] for label in y_test]
    test_accuracy = (y_test_pred == y_test).sum() / len(y_test)
    print("The model has an accuracy of {:.4}% on the TEST data.".format(test_accuracy * 100))
    # Save RandomForestClassifer model into save directory
    with open('/app/data/model.pkl', 'wb') as f:
        pickle.dump(SVM_model, f)

def validation():
    # Load the model from the file
    with open('/app/data/model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
        
    y_val_pred = loaded_model.predict(X_val)

    valid_accuracy = (y_val_pred == y_val).sum() / len(y_val)
    print("The model has an accuracy of {:.4}% on the VALIDATION data.".format(valid_accuracy * 100))

if __name__ == "__main__":
    train()
    validation()
