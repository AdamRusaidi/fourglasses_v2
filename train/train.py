import os
import pickle
import pandas as pd
import time
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split

# saving model path
model_dir = '/mnt/model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

#saving validation path
validation_dir = '/mnt/validation'
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir, exist_ok=True)

# Define directory to save model
model_path = "/mnt/model/model.pkl"

# Load dataset
df_reduced_model = pd.read_csv('/mnt/preprocessed/preprocessed_data.csv')

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

# # Assuming X_val is a DataFrame or a NumPy array
# if not isinstance(X_val, pd.DataFrame):
#     X_val = pd.DataFrame(X_val)

# if not isinstance(y_val, pd.Series):
#     y_val = pd.Series(y_val, name='label')

# Concatenate X_val and y_val along the columns
val_data = pd.concat([X_val, y_val], axis=1)

# Save the combined DataFrame to a CSV file
val_data.to_csv('/mnt/validation/validation_data.csv')

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
    """Endpoint to train the model."""
    train_model(X_train, y_train, model_path)

def test_predict():
    """Endpoint to predict on test set."""
    y_test_pred = predict(model_path, X_test)

    # Reverse mapping to decode the predictions and original test values
    reverse_mapping = {2: 'POSITIVE', 1: 'NEUTRAL', 0: 'NEGATIVE'}
    y_test_pred_decoded = [reverse_mapping[label] for label in y_test_pred]
    y_test_decoded = [reverse_mapping[label] for label in y_test]

    test_accuracy = (y_test_pred == y_test).sum() / len(y_test)

    print("Test predictions made!")
    print("Test accuracy: {}".format(test_accuracy))
    print("Classification report: {}".format(classification_report(y_test_decoded, y_test_pred_decoded, output_dict=True)))

    while True:
        time.sleep(100)

train()
test_predict()
