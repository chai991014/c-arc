import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import scipy.sparse
import pandas as pd
import os

# Paths
INPUT_DIR = "./processed_data"
MODEL_DIR = "./saved_model"


def train_model():
    print("Loading feature matrix...")
    X = scipy.sparse.load_npz(f"{INPUT_DIR}/feature_matrix.npz")
    y_raw = pd.read_csv(f"{INPUT_DIR}/labels.csv")['soc_code']

    # Encode SOC codes into integers for XGBoost
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Perform the split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training on {X_train.shape[0]} records, testing on {X_test.shape[0]} records...")

    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='multi:softprob',
        tree_method='hist'
    )

    # Train only on the training set
    model.fit(X_train, y_train)

    # Evaluate on the test set
    print("Evaluating model...")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(f"{MODEL_DIR}/career_expert_v1.json")

    # Save the label encoder so we can map IDs back to SOC codes later
    import joblib
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")

    print(f"✅ Model trained and saved to {MODEL_DIR}")


if __name__ == "__main__":
    train_model()
