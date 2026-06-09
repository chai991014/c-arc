import xgboost as xgb
import pickle
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration ---
ASSETS_DIR = "./assets"


def train_scoring_model():
    print("🚀 Start Model Training (XGBoost Regressor)...")
    start_time = time.time()

    # 1. Load Data Splits
    print("Loading Train, Validation, and Test sets from disk...")
    with open(os.path.join(ASSETS_DIR, "X_train.pkl"), 'rb') as f: X_train = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, "y_train.pkl"), 'rb') as f: y_train = pickle.load(f)

    with open(os.path.join(ASSETS_DIR, "X_val.pkl"), 'rb') as f: X_val = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, "y_val.pkl"), 'rb') as f: y_val = pickle.load(f)

    with open(os.path.join(ASSETS_DIR, "X_test.pkl"), 'rb') as f: X_test = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, "y_test.pkl"), 'rb') as f: y_test = pickle.load(f)

    print(f"Data loaded in {time.time() - start_time:.2f} seconds.")

    # 2. Initialize XGBoost Regressor
    # - 'hist': Highly optimized for sparse matrices and legacy CPUs
    # - 'early_stopping_rounds': Stops training if validation RMSE doesn't improve for 50 trees
    model = xgb.XGBRegressor(
        n_estimators=1000,  # Max number of trees
        learning_rate=0.05,  # Step size shrinkage
        max_depth=6,  # Maximum depth of a tree
        objective='reg:squarederror',  # Optimization target for continuous variables
        tree_method='hist',  # Fast histogram optimization
        early_stopping_rounds=50,
        eval_metric='rmse',  # Track Root Mean Squared Error
        device='cuda',
        random_state=42,
        n_jobs=-1
    )

    # 3. Train the Model
    print("\nTraining XGBoost model (Monitoring Validation Set for Early Stopping)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50  # Print updates every 50 trees
    )

    # 4. Final Blind Evaluation on Test Set
    print("\n📊 Running Final Evaluation on Blind Test Set...")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test R² Score : {r2:.4f} (Closer to 1.0 is better)")
    print(f"Test MAE      : {mae:.4f} (Average error margin in capability score)")
    print(f"Test MSE      : {mse:.4f}")

    # 5. Save the Final Model
    model_path = os.path.join(ASSETS_DIR, "career_expert_xgboost_regressor.json")
    print(f"\n💾 Saving trained model to {model_path}...")
    model.save_model(model_path)

    # 6. Plot Learning Curves
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE', color='blue', linewidth=2)
    plt.plot(x_axis, results['validation_1']['rmse'], label='Validation RMSE', color='orange', linewidth=2)
    plt.legend()
    plt.title('XGBoost Regressor Learning Curve (RMSE)')
    plt.xlabel('Boosting Round (Trees)')
    plt.ylabel('Root Mean Squared Error')
    plt.grid(True, linestyle='--', alpha=0.5)

    plot_path = os.path.join(ASSETS_DIR, "training_learning_curve.png")
    plt.savefig(plot_path)
    print(f"📈 Learning curve plot saved to {plot_path}")
    print("✅ Training Complete!")


if __name__ == "__main__":
    train_scoring_model()
