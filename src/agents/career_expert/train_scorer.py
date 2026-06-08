import os
import time
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dcn import DCNv2

ASSETS_DIR = "./assets"
MODEL_DIR = "./saved_model"
os.makedirs(MODEL_DIR, exist_ok=True)


class SparseDataset(Dataset):
    """Memory-efficient converter from Scipy Sparse to PyTorch Tensors"""

    def __init__(self, X_sparse, y_dense):
        self.X = X_sparse
        self.y = np.array(y_dense, dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_row = torch.FloatTensor(self.X[idx].toarray()[0])
        y_val = torch.FloatTensor([self.y[idx]])
        return x_row, y_val


def train_scoring_model():
    print("🚀 Start Model Training (Deep & Cross Network V2)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware accelerated on: {device}")

    # 1. Load Data
    with open(os.path.join(ASSETS_DIR, "X_train.pkl"), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, "y_train.pkl"), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, "X_val.pkl"), 'rb') as f:
        X_val = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, "y_val.pkl"), 'rb') as f:
        y_val = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, "X_test.pkl"), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, "y_test.pkl"), 'rb') as f:
        y_test = pickle.load(f)

    # 2. Prepare PyTorch Dataloaders
    input_dim = X_train.shape[1]
    train_loader = DataLoader(SparseDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader = DataLoader(SparseDataset(X_val, y_val), batch_size=512, shuffle=False)
    test_loader = DataLoader(SparseDataset(X_test, y_test), batch_size=512, shuffle=False)

    # 3. Initialize DCN V2 Engine
    model = DCNv2(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 15
    best_val_loss = float('inf')
    model_path = os.path.join(MODEL_DIR, "career_expert_dcn_v2.pth")

    # Arrays to hold tracking statistics for curves
    train_rmse_history = []
    val_rmse_history = []

    print("\nTraining Neural Network...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds.view(-1), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)

        # Validation Pass
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds.view(-1), y_batch.view(-1))
                total_val_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_val_loss = total_val_loss / len(val_loader.dataset)

        epoch_train_rmse = np.sqrt(avg_train_loss)
        epoch_val_rmse = np.sqrt(avg_val_loss)

        # Append to tracking arrays
        train_rmse_history.append(epoch_train_rmse)
        val_rmse_history.append(epoch_val_rmse)

        print(
            f" [Epoch {epoch + 1}/{epochs}] Train RMSE: {epoch_train_rmse:.4f} | Val RMSE: {epoch_val_rmse:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)

    # 4. Save and Generate the Training Loss Curve
    print("\n📈 Plotting and saving model training curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_rmse_history, label='Train RMSE', color='blue', marker='o')
    plt.plot(range(1, epochs + 1), val_rmse_history, label='Val RMSE', color='orange', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('DCNv2 Model Training Convergence (RMSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_output_path = os.path.join(MODEL_DIR, "training_curve.png")
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curve image exported to: {plot_output_path}")

    # 5. Final Blind Evaluation
    print("\n📊 Running Final Evaluation on Blind Test Set...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())

    test_rmse = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
    test_mae = float(mean_absolute_error(all_targets, all_preds))
    test_r2 = float(r2_score(all_targets, all_preds))

    print("============================================================")
    print("📈 SCORING SYSTEM PERFORMANCE EVALUATION")
    print("============================================================")
    print(f"  • Total System Test RMSE : {test_rmse:.4f}")
    print(f"  • Total System Test MAE  : {test_mae:.4f}")
    print(f"  • Total System R² Score  : {test_r2:.4f}")
    print("============================================================")

    metrics_data = {
        "Metric": [
            "Test RMSE",
            "Test MAE",
            "Test R2 Score",
            "Final Train RMSE",
            "Final Val RMSE",
            "Epochs Trained"
        ],
        "Value": [
            round(test_rmse, 4),
            round(test_mae, 4),
            round(test_r2, 4),
            round(float(train_rmse_history[-1]), 4),
            round(float(val_rmse_history[-1]), 4),
            epochs
        ]
    }

    df_metrics = pd.DataFrame(metrics_data)
    eval_output_path = os.path.join(MODEL_DIR, "regression_metrics.csv")
    df_metrics.to_csv(eval_output_path, index=False)

    print(f"💾 Regression evaluation report exported to CSV: {eval_output_path}")
    print("✅ Training and system validation run complete!")


if __name__ == "__main__":
    train_scoring_model()
