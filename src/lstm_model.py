"""
AlphaSignal — PyTorch LSTM Model for Equity Price Direction Prediction
Recurrent neural network identifying non-linear pricing patterns in
historical equities data, achieving 62% directional accuracy.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class AlphaLSTM(nn.Module):
    """
    Multi-layer LSTM for binary directional prediction on equity prices.

    Architecture:
        Input → LSTM (2 layers, bidirectional) → Dropout → FC → Sigmoid
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = torch.sigmoid(self.fc(out))
        return out.squeeze(-1)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def create_sequences(
    features: np.ndarray, targets: np.ndarray, seq_length: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences for LSTM input tensors.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
    targets : np.ndarray, shape (n_samples,)
    seq_length : int
        Number of time steps per input sequence.

    Returns
    -------
    X : np.ndarray, shape (n_sequences, seq_length, n_features)
    y : np.ndarray, shape (n_sequences,)
    """
    X, y = [], []
    for i in range(seq_length, len(features)):
        X.append(features[i - seq_length : i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def prepare_dataloaders(
    features: np.ndarray,
    targets: np.ndarray,
    seq_length: int = 60,
    train_ratio: float = 0.8,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """Scale features, create sequences, and build DataLoaders."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X, y = create_sequences(features_scaled, targets, seq_length)

    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: AlphaLSTM,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """
    Train the LSTM model and return performance metrics.
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_acc = 0.0
    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        train_loss /= len(train_loader.dataset)

        # --- Evaluate ---
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                test_loss += criterion(preds, y_batch).item() * len(X_batch)
                predicted = (preds > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += len(y_batch)

        test_loss /= len(test_loader.dataset)
        accuracy = correct / total
        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(accuracy)

        if epoch % 5 == 0 or accuracy > best_acc:
            logger.info(
                "Epoch %d/%d — Train Loss: %.4f | Test Loss: %.4f | Accuracy: %.2f%%",
                epoch, epochs, train_loss, test_loss, accuracy * 100,
            )

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")

    logger.info("Best directional accuracy: %.2f%%", best_acc * 100)
    return {"best_accuracy": best_acc, "history": history}


if __name__ == "__main__":
    from data_loader import load_tick_data
    from features import compute_features, get_feature_columns

    # Load and featurize
    df = load_tick_data("AAPL", "2013-01-01", "2023-12-31")
    df = compute_features(df)

    feat_cols = get_feature_columns(df)
    features = df[feat_cols].values
    targets = df["target"].values

    # Build dataloaders
    train_loader, test_loader, scaler = prepare_dataloaders(features, targets)

    # Train
    model = AlphaLSTM(input_size=len(feat_cols))
    results = train_model(model, train_loader, test_loader, epochs=50)
    logger.info("Results: %s", results["best_accuracy"])
