import os
import json
import random
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# =========================================================
# REPRODUCIBILITY
# =========================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# CONFIGURATION
# =========================================================
CLIENTS = ["A", "B", "C", "D", "F"]

DATA_DIR = "processed/clients"
OUTPUT_DIR = "outputs/local_ecommerce"

# More training capacity than the old model.
LOCAL_EPOCHS = 120
BATCH_SIZE = 256

LEARNING_RATE = 7e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 2e-5

VALIDATION_SIZE = 0.20
PATIENCE = 20
LR_PATIENCE = 5
GRAD_CLIP = 1.0

HIDDEN_DIM = 256
NUM_BLOCKS = 4
DROPOUT = 0.10

# Small label smoothing improves generalization without changing labels.
LABEL_SMOOTHING = 0.02

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# MODEL
# Improved residual MLP for tabular E-Commerce data
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=DROPOUT):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = out + identity

        return out


class ResNetMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x)

        return self.head(x)


# =========================================================
# LIVE STATUS HELPERS
# =========================================================
def atomic_json_write(path, data):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        prefix=".status_",
        suffix=".json",
        dir=directory,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, path)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def write_live_status(client, status, **kwargs):
    client_dir = os.path.join(
        OUTPUT_DIR,
        f"client_{client}",
    )

    path = os.path.join(
        client_dir,
        "live_status.json",
    )

    payload = {
        "dataset": "E-Commerce",
        "client": client,
        "status": status,

        "current_epoch": kwargs.get("current_epoch", 0),
        "total_epochs": LOCAL_EPOCHS,

        "train_loss": kwargs.get("train_loss"),
        "val_loss": kwargs.get("val_loss"),

        "train_accuracy": kwargs.get("train_accuracy"),
        "val_accuracy": kwargs.get("val_accuracy"),

        "train_precision": kwargs.get("train_precision"),
        "val_precision": kwargs.get("val_precision"),

        "train_recall": kwargs.get("train_recall"),
        "val_recall": kwargs.get("val_recall"),

        "train_f1": kwargs.get("train_f1"),
        "val_f1": kwargs.get("val_f1"),

        "best_val_accuracy": kwargs.get(
            "best_val_accuracy",
            0.0,
        ),

        "best_epoch": kwargs.get(
            "best_epoch",
            0,
        ),

        "learning_rate": kwargs.get(
            "learning_rate",
            LEARNING_RATE,
        ),

        "train_samples": kwargs.get(
            "train_samples",
            0,
        ),

        "val_samples": kwargs.get(
            "val_samples",
            0,
        ),
    }

    atomic_json_write(path, payload)


# =========================================================
# LOAD CLIENT DATA
# =========================================================
def load_client_data(client):
    file_path = os.path.join(
        DATA_DIR,
        f"client_{client}.csv",
    )

    print("\n----------------------------------------")
    print("Loading E-Commerce client:", client)
    print("File:", file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Client file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    target_column = "Reached.on.Time_Y.N"

    if target_column not in df.columns:
        raise ValueError(
            f"'{target_column}' not found in {file_path}"
        )

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------
    y_raw = df[target_column].values

    unique_values = np.sort(
        np.unique(y_raw)
    )

    if len(unique_values) < 2:
        raise ValueError(
            f"{file_path} contains fewer than two classes."
        )

    value_to_class = {
        value: index
        for index, value in enumerate(unique_values)
    }

    y = np.array(
        [
            value_to_class[value]
            for value in y_raw
        ],
        dtype=np.int64,
    )

    # -----------------------------------------------------
    # REMOVE ONLY THE TARGET
    #
    # Do NOT change the DataCo script.
    # For E-Commerce, we keep the existing feature set.
    # -----------------------------------------------------
    X = df.drop(
        columns=[
            target_column
        ],
    )

    # Keep only numeric representation created by preprocessing.
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0.0)

    X = X.values.astype(
        np.float32
    )

    print("Dataset shape:", df.shape)
    print("Number of features:", X.shape[1])
    print("Number of classes:", len(unique_values))
    print("Class distribution:", np.bincount(y))

    return X, y


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================
def split_data(X, y):
    class_counts = np.bincount(y)

    stratify = (
        y
        if len(class_counts) > 1
        and class_counts.min() >= 2
        else None
    )

    return train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=stratify,
    )


# =========================================================
# DATALOADERS
# =========================================================
def create_loaders(
    X_train,
    X_val,
    y_train,
    y_val,
):
    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train,
            dtype=torch.long,
        ),
    )

    val_dataset = TensorDataset(
        torch.tensor(
            X_val,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_val,
            dtype=torch.long,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(
            BATCH_SIZE,
            len(train_dataset),
        ),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=min(
            BATCH_SIZE,
            len(val_dataset),
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


# =========================================================
# METRICS
# =========================================================
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return accuracy, precision, recall, f1


# =========================================================
# TRAIN ONE EPOCH
# =========================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            X_batch
        )

        loss = criterion(
            outputs,
            y_batch,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_targets.extend(
            y_batch.detach()
            .cpu()
            .numpy()
        )

    average_loss = (
        total_loss
        / max(total_samples, 1)
    )

    accuracy, precision, recall, f1 = (
        calculate_metrics(
            all_targets,
            all_predictions,
        )
    )

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1,
    )


# =========================================================
# VALIDATION
# =========================================================
@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(
            X_batch
        )

        loss = criterion(
            outputs,
            y_batch,
        )

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            y_batch.cpu().numpy()
        )

    average_loss = (
        total_loss
        / max(total_samples, 1)
    )

    accuracy, precision, recall, f1 = (
        calculate_metrics(
            all_targets,
            all_predictions,
        )
    )

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1,
        np.array(all_targets),
        np.array(all_predictions),
    )


# =========================================================
# SAVE TRAINING GRAPHS
# =========================================================
def save_graphs(
    history,
    client,
    output_dir,
):
    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    plots = [
        (
            "loss",
            "Loss",
            ["train_loss", "val_loss"],
            ["Training Loss", "Validation Loss"],
        ),
        (
            "accuracy",
            "Accuracy",
            ["train_accuracy", "val_accuracy"],
            ["Training Accuracy", "Validation Accuracy"],
        ),
        (
            "precision",
            "Precision",
            ["train_precision", "val_precision"],
            ["Training Precision", "Validation Precision"],
        ),
        (
            "recall",
            "Recall",
            ["train_recall", "val_recall"],
            ["Training Recall", "Validation Recall"],
        ),
        (
            "f1",
            "F1 Score",
            ["train_f1", "val_f1"],
            ["Training F1", "Validation F1"],
        ),
        (
            "learning_rate",
            "Learning Rate",
            ["learning_rate"],
            ["Learning Rate"],
        ),
    ]

    for filename, ylabel, keys, labels in plots:
        plt.figure(
            figsize=(8, 5)
        )

        for key, label in zip(
            keys,
            labels,
        ):
            plt.plot(
                epochs,
                history[key],
                label=label,
            )

        plt.xlabel("Local Epoch")
        plt.ylabel(ylabel)

        plt.title(
            f"E-Commerce - Client {client} - {ylabel}"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"{filename}_curve.png",
            ),
            dpi=200,
        )

        plt.close()


# =========================================================
# CONFUSION MATRIX
# =========================================================
def save_confusion_matrix(
    y_true,
    y_pred,
    client,
    output_dir,
    num_classes,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(num_classes)
        ),
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(cm)

    plt.title(
        f"E-Commerce - Client {client} - Confusion Matrix"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.colorbar()

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "confusion_matrix.png",
        ),
        dpi=200,
    )

    plt.close()


# =========================================================
# TRAIN ONE CLIENT
# =========================================================
def train_client(client):
    client_output = os.path.join(
        OUTPUT_DIR,
        f"client_{client}",
    )

    os.makedirs(
        client_output,
        exist_ok=True,
    )

    write_live_status(
        client,
        "starting",
    )

    X, y = load_client_data(
        client
    )

    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = split_data(
        X,
        y,
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # Fit the scaler ONLY on training data.
    # This avoids validation-data leakage.
    # -----------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_val = scaler.transform(
        X_val
    ).astype(np.float32)

    # Save scaler parameters so the exact preprocessing
    # can be reproduced later.
    np.savez(
        os.path.join(
            client_output,
            "feature_scaler.npz",
        ),
        mean=scaler.mean_,
        scale=scaler.scale_,
    )

    (
        train_loader,
        val_loader,
    ) = create_loaders(
        X_train,
        X_val,
        y_train,
        y_val,
    )

    num_classes = len(
        np.unique(y)
    )

    input_dim = X_train.shape[1]

    model = ResNetMLP(
        input_dim=input_dim,
        num_classes=num_classes,
    ).to(device)

    # -----------------------------------------------------
    # CLASS-WEIGHTED LOSS
    # Helps the binary E-Commerce target when classes are
    # imbalanced.
    # -----------------------------------------------------
    class_counts = np.bincount(
        y_train,
        minlength=num_classes,
    ).astype(np.float32)

    class_weights = (
        len(y_train)
        / (
            num_classes
            * np.maximum(class_counts, 1.0)
        )
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Smooth cosine decay usually works better for this
    # type of tabular neural network than a fixed LR.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=LOCAL_EPOCHS,
        eta_min=MIN_LR,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1": [],
        "val_f1": [],
        "learning_rate": [],
    }

    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None

    write_live_status(
        client,
        "training",
        current_epoch=0,
        train_samples=len(X_train),
        val_samples=len(X_val),
    )

    # =====================================================
    # TRAINING LOOP
    # =====================================================
    for epoch in range(
        1,
        LOCAL_EPOCHS + 1,
    ):
        (
            train_loss,
            train_accuracy,
            train_precision,
            train_recall,
            train_f1,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            y_true,
            y_pred,
        ) = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(
            float(train_loss)
        )

        history["val_loss"].append(
            float(val_loss)
        )

        history["train_accuracy"].append(
            float(train_accuracy)
        )

        history["val_accuracy"].append(
            float(val_accuracy)
        )

        history["train_precision"].append(
            float(train_precision)
        )

        history["val_precision"].append(
            float(val_precision)
        )

        history["train_recall"].append(
            float(train_recall)
        )

        history["val_recall"].append(
            float(val_recall)
        )

        history["train_f1"].append(
            float(train_f1)
        )

        history["val_f1"].append(
            float(val_f1)
        )

        history["learning_rate"].append(
            float(current_lr)
        )

        # -------------------------------------------------
        # BEST MODEL
        # Prefer higher validation accuracy.
        # If accuracy ties, prefer lower validation loss.
        # -------------------------------------------------
        improved = (
            val_accuracy
            > best_val_accuracy + 1e-5
        )

        tie_but_lower_loss = (
            abs(
                val_accuracy
                - best_val_accuracy
            ) <= 1e-5
            and val_loss < best_val_loss
        )

        if improved or tie_but_lower_loss:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            torch.save(
                {
                    "client": client,
                    "model_state_dict": best_state,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "hidden_dim": HIDDEN_DIM,
                    "num_blocks": NUM_BLOCKS,
                    "dropout": DROPOUT,
                    "best_val_accuracy": best_val_accuracy,
                    "best_epoch": best_epoch,
                },
                os.path.join(
                    client_output,
                    "best_local_model.pth",
                ),
            )

        else:
            epochs_without_improvement += 1

        # -------------------------------------------------
        # WRITE CSV EVERY EPOCH
        # -------------------------------------------------
        history_df = pd.DataFrame(
            history
        )

        history_df.insert(
            0,
            "epoch",
            range(
                1,
                len(history_df) + 1,
            ),
        )

        history_df.to_csv(
            os.path.join(
                client_output,
                "training_metrics.csv",
            ),
            index=False,
        )

        # -------------------------------------------------
        # LIVE DASHBOARD STATUS
        # -------------------------------------------------
        write_live_status(
            client,
            "training",
            current_epoch=epoch,

            train_loss=train_loss,
            val_loss=val_loss,

            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,

            train_precision=train_precision,
            val_precision=val_precision,

            train_recall=train_recall,
            val_recall=val_recall,

            train_f1=train_f1,
            val_f1=val_f1,

            best_val_accuracy=best_val_accuracy,
            best_epoch=best_epoch,

            learning_rate=current_lr,

            train_samples=len(X_train),
            val_samples=len(X_val),
        )

        print(
            f"[E-Commerce/{client}] "
            f"Epoch {epoch:03d}/{LOCAL_EPOCHS} | "
            f"Train Acc={train_accuracy:.4f} | "
            f"Val Acc={val_accuracy:.4f} | "
            f"Best={best_val_accuracy:.4f} | "
            f"LR={current_lr:.2e}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}"
            )
            break

    # =====================================================
    # RESTORE BEST MODEL
    # =====================================================
    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    (
        best_loss,
        best_acc,
        best_precision,
        best_recall,
        best_f1,
        y_true,
        y_pred,
    ) = validate(
        model,
        val_loader,
        criterion,
    )

    # =====================================================
    # SAVE FINAL OUTPUTS
    # =====================================================
    save_graphs(
        history,
        client,
        client_output,
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        client,
        client_output,
        num_classes,
    )

    with open(
        os.path.join(
            client_output,
            "classification_report.txt",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            classification_report(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

    # Save a compact final result for the dashboard.
    final_result = {
        "dataset": "E-Commerce",
        "client": client,
        "final_accuracy": float(best_acc),
        "final_precision": float(best_precision),
        "final_recall": float(best_recall),
        "final_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "total_epochs_run": len(history["train_loss"]),
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
    }

    atomic_json_write(
        os.path.join(
            client_output,
            "final_result.json",
        ),
        final_result,
    )

    write_live_status(
        client,
        "completed",

        current_epoch=len(
            history["train_loss"]
        ),

        train_loss=history["train_loss"][-1],
        val_loss=best_loss,

        train_accuracy=history["train_accuracy"][-1],
        val_accuracy=best_acc,

        train_precision=history["train_precision"][-1],
        val_precision=best_precision,

        train_recall=history["train_recall"][-1],
        val_recall=best_recall,

        train_f1=history["train_f1"][-1],
        val_f1=best_f1,

        best_val_accuracy=best_acc,
        best_epoch=best_epoch,

        learning_rate=history["learning_rate"][-1],

        train_samples=len(X_train),
        val_samples=len(X_val),
    )

    print("\nFinal E-Commerce result")
    print("----------------------------------------")
    print("Client:", client.upper())
    print(
        f"Best Validation Accuracy: {best_acc:.4%}"
    )
    print(
        f"Precision: {best_precision:.4%}"
    )
    print(
        f"Recall: {best_recall:.4%}"
    )
    print(
        f"F1 Score: {best_f1:.4%}"
    )
    print(
        f"Best Epoch: {best_epoch}"
    )


# =========================================================
# TRAIN ALL CLIENTS
# =========================================================
if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("IMPROVED E-COMMERCE LOCAL MODEL TRAINING")
    print("=" * 70)

    results = []

    for client in CLIENTS:
        train_client(client)

        result_path = os.path.join(
            OUTPUT_DIR,
            f"client_{client}",
            "final_result.json",
        )

        if os.path.exists(result_path):
            with open(
                result_path,
                "r",
                encoding="utf-8",
            ) as f:
                results.append(
                    json.load(f)
                )

    if results:
        average_accuracy = float(
            np.mean(
                [
                    r["final_accuracy"]
                    for r in results
                ]
            )
        )

        best_accuracy = float(
            np.max(
                [
                    r["final_accuracy"]
                    for r in results
                ]
            )
        )

        summary = {
            "dataset": "E-Commerce",
            "clients_completed": len(results),
            "average_final_accuracy": average_accuracy,
            "best_client_accuracy": best_accuracy,
            "clients": results,
        }

        atomic_json_write(
            os.path.join(
                OUTPUT_DIR,
                "final_summary.json",
            ),
            summary,
        )

    print("\n")
    print("=" * 70)
    print("ALL E-COMMERCE LOCAL MODELS COMPLETED")
    print("=" * 70)
    print(
        "Outputs saved in: outputs/local_ecommerce/"
    )
