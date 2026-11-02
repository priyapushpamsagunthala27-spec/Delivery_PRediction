
"""
FedPRADA - Module 2
Server Initialization and Client Update Module

Implements ONLY:
1. Server initializes one global model.
2. Server broadcasts the global model to all participating clients.
3. Each client initializes from the received global model.
4. Each client performs local FedProx-style updating:
       L_prox = L_local + (mu/2) * ||theta_m - theta_g||^2
5. Clients return local model parameters, training sample count,
   local validation loss, and training-split label counts.
6. Server constructs the global label distribution.

IMPORTANT:
- This module DOES NOT aggregate local models.
- FedPRADA adaptive/KL-based weighting and model aggregation belong
  to the next server aggregation module.
- Run separately for DataCo and E-Commerce Shipping because their
  feature/label spaces are different. They cannot share one neural
  network instance.

Paper settings used here:
- 5 clients for DataCo: africa, europe, latam, pacific_asia, usca
- 5 clients for E-Commerce Shipping: A, B, C, D, F
- local epochs = 3
- Adam optimizer
- full-batch local update for FedProx
- DataCo learning rate = 1e-3
- E-Commerce learning rate = 5e-4
- proximal coefficient is configurable; 0.1 is the default from the
  paper's FedProx candidate grid {0.001, 0.01, 0.1, 0.5, 1}
"""

import os
import json
import copy
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from secure_transport import (
    FederatedRoundCoordinator,
    TLSFederatedClient,
    TLSFederatedServer,
    add_tls_arguments,
    tls_config_from_args,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# FEDPRADA MODULE-2 SETTINGS
# ============================================================

GLOBAL_ROUNDS = 1
LOCAL_EPOCHS = 3

VALIDATION_SIZE = 0.20

# Paper reports Adam for the FL experiments.
WEIGHT_DECAY = 0.0

# FedProx/FedPRADA proximal term.
# Paper searches mu in {0.001, 0.01, 0.1, 0.5, 1}.
MU_PROX = 0.1

# DataCo and E-Commerce learning rates from the paper.
LEARNING_RATE = {
    "dataco": 1e-3,
    "ecommerce": 5e-4,
}


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_CONFIG = {
    "dataco": {
        "name": "DataCo",
        "clients": [
            "africa",
            "europe",
            "latam",
            "pacific_asia",
            "usca",
        ],
        "data_dir": "processed/clients/dataco",
        "output_dir": "outputs/fedprada_module2_dataco",
        "target": "Delivery Status",

        # Same feature-removal logic used in local_dataco.py
        "drop_columns": [
            "Delivery Status",
            "Days for shipping (real)",
            "Late_delivery_risk",
            "Market",
        ],

        # Same DataCo model family used in local_dataco.py
        "hidden_dim": 128,
        "num_blocks": 3,
        "dropout": 0.20,
        "model_type": "dataco",
    },

    "ecommerce": {
        "name": "E-Commerce Shipping",
        "clients": [
            "A",
            "B",
            "C",
            "D",
            "F",
        ],
        "data_dir": "processed/clients",
        "output_dir": "outputs/fedprada_module2_ecommerce",
        "target": "Reached.on.Time_Y.N",

        # Same feature logic used in local_ecommerce.py:
        # remove only target.
        "drop_columns": [
            "Reached.on.Time_Y.N",
        ],

        # Same E-Commerce model family used in local_ecommerce.py
        "hidden_dim": 256,
        "num_blocks": 4,
        "dropout": 0.10,
        "model_type": "ecommerce",
    },
}


# ============================================================
# MODEL - DATACO
# ============================================================

class DataCoResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout):

        super().__init__()

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.layer_norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(self, x):

        identity = x

        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x + identity

        return self.layer_norm(x)


class DataCoResNetMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=128,
        num_blocks=3,
        dropout=0.20,
    ):

        super().__init__()

        self.input_layer = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.input_relu = nn.ReLU()

        self.blocks = nn.ModuleList(
            [
                DataCoResidualBlock(
                    hidden_dim,
                    dropout
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):

        x = self.input_layer(x)
        x = self.input_relu(x)

        for block in self.blocks:
            x = block(x)

        return self.output_layer(x)


# ============================================================
# MODEL - E-COMMERCE
# ============================================================

class EcommerceResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout):

        super().__init__()

        self.fc1 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.bn1 = nn.BatchNorm1d(
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.bn2 = nn.BatchNorm1d(
            hidden_dim
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

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


class EcommerceResNetMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=256,
        num_blocks=4,
        dropout=0.10,
    ):

        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.BatchNorm1d(
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [
                EcommerceResidualBlock(
                    hidden_dim,
                    dropout
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),
            nn.BatchNorm1d(
                hidden_dim // 2
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim // 2,
                num_classes
            ),
        )

    def forward(self, x):

        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x)

        return self.head(x)


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    dataset_key,
    input_dim,
    num_classes
):

    config = DATASET_CONFIG[dataset_key]

    if config["model_type"] == "dataco":

        return DataCoResNetMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=config["hidden_dim"],
            num_blocks=config["num_blocks"],
            dropout=config["dropout"],
        )

    return EcommerceResNetMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config["hidden_dim"],
        num_blocks=config["num_blocks"],
        dropout=config["dropout"],
    )


# ============================================================
# LABEL MAPPING
# ============================================================
# A single mapping is created for ALL clients in a dataset.
# This is important: class index 0/1/2/... must mean the same
# class at every client and at the server.

def collect_global_label_values(dataset_key):

    config = DATASET_CONFIG[dataset_key]

    all_values = []

    for client in config["clients"]:

        file_path = os.path.join(
            config["data_dir"],
            f"client_{client}.csv"
        )

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Client file not found: {file_path}"
            )

        df = pd.read_csv(file_path)

        target = config["target"]

        if target not in df.columns:

            raise ValueError(
                f"Target '{target}' not found in {file_path}"
            )

        all_values.extend(
            df[target].dropna().tolist()
        )

    unique_values = sorted(
        pd.Series(all_values).unique().tolist()
    )

    if len(unique_values) < 2:

        raise ValueError(
            "Dataset must contain at least two classes."
        )

    value_to_class = {
        value: index
        for index, value in enumerate(unique_values)
    }

    return unique_values, value_to_class


# ============================================================
# LOAD CLIENT DATA
# ============================================================

def load_client_data(
    dataset_key,
    client,
    value_to_class
):

    config = DATASET_CONFIG[dataset_key]

    file_path = os.path.join(
        config["data_dir"],
        f"client_{client}.csv"
    )

    df = pd.read_csv(file_path)

    target = config["target"]

    y_raw = df[target].tolist()

    y = np.array(
        [
            value_to_class[value]
            for value in y_raw
        ],
        dtype=np.int64
    )

    # Keep the preprocessing representation already created
    # by the user's local scripts.
    drop_columns = [
        c
        for c in config["drop_columns"]
        if c in df.columns
    ]

    X_df = df.drop(
        columns=drop_columns
    )

    X_df = X_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_df = X_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_df = X_df.fillna(0.0)

    X = X_df.values.astype(
        np.float32
    )

    if len(X) != len(y):

        raise ValueError(
            f"Feature/label length mismatch for client {client}."
        )

    print()
    print("--------------------------------------------")
    print(
        f"{config['name']} | Client: {client}"
    )
    print(
        "Samples:",
        len(y)
    )
    print(
        "Features:",
        X.shape[1]
    )

    return X, y


# ============================================================
# TRAIN/VALIDATION SPLIT
# ============================================================

def split_client_data(X, y):

    class_counts = np.bincount(y)

    stratify = (
        y
        if (
            len(class_counts) > 1
            and class_counts.min() >= 2
        )
        else None
    )

    return train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=stratify,
    )


# ============================================================
# FULL-BATCH LOADERS
# ============================================================

def create_full_batch_loaders(
    X_train,
    y_train,
    X_val,
    y_val
):

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train,
            dtype=torch.long
        ),
    )

    val_dataset = TensorDataset(
        torch.tensor(
            X_val,
            dtype=torch.float32
        ),
        torch.tensor(
            y_val,
            dtype=torch.long
        ),
    )

    # FedProx experiment in the paper uses full-batch
    # local update for the federated methods.
    train_loader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


# ============================================================
# PROXIMAL LOSS
# ============================================================

def calculate_proximal_term(
    local_model,
    global_model
):

    proximal_term = torch.tensor(
        0.0,
        device=device
    )

    for local_parameter, global_parameter in zip(
        local_model.parameters(),
        global_model.parameters()
    ):

        proximal_term = (
            proximal_term
            + torch.sum(
                (
                    local_parameter
                    - global_parameter.detach()
                ) ** 2
            )
        )

    return proximal_term


# ============================================================
# LOCAL VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_true = []
    all_pred = []

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(
            X_batch
        )

        loss = criterion(
            outputs,
            y_batch
        )

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_true.extend(
            y_batch.cpu().numpy()
        )

        all_pred.extend(
            predictions.cpu().numpy()
        )

    average_loss = (
        total_loss
        / max(total_samples, 1)
    )

    accuracy = accuracy_score(
        all_true,
        all_pred
    )

    precision = precision_score(
        all_true,
        all_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_true,
        all_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_true,
        all_pred,
        average="weighted",
        zero_division=0
    )

    return (
        float(average_loss),
        float(accuracy),
        float(precision),
        float(recall),
        float(f1),
        np.asarray(all_true),
        np.asarray(all_pred),
    )


# ============================================================
# CLIENT UPDATE
# ============================================================

def client_update(
    dataset_key,
    client,
    global_model,
    X,
    y,
    num_classes,
    round_number
):

    config = DATASET_CONFIG[dataset_key]

    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = split_client_data(
        X,
        y
    )

    train_loader, val_loader = (
        create_full_batch_loaders(
            X_train,
            y_train,
            X_val,
            y_val
        )
    )

    # --------------------------------------------------------
    # IMPORTANT PAPER STEP:
    # theta_m <- theta_g
    # --------------------------------------------------------

    local_model = create_model(
        dataset_key,
        input_dim=X.shape[1],
        num_classes=num_classes
    ).to(device)

    local_model.load_state_dict(
        copy.deepcopy(
            global_model.state_dict()
        )
    )

    # Local classification loss.
    criterion = nn.CrossEntropyLoss()

    # Paper experimental setup: Adam.
    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE[dataset_key],
        weight_decay=WEIGHT_DECAY
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "proximal_loss": [],
    }

    # --------------------------------------------------------
    # LOCAL TRAINING
    # --------------------------------------------------------

    for epoch in range(
        1,
        LOCAL_EPOCHS + 1
    ):

        local_model.train()

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = local_model(
                X_batch
            )

            classification_loss = criterion(
                outputs,
                y_batch
            )

            # FedProx/FedPRADA local regularization:
            #
            # L_prox =
            # L_local + (mu/2)||theta_m-theta_g||^2

            proximal_term = calculate_proximal_term(
                local_model,
                global_model
            )

            proximal_loss = (
                MU_PROX / 2.0
            ) * proximal_term

            total_loss = (
                classification_loss
                + proximal_loss
            )

            total_loss.backward()

            optimizer.step()

        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            y_true,
            y_pred,
        ) = evaluate(
            local_model,
            val_loader,
            criterion
        )

        history["epoch"].append(
            epoch
        )

        history["train_loss"].append(
            float(total_loss.detach().cpu())
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["val_precision"].append(
            val_precision
        )

        history["val_recall"].append(
            val_recall
        )

        history["val_f1"].append(
            val_f1
        )

        history["proximal_loss"].append(
            float(proximal_loss.detach().cpu())
        )

        print(
            f"[{config['name']}/{client}] "
            f"Round {round_number} | "
            f"Local Epoch {epoch}/{LOCAL_EPOCHS} | "
            f"Val Loss={val_loss:.6f} | "
            f"Val Acc={val_accuracy:.4f} | "
            f"Val F1={val_f1:.4f}"
        )

    # --------------------------------------------------------
    # PAPER ROUND-LEVEL REPORT:
    # n_m = |D_m^train|
    #
    # c_m(c) = label count on the TRAINING SPLIT.
    # --------------------------------------------------------

    label_counts = np.bincount(
        y_train,
        minlength=num_classes
    )

    # Validation loss required by the paper's round-level
    # workflow for later temporal-stability weighting.
    final_val_loss = history["val_loss"][-1]

    # --------------------------------------------------------
    # SAVE LOCAL MODEL UPDATE
    # --------------------------------------------------------

    client_output = os.path.join(
        config["output_dir"],
        f"round_{round_number}",
        f"client_{client}"
    )

    os.makedirs(
        client_output,
        exist_ok=True
    )

    local_state_cpu = {
        key: value.detach()
        .cpu()
        .clone()
        for key, value
        in local_model.state_dict().items()
    }

    torch.save(
        {
            "dataset": config["name"],
            "client": client,
            "round": round_number,
            "model_state_dict": local_state_cpu,
            "num_train_samples": len(X_train),
            "num_val_samples": len(X_val),
            "label_counts": label_counts.tolist(),
            "validation_loss": final_val_loss,
            "learning_rate": LEARNING_RATE[dataset_key],
            "mu_prox": MU_PROX,
            "local_epochs": LOCAL_EPOCHS,
        },
        os.path.join(
            client_output,
            "local_model_update.pth"
        )
    )

    # --------------------------------------------------------
    # SAVE LOCAL HISTORY
    # --------------------------------------------------------

    pd.DataFrame(history).to_csv(
        os.path.join(
            client_output,
            "local_training_history.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

    plt.title(
        f"{config['name']} - Client {client} "
        f"- Module 2 Validation Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.colorbar()

    for i in range(num_classes):

        for j in range(num_classes):

            plt.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            client_output,
            "confusion_matrix.png"
        ),
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE CLASSIFICATION REPORT
    # --------------------------------------------------------

    with open(
        os.path.join(
            client_output,
            "classification_report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            classification_report(
                y_true,
                y_pred,
                zero_division=0
            )
        )

    return {
        "client": client,
        "num_train_samples": int(len(X_train)),
        "num_val_samples": int(len(X_val)),
        "label_counts": label_counts.astype(
            int
        ).tolist(),
        "validation_loss": float(
            final_val_loss
        ),
        "validation_accuracy": float(
            history["val_accuracy"][-1]
        ),
        "validation_precision": float(
            history["val_precision"][-1]
        ),
        "validation_recall": float(
            history["val_recall"][-1]
        ),
        "validation_f1": float(
            history["val_f1"][-1]
        ),
        "local_model_file": os.path.join(
            client_output,
            "local_model_update.pth"
        ),
    }


# ============================================================
# SERVER: GLOBAL MODEL INITIALIZATION
# ============================================================

def initialize_global_model(
    dataset_key,
    input_dim,
    num_classes
):

    global_model = create_model(
        dataset_key,
        input_dim=input_dim,
        num_classes=num_classes
    ).to(device)

    # Initial server-side model theta_g^1.
    initial_state = {
        key: value.detach()
        .cpu()
        .clone()
        for key, value
        in global_model.state_dict().items()
    }

    return global_model, initial_state


# ============================================================
# SERVER: GLOBAL LABEL DISTRIBUTION
# ============================================================

def construct_global_label_distribution(
    client_results,
    class_values
):

    num_classes = len(
        class_values
    )

    global_counts = np.zeros(
        num_classes,
        dtype=np.int64
    )

    total_train_samples = 0

    for result in client_results:

        counts = np.asarray(
            result["label_counts"],
            dtype=np.int64
        )

        global_counts += counts

        total_train_samples += int(
            result["num_train_samples"]
        )

    if total_train_samples > 0:

        global_distribution = (
            global_counts
            / total_train_samples
        )

    else:

        global_distribution = np.zeros(
            num_classes,
            dtype=np.float64
        )

    return (
        global_counts,
        global_distribution,
        total_train_samples
    )


# ============================================================
# SAVE GLOBAL LABEL DISTRIBUTION GRAPH
# ============================================================

def save_label_distribution_graph(
    output_dir,
    class_values,
    global_counts,
    global_distribution
):

    labels = [
        str(value)
        for value in class_values
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        labels,
        global_counts
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Training Samples"
    )

    plt.title(
        "Global Label Distribution - Module 2"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "global_label_distribution.png"
        ),
        dpi=200
    )

    plt.close()

    # Percentage distribution graph

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        labels,
        global_distribution * 100.0
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Percentage (%)"
    )

    plt.title(
        "Global Label Distribution (%) - Module 2"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "global_label_distribution_percentage.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# SAVE CLIENT RESULT GRAPH
# ============================================================

def save_client_metrics_graph(
    output_dir,
    client_results
):

    clients = [
        result["client"]
        for result in client_results
    ]

    accuracies = [
        result["validation_accuracy"]
        for result in client_results
    ]

    f1_scores = [
        result["validation_f1"]
        for result in client_results
    ]

    x = np.arange(
        len(clients)
    )

    width = 0.35

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        x - width / 2,
        accuracies,
        width,
        label="Validation Accuracy"
    )

    plt.bar(
        x + width / 2,
        f1_scores,
        width,
        label="Validation F1"
    )

    plt.xticks(
        x,
        clients
    )

    plt.ylim(
        0,
        1
    )

    plt.xlabel(
        "Client"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Client Validation Results - Module 2"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "client_validation_results.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# MAIN MODULE 2
# ============================================================

def _run_module_2(dataset_key, secure_clients=None, coordinator=None):

    config = DATASET_CONFIG[dataset_key]

    output_dir = config["output_dir"]

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print()
    print("=" * 75)
    print(
        f"FedPRADA MODULE 2 - {config['name']}"
    )
    print("=" * 75)

    print(
        "Global rounds:",
        GLOBAL_ROUNDS
    )

    print(
        "Local epochs:",
        LOCAL_EPOCHS
    )

    print(
        "Optimizer: Adam"
    )

    print(
        "Learning rate:",
        LEARNING_RATE[dataset_key]
    )

    print(
        "Proximal coefficient mu:",
        MU_PROX
    )

    # --------------------------------------------------------
    # SERVER PREPARATION
    # --------------------------------------------------------

    class_values, value_to_class = (
        collect_global_label_values(
            dataset_key
        )
    )

    num_classes = len(
        class_values
    )

    print()
    print(
        "Global class values:",
        class_values
    )

    print(
        "Number of classes:",
        num_classes
    )

    # --------------------------------------------------------
    # Load first client only to determine common input size.
    # Every client must have the same feature representation.
    # --------------------------------------------------------

    first_client = config["clients"][0]

    X_first, y_first = load_client_data(
        dataset_key,
        first_client,
        value_to_class
    )

    input_dim = X_first.shape[1]

    # Check all clients have same number of features.
    for client in config["clients"][1:]:

        X_check, _ = load_client_data(
            dataset_key,
            client,
            value_to_class
        )

        if X_check.shape[1] != input_dim:

            raise ValueError(
                "All participating clients must have "
                "the same input feature dimension. "
                f"{first_client} has {input_dim}, "
                f"but {client} has {X_check.shape[1]}."
            )

    print()
    print(
        "Server input dimension:",
        input_dim
    )

    # --------------------------------------------------------
    # 1. INITIALIZE GLOBAL MODEL
    # --------------------------------------------------------

    global_model, initial_state = (
        initialize_global_model(
            dataset_key,
            input_dim,
            num_classes
        )
    )

    initial_model_path = os.path.join(
        output_dir,
        "global_model_initial.pth"
    )

    torch.save(
        {
            "dataset": config["name"],
            "round": 1,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "class_values": class_values,
            "model_state_dict": initial_state,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE[
                dataset_key
            ],
            "mu_prox": MU_PROX,
            "local_epochs": LOCAL_EPOCHS,
        },
        initial_model_path
    )

    print()
    print(
        "SERVER: Global model initialized."
    )

    print(
        "Saved:",
        initial_model_path
    )

    # --------------------------------------------------------
    # 2. BROADCAST GLOBAL MODEL
    # --------------------------------------------------------

    print()
    print(
        "SERVER: Broadcasting global model "
        "to participating clients..."
    )

    broadcast_state = copy.deepcopy(
        global_model.state_dict()
    )

    if coordinator is not None:
        coordinator.begin_round(
            config["name"],
            1,
            broadcast_state,
        )

    client_results = []

    # --------------------------------------------------------
    # 3. CLIENT LOCAL UPDATES
    # --------------------------------------------------------

    for client in config["clients"]:

        print()
        print(
            "-" * 75
        )

        print(
            f"SERVER -> CLIENT {client}: "
            "global model sent."
        )

        # Reload private client data.
        if client == first_client:

            X = X_first
            y = y_first

        else:

            X, y = load_client_data(
                dataset_key,
                client,
                value_to_class
            )

        # Every client starts from the exact same server model.
        if secure_clients is not None:
            broadcast = secure_clients[client].exchange(
                {
                    "action": "broadcast",
                    "dataset": config["name"],
                    "round": 1,
                    "metadata": {},
                }
            )
            client_broadcast_state = broadcast["model_state"]
        else:
            client_broadcast_state = broadcast_state

        global_model.load_state_dict(
            copy.deepcopy(
                client_broadcast_state
            )
        )

        result = client_update(
            dataset_key=dataset_key,
            client=client,
            global_model=global_model,
            X=X,
            y=y,
            num_classes=num_classes,
            round_number=1,
        )

        if secure_clients is not None:
            client_checkpoint = torch.load(
                result["local_model_file"],
                map_location="cpu",
                weights_only=True,
            )
            acknowledgment = secure_clients[client].exchange(
                {
                    "action": "update",
                    "dataset": config["name"],
                    "round": 1,
                    "metadata": {
                        key: value
                        for key, value in result.items()
                        if key != "local_model_file"
                    },
                    "model_state": client_checkpoint["model_state_dict"],
                }
            )
            if acknowledgment.get("metadata", {}).get("accepted") is not True:
                raise RuntimeError(
                    f"Server did not accept the update from client {client}."
                )

        client_results.append(
            result
        )

        print(
            f"CLIENT {client} -> SERVER: "
            "local model parameters + "
            "sample count + label counts + "
            "validation loss returned."
        )

    # --------------------------------------------------------
    # 4. CONSTRUCT GLOBAL LABEL DISTRIBUTION
    # --------------------------------------------------------

    (
        global_counts,
        global_distribution,
        total_train_samples
    ) = construct_global_label_distribution(
        client_results,
        class_values
    )

    print()
    print(
        "=" * 75
    )

    print(
        "SERVER: GLOBAL LABEL DISTRIBUTION"
    )

    print(
        "=" * 75
    )

    for class_value, count, probability in zip(
        class_values,
        global_counts,
        global_distribution
    ):

        print(
            f"Class {class_value}: "
            f"count={int(count)}, "
            f"distribution={probability:.6f}"
        )

    # --------------------------------------------------------
    # SAVE GLOBAL LABEL DISTRIBUTION
    # --------------------------------------------------------

    label_distribution = {
        "dataset": config["name"],
        "round": 1,
        "total_training_samples": int(
            total_train_samples
        ),
        "classes": [
            {
                "class": str(class_value),
                "count": int(count),
                "distribution": float(probability),
            }
            for class_value, count, probability
            in zip(
                class_values,
                global_counts,
                global_distribution
            )
        ],
    }

    with open(
        os.path.join(
            output_dir,
            "global_label_distribution.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            label_distribution,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # SAVE CLIENT UPDATE SUMMARY
    # --------------------------------------------------------

    with open(
        os.path.join(
            output_dir,
            "client_updates_round_1.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            client_results,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # SAVE MODULE SUMMARY
    # --------------------------------------------------------

    summary = {
        "module": "2. Server Initialization and Client Update Module",
        "dataset": config["name"],
        "global_rounds": GLOBAL_ROUNDS,
        "clients": config["clients"],
        "num_clients": len(config["clients"]),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "classes": [
            str(value)
            for value in class_values
        ],
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE[
            dataset_key
        ],
        "local_epochs": LOCAL_EPOCHS,
        "mu_prox": MU_PROX,
        "aggregation_performed": False,
        "global_model_initialized": True,
        "global_model_broadcast": True,
        "local_updates_collected": True,
        "global_label_distribution_constructed": True,
        "client_results": client_results,
        "global_label_distribution": label_distribution,
    }

    with open(
        os.path.join(
            output_dir,
            "module_2_summary.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # GRAPHS
    # --------------------------------------------------------

    save_label_distribution_graph(
        output_dir,
        class_values,
        global_counts,
        global_distribution
    )

    save_client_metrics_graph(
        output_dir,
        client_results
    )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        f"MODULE 2 COMPLETED - {config['name']}"
    )
    print("=" * 75)

    print(
        "Global model:",
        initial_model_path
    )

    print(
        "Global label distribution:",
        os.path.join(
            output_dir,
            "global_label_distribution.json"
        )
    )

    print(
        "Client updates:",
        os.path.join(
            output_dir,
            "client_updates_round_1.json"
        )
    )

    print(
        "Results directory:",
        output_dir
    )

    print()
    print(
        "IMPORTANT: No server model aggregation was "
        "performed in Module 2."
    )

    print(
        "The collected local model updates are ready "
        "for the next FedPRADA aggregation module."
    )


def run_module_2(dataset_key, tls_config=None):
    if tls_config is None:
        return _run_module_2(dataset_key)

    clients = DATASET_CONFIG[dataset_key]["clients"]
    missing_tokens = set(clients).difference(tls_config["client_tokens"])
    if missing_tokens:
        raise ValueError(
            f"Missing TLS client tokens for: {sorted(missing_tokens)}"
        )

    coordinator = FederatedRoundCoordinator(clients)
    with TLSFederatedServer(
        certfile=tls_config["certfile"],
        keyfile=tls_config["keyfile"],
        client_tokens=tls_config["client_tokens"],
        handler=coordinator.handle,
    ) as server:
        secure_clients = {
            client: TLSFederatedClient(
                host=server.address[0],
                port=server.address[1],
                ca_file=tls_config["ca_file"],
                client_id=client,
                token=tls_config["client_tokens"][client],
            )
            for client in clients
        }
        return _run_module_2(
            dataset_key,
            secure_clients=secure_clients,
            coordinator=coordinator,
        )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "FedPRADA Module 2: Server Initialization "
            "and Client Update"
        )
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "dataco",
            "ecommerce",
        ],
        required=True,
        help=(
            "Run Module 2 for DataCo or "
            "E-Commerce Shipping."
        ),
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=MU_PROX,
        help=(
            "FedProx/FedPRADA proximal coefficient. "
            "Paper candidate values: "
            "0.001, 0.01, 0.1, 0.5, 1."
        ),
    )
    add_tls_arguments(parser)

    args = parser.parse_args()

    MU_PROX = args.mu

    run_module_2(
        args.dataset,
        tls_config=tls_config_from_args(args),
    )
