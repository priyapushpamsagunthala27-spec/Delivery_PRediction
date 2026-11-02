"""
FedPRADA Module 4: Prediction and Performance Evaluation

Algorithms:
    - FedPRADA final global model
    - ResNet-MLP

Metrics:
    - ACC: Accuracy
    - SEN: Sensitivity
    - SPE: Specificity
    - AUC: Area Under the ROC Curve

Exactly four graphs are generated.
"""

import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


# ============================================================
# 1. REPRODUCIBILITY AND DEVICE
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "dataco": {
        "display_name": "DataCo",

        "clients": [
            "africa",
            "europe",
            "latam",
            "pacific_asia",
            "usca",
        ],

        "data_dir": Path("processed/clients/dataco"),

        "target": "Delivery Status",

        "drop_columns": [
            "Delivery Status",
            "Days for shipping (real)",
            "Late_delivery_risk",
            "Market",
        ],

        "model_type": "dataco",

        "hidden_dim": 128,
        "num_blocks": 3,
        "dropout": 0.20,

        # Change this if your Module 3 uses a different filename.
        "model_path": Path(
            "outputs/fedprada_module3_dataco/"
            "global_model_final.pth"
        ),
    },

    "ecommerce": {
        "display_name": "E-Commerce",

        "clients": ["A", "B", "C", "D", "F"],

        "data_dir": Path("processed/clients"),

        "target": "Reached.on.Time_Y.N",

        "drop_columns": [
            "Reached.on.Time_Y.N",
        ],

        "model_type": "ecommerce",

        "hidden_dim": 256,
        "num_blocks": 4,
        "dropout": 0.10,

        # Change this if your Module 3 uses a different filename.
        "model_path": Path(
            "outputs/fedprada_module3_ecommerce/"
            "global_model_final.pth"
        ),
    },
}


OUT_DIR = Path(
    "outputs/fedprada_module4_prediction_evaluation"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. DATACO RESNET-MLP MODEL
# ============================================================

class DataCoResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout):

        super().__init__()

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        identity = x

        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.layer_norm(x + identity)


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

        self.blocks = nn.ModuleList([
            DataCoResidualBlock(
                hidden_dim,
                dropout
            )
            for _ in range(num_blocks)
        ])

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
# 4. E-COMMERCE RESNET-MLP MODEL
# ============================================================

class EcommerceResidualBlock(nn.Module):

    def __init__(self, hidden_dim, dropout):

        super().__init__()

        self.fc1 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

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

        return out + identity


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

            nn.BatchNorm1d(hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            EcommerceResidualBlock(
                hidden_dim,
                dropout
            )
            for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.BatchNorm1d(hidden_dim // 2),

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
# 5. MODEL CREATION
# ============================================================

def make_model(cfg, input_dim, num_classes):

    if cfg["model_type"] == "dataco":

        return DataCoResNetMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg["hidden_dim"],
            num_blocks=cfg["num_blocks"],
            dropout=cfg["dropout"],
        )

    return EcommerceResNetMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"],
        dropout=cfg["dropout"],
    )


# ============================================================
# 6. DATA PREPARATION
# ============================================================

def safe_split(X, y):

    counts = np.bincount(y)

    stratify = (
        y
        if len(counts) > 1 and counts.min() >= 2
        else None
    )

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=stratify,
    )


def normalize_label(value):

    key = str(value)

    try:
        numeric = float(value)

        if numeric.is_integer():
            return str(int(numeric))

    except (ValueError, TypeError):
        pass

    return key


def load_dataset(cfg, class_values):

    class_to_index = {
        normalize_label(value): i
        for i, value in enumerate(class_values)
    }

    all_X = []
    all_y = []
    all_ids = []

    expected_features = None

    for client in cfg["clients"]:

        csv_path = (
            cfg["data_dir"]
            / f"client_{client}.csv"
        )

        if not csv_path.exists():

            raise FileNotFoundError(
                f"Missing processed client file: {csv_path}\n"
                "Run your existing preprocessing workflow first."
            )

        df = pd.read_csv(csv_path)

        target = cfg["target"]

        if target not in df.columns:

            raise ValueError(
                f"Target column '{target}' missing in {csv_path}"
            )

        y_values = []

        for value in df[target].tolist():

            key = normalize_label(value)

            if key not in class_to_index:

                raise ValueError(
                    f"Unknown label {value} in {csv_path}. "
                    f"Saved model classes: {class_values}"
                )

            y_values.append(class_to_index[key])

        y = np.asarray(
            y_values,
            dtype=np.int64
        )

        drop_columns = [
            col
            for col in cfg["drop_columns"]
            if col in df.columns
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
        ).fillna(0.0)

        X = X_df.to_numpy(
            dtype=np.float32
        )

        if expected_features is None:

            expected_features = X.shape[1]

        elif X.shape[1] != expected_features:

            raise ValueError(
                f"Feature count mismatch in {csv_path}"
            )

        X_train, X_test, y_train, y_test = safe_split(X, y)

        all_X.append(X_test)
        all_y.append(y_test)

        all_ids.extend([
            f"{client}:{i}"
            for i in range(len(y_test))
        ])

        print(
            f"{cfg['display_name']} | Client {client} | "
            f"Test samples: {len(y_test)}"
        )

    return (
        np.concatenate(all_X),
        np.concatenate(all_y),
        all_ids,
        expected_features,
    )


# ============================================================
# 7. LOAD THE FINAL GLOBAL MODEL
# ============================================================

def load_checkpoint(model_path):

    if not model_path.exists():

        raise FileNotFoundError(
            f"\nGlobal model not found:\n{model_path}\n\n"
            "Run Module 3 first and check the exact saved model "
            "filename. Update model_path in DATASETS if necessary."
        )

    try:

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

    return checkpoint


def get_model_state(checkpoint):

    if not isinstance(checkpoint, dict):

        raise ValueError(
            "Unsupported model checkpoint format."
        )

    # Supports common checkpoint key names.
    for key in [
        "model_state_dict",
        "global_model_state_dict",
        "state_dict",
    ]:

        if key in checkpoint:

            return checkpoint[key]

    # Also supports a checkpoint containing only state_dict tensors.
    if checkpoint and all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):

        return checkpoint

    raise ValueError(
        "Could not find the model state dictionary in the checkpoint. "
        "Check how Module 3 saves its final global model."
    )


def get_class_values(checkpoint, cfg):

    for key in [
        "class_values",
        "classes",
        "label_values",
    ]:

        if key in checkpoint:

            return list(checkpoint[key])

    # If the checkpoint does not save label values, derive them from
    # the processed client CSV files in a consistent order.
    values = []

    for client in cfg["clients"]:

        path = (
            cfg["data_dir"]
            / f"client_{client}.csv"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Client CSV not found: {path}"
            )

        df = pd.read_csv(path)

        values.extend(
            df[cfg["target"]].dropna().tolist()
        )

    normalized = sorted({
        normalize_label(value)
        for value in values
    })

    return normalized


# ============================================================
# 8. METRIC CALCULATION
# ============================================================

def calculate_specificity(y_true, y_pred, num_classes):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
    )

    class_specificity = []

    for i in range(num_classes):

        tp = cm[i, i]

        fn = cm[i, :].sum() - tp

        fp = cm[:, i].sum() - tp

        tn = cm.sum() - tp - fn - fp

        denominator = tn + fp

        if denominator > 0:

            value = tn / denominator

        else:

            value = np.nan

        class_specificity.append(float(value))

    mean_specificity = float(
        np.nanmean(class_specificity)
    )

    return mean_specificity, class_specificity, cm


def calculate_auc(y_true, probabilities, num_classes):

    try:

        if num_classes == 2:

            return float(
                roc_auc_score(
                    y_true,
                    probabilities[:, 1]
                )
            )

        return float(
            roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=np.arange(num_classes),
            )
        )

    except ValueError as exc:

        warnings.warn(
            f"AUC could not be calculated: {exc}"
        )

        return float("nan")


# ============================================================
# 9. SHIPMENT PREDICTION AND EVALUATION
# ============================================================

def evaluate_one(dataset_key, cfg):

    checkpoint = load_checkpoint(
        cfg["model_path"]
    )

    class_values = get_class_values(
        checkpoint,
        cfg
    )

    state_dict = get_model_state(
        checkpoint
    )

    X_test, y_true, row_ids, input_dim = load_dataset(
        cfg,
        class_values
    )

    num_classes = len(class_values)

    model = make_model(
        cfg,
        input_dim,
        num_classes
    )

    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError as exc:

        raise RuntimeError(
            f"\nModel architecture or parameters do not match "
            f"the saved {cfg['display_name']} global model.\n"
            "Ensure Module 4 uses the exact ResNet-MLP architecture "
            "and feature preprocessing from Module 3.\n"
            f"Details: {exc}"
        ) from exc

    model = model.to(DEVICE)
    model.eval()

    probability_parts = []

    # Batch prediction helps CPU-only laptops.
    with torch.no_grad():

        for start in range(0, len(X_test), 2048):

            batch = torch.tensor(
                X_test[start:start + 2048],
                dtype=torch.float32,
                device=DEVICE,
            )

            logits = model(batch)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            probability_parts.append(
                probabilities.cpu().numpy()
            )

    probabilities = np.concatenate(
        probability_parts,
        axis=0
    )

    y_pred = probabilities.argmax(axis=1)

    # ACC
    acc = float(
        accuracy_score(
            y_true,
            y_pred
        )
    )

    # SEN: macro-average class recall
    sensitivity = float(
        recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    # SPE: macro-average one-vs-rest specificity
    specificity, class_specificity, cm = (
        calculate_specificity(
            y_true,
            y_pred,
            num_classes,
        )
    )

    # AUC
    auc_value = calculate_auc(
        y_true,
        probabilities,
        num_classes,
    )

    class_sensitivity = recall_score(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        average=None,
        zero_division=0,
    ).astype(float).tolist()

    result = {
        "dataset_key": dataset_key,
        "dataset": cfg["display_name"],
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc_value,
        "num_test_samples": int(len(y_true)),
        "num_classes": int(num_classes),
        "class_values": [
            str(value)
            for value in class_values
        ],
        "class_sensitivity": class_sensitivity,
        "class_specificity": class_specificity,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "confusion_matrix": cm,
    }

    # Save shipment prediction results.
    pred_df = pd.DataFrame({
        "Sample_ID": row_ids,

        "Actual_Status": [
            str(class_values[int(i)])
            for i in y_true
        ],

        "Predicted_Status": [
            str(class_values[int(i)])
            for i in y_pred
        ],

        "Correct": (
            y_true == y_pred
        ),
    })

    for i, class_value in enumerate(class_values):

        pred_df[
            f"Probability_{class_value}"
        ] = probabilities[:, i]

    pred_df.to_csv(
        OUT_DIR / f"{dataset_key}_shipment_predictions.csv",
        index=False,
    )

    return result


# ============================================================
# 10. GRAPH 1: ACC, SEN, SPE, AUC COMPARISON
# ============================================================

def save_metric_comparison(results):

    metric_keys = [
        "accuracy",
        "sensitivity",
        "specificity",
        "auc",
    ]

    metric_labels = [
        "ACC",
        "SEN",
        "SPE",
        "AUC",
    ]

    x = np.arange(
        len(metric_keys)
    )

    width = 0.34

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for index, result in enumerate(results):

        values = [
            result[key]
            for key in metric_keys
        ]

        values = [
            value if np.isfinite(value) else 0
            for value in values
        ]

        bars = ax.bar(
            x + (
                index - (len(results) - 1) / 2
            ) * width,
            values,
            width,
            label=result["dataset"],
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8,
        )

    ax.set_xticks(
        x,
        metric_labels
    )

    ax.set_ylim(
        0,
        1.12
    )

    ax.set_ylabel(
        "Metric score (0–1)"
    )

    ax.set_title(
        "FedPRADA ResNet-MLP Performance Comparison"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "01_metrics_comparison.png",
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# 11. GRAPHS 2 AND 3: ROC CURVES
# ============================================================

def save_roc_graph(result):

    y_true = result["y_true"]

    probabilities = result["probabilities"]

    class_values = result["class_values"]

    num_classes = len(class_values)

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    if num_classes == 2:

        if len(np.unique(y_true)) == 2:

            fpr, tpr, _ = roc_curve(
                y_true,
                probabilities[:, 1],
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"AUC = {roc_auc:.3f}",
            )

        else:

            ax.text(
                0.5,
                0.5,
                "ROC unavailable: only one class in test data",
                ha="center",
            )

    else:

        binary_labels = label_binarize(
            y_true,
            classes=np.arange(num_classes),
        )

        for i in range(num_classes):

            if len(
                np.unique(binary_labels[:, i])
            ) < 2:

                continue

            fpr, tpr, _ = roc_curve(
                binary_labels[:, i],
                probabilities[:, i],
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            ax.plot(
                fpr,
                tpr,
                linewidth=1.8,
                label=(
                    f"{class_values[i]} "
                    f"(AUC={roc_auc:.3f})"
                ),
            )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        alpha=0.6,
        label="Random baseline",
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1.04
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate (Sensitivity)"
    )

    ax.set_title(
        f"{result['dataset']} Shipment Prediction ROC Curve"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="lower right",
        fontsize=8,
    )

    fig.tight_layout()

    if result["dataset_key"] == "dataco":

        filename = "02_dataco_roc_curve.png"

    else:

        filename = "03_ecommerce_roc_curve.png"

    fig.savefig(
        OUT_DIR / filename,
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# 12. GRAPH 4: CLASS-WISE SENSITIVITY AND SPECIFICITY
# ============================================================

def save_classwise_graph(results):

    labels = []
    sensitivity_values = []
    specificity_values = []

    for result in results:

        for i, class_value in enumerate(
            result["class_values"]
        ):

            labels.append(
                f"{result['dataset']}\n{class_value}"
            )

            sensitivity_values.append(
                result["class_sensitivity"][i]
            )

            value = result["class_specificity"][i]

            specificity_values.append(
                value if np.isfinite(value) else 0
            )

    x = np.arange(
        len(labels)
    )

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(max(11, len(labels) * 0.9), 6)
    )

    bars1 = ax.bar(
        x - width / 2,
        sensitivity_values,
        width,
        label="Sensitivity",
    )

    bars2 = ax.bar(
        x + width / 2,
        specificity_values,
        width,
        label="Specificity",
    )

    ax.set_xticks(
        x,
        labels,
        fontsize=8,
    )

    ax.set_ylim(
        0,
        1.12
    )

    ax.set_ylabel(
        "Score (0–1)"
    )

    ax.set_title(
        "Class-wise Sensitivity and Specificity Variation"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    ax.bar_label(
        bars1,
        fmt="%.2f",
        padding=2,
        fontsize=7,
    )

    ax.bar_label(
        bars2,
        fmt="%.2f",
        padding=2,
        fontsize=7,
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "04_classwise_sensitivity_specificity.png",
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# 13. SAVE METRICS AND REPORT
# ============================================================

def save_reports(results):

    metric_rows = []

    report_lines = [
        "FedPRADA Module 4",
        "Prediction and Performance Evaluation",
        "Model: Final global ResNet-MLP",
        "",
        "ACC: Accuracy",
        "SEN: Macro-average sensitivity",
        "SPE: Macro-average specificity",
        "AUC: ROC area under the curve",
        "",
        "Evaluation uses the held-out 20% split from each client CSV.",
        "",
    ]

    for result in results:

        metric_rows.append({
            "Dataset": result["dataset"],

            "Model": (
                "FedPRADA Final Global ResNet-MLP"
            ),

            "Test_Samples": result["num_test_samples"],

            "ACC": result["accuracy"],

            "SEN": result["sensitivity"],

            "SPE": result["specificity"],

            "AUC": result["auc"],
        })

        report_lines.extend([
            result["dataset"],

            f"Test samples: {result['num_test_samples']}",

            f"Classes: {', '.join(result['class_values'])}",

            f"ACC: {result['accuracy']:.6f}",

            f"SEN: {result['sensitivity']:.6f}",

            f"SPE: {result['specificity']:.6f}",

            (
                f"AUC: {result['auc']:.6f}"
                if np.isfinite(result["auc"])
                else "AUC: unavailable"
            ),

            "",
        ])

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df.to_csv(
        OUT_DIR / "evaluation_metrics.csv",
        index=False,
    )

    (
        OUT_DIR / "module4_evaluation_report.txt"
    ).write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    safe_rows = []

    for row in metric_rows:

        safe_row = {}

        for key, value in row.items():

            if isinstance(value, (float, np.floating)):

                safe_row[key] = (
                    float(value)
                    if np.isfinite(value)
                    else None
                )

            else:

                safe_row[key] = value

        safe_rows.append(safe_row)

    (
        OUT_DIR / "module4_summary.json"
    ).write_text(
        json.dumps(
            safe_rows,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("=" * 65)

    print(
        "FedPRADA Module 4: Prediction and Performance Evaluation"
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "Output directory:",
        OUT_DIR.resolve()
    )

    print("=" * 65)

    results = []

    for key, cfg in DATASETS.items():

        print(
            f"\nEvaluating {cfg['display_name']} global model..."
        )

        result = evaluate_one(
            key,
            cfg
        )

        results.append(result)

        print(
            f"ACC (Accuracy): {result['accuracy']:.4f}"
        )

        print(
            f"SEN (Sensitivity): {result['sensitivity']:.4f}"
        )

        print(
            f"SPE (Specificity): {result['specificity']:.4f}"
        )

        if np.isfinite(result["auc"]):

            print(
                f"AUC: {result['auc']:.4f}"
            )

        else:

            print(
                "AUC: unavailable for this test split"
            )

        print(
            f"Test samples: {result['num_test_samples']}"
        )

    # Exactly four graphs.
    save_metric_comparison(results)

    for result in results:
        save_roc_graph(result)

    save_classwise_graph(results)

    save_reports(results)

    print("\nModule 4 completed successfully.")

    print("\nExactly four graphs generated:")

    for filename in [
        "01_metrics_comparison.png",
        "02_dataco_roc_curve.png",
        "03_ecommerce_roc_curve.png",
        "04_classwise_sensitivity_specificity.png",
    ]:

        print(
            OUT_DIR / filename
        )

    print("\nOther output files:")

    for filename in [
        "evaluation_metrics.csv",
        "dataco_shipment_predictions.csv",
        "ecommerce_shipment_predictions.csv",
        "module4_evaluation_report.txt",
        "module4_summary.json",
    ]:

        print(
            OUT_DIR / filename
        )


if __name__ == "__main__":
    main()