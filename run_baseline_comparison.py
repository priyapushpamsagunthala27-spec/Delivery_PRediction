"""Run isolated FedPRADA baseline comparisons on the existing shipment data."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import fedprada_module3 as module3


OUTPUT_DIR = Path("outputs/fedprada_baseline_comparison")
PROX_CANDIDATES = (0.001, 0.01, 0.1, 0.5, 1.0)
FEDAWA_ALPHA_CANDIDATES = (100.0, 1000.0, 10000.0, 100000.0)
COMPARISON_METHODS = ("FedAvg", "FedProx", "FedAWA", "FedPRADA")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    module3.set_seed(seed)


def join_client_split(tensors, clients, split):
    features = torch.cat([tensors[client][f"X{split}"] for client in clients])
    labels = torch.cat([tensors[client][f"y{split}"] for client in clients])
    return features, labels


def score_predictions(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
    )
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "test_recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "test_f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "test_sample_count": int(len(y_true)),
        "confusion_matrix": matrix.tolist(),
    }


def state_to_cpu(state):
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
    }


def state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, state, dataset, method, num_classes, input_dim,
                    settings, class_values=None, feature_columns=None):
    checkpoint = {
        "model_state_dict": state_to_cpu(state),
        "dataset": dataset,
        "method": method,
        "num_classes": num_classes,
        "input_dim": input_dim,
        "settings": settings,
        "state_sha256": state_sha256(state),
    }
    if class_values is not None:
        checkpoint["class_values"] = list(class_values)
    if feature_columns is not None:
        checkpoint["feature_columns"] = list(feature_columns)
    torch.save(checkpoint, path)


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Final FedPRADA checkpoint not found: {path}. "
            "Run the existing Module 3 before this comparison."
        )
    try:
        return torch.load(path, map_location=module3.device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=module3.device)


def raw_feature_frame(frame, ds_key):
    features = frame.copy()
    if ds_key == "dataco":
        leakage_columns = (
            "Delivery Status",
            "Days for shipping (real)",
            "Late_delivery_risk",
            "Market",
            "Order Status",
            "order date (DateOrders)",
            "shipping date (DateOrders)",
        )
    else:
        leakage_columns = (
            "Reached.on.Time_Y.N",
            "Warehouse_block",
            "ID",
        )

    missing_columns = [
        column for column in leakage_columns if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{ds_key} raw dataset is missing required excluded columns: "
            f"{missing_columns}"
        )
    return features.drop(columns=list(leakage_columns))


def fit_preprocessor(train_features, ds_key):
    feature_columns = train_features.columns.tolist()
    numeric_columns = train_features.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]
    medians = {}
    modes = {}
    category_values = {}
    encoded_train = train_features.copy()

    for column in numeric_columns:
        values = pd.to_numeric(encoded_train[column], errors="coerce")
        finite_values = values.replace([np.inf, -np.inf], np.nan)
        median = (
            float(finite_values.median())
            if finite_values.notna().any()
            else 0.0
        )
        if not np.isfinite(median):
            median = 0.0
        medians[column] = median
        encoded_train[column] = (
            values.replace([np.inf, -np.inf], np.nan)
            .fillna(median)
            .astype(np.float64)
        )

    for column in categorical_columns:
        values = encoded_train[column].astype("string")
        non_missing = values.dropna()
        mode = str(non_missing.mode().iloc[0]) if not non_missing.empty else ""
        cleaned = values.fillna(mode).astype(str)
        categories = sorted(cleaned.unique().tolist())
        category_values[column] = categories
        modes[column] = mode
        mapping = {value: index for index, value in enumerate(categories)}
        encoded_train[column] = cleaned.map(mapping).astype(np.float64)

    if ds_key == "dataco":
        scaled_columns = feature_columns
    else:
        scaled_columns = [
            column
            for column in (
                "Customer_care_calls",
                "Customer_rating",
                "Cost_of_the_Product",
                "Prior_purchases",
                "Discount_offered",
                "Weight_in_gms",
            )
            if column in feature_columns
        ]
    if not scaled_columns:
        raise ValueError(f"No training-fitted numeric features found for {ds_key}.")

    initial_scaler = StandardScaler()
    initial_scaler.fit(encoded_train[scaled_columns].to_numpy(dtype=np.float64))
    preprocessing = {
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "medians": medians,
        "modes": modes,
        "category_values": category_values,
        "scaled_columns": scaled_columns,
        "initial_scaler_mean": initial_scaler.mean_,
        "initial_scaler_scale": initial_scaler.scale_,
        "initial_scaler": initial_scaler,
    }
    return preprocessing


def transform_features(frame, preprocessing, ds_key):
    if frame.columns.tolist() != preprocessing["feature_columns"]:
        raise ValueError("Feature columns differ from the training-fitted schema.")
    transformed = frame.copy()
    for column in preprocessing["numeric_columns"]:
        values = pd.to_numeric(transformed[column], errors="coerce")
        transformed[column] = (
            values.replace([np.inf, -np.inf], np.nan)
            .fillna(preprocessing["medians"][column])
            .astype(np.float64)
        )

    for column in preprocessing["categorical_columns"]:
        categories = preprocessing["category_values"][column]
        mapping = {value: index for index, value in enumerate(categories)}
        values = transformed[column].astype("string").fillna(
            preprocessing["modes"][column]
        ).astype(str)
        transformed[column] = values.map(mapping).fillna(-1).astype(np.float64)

    transformed.loc[:, preprocessing["scaled_columns"]] = (
        preprocessing["initial_scaler"].transform(
            transformed[preprocessing["scaled_columns"]].to_numpy(dtype=np.float64)
        )
    )
    return transformed.to_numpy(dtype=np.float32)


def load_leakage_free_dataset(ds_key):
    if ds_key == "dataco":
        source_path = Path("DataCo/DataCoSupplyChainDataset.csv")
        if not source_path.exists():
            raise FileNotFoundError(f"DataCo raw source not found: {source_path}")
        frame = pd.read_csv(source_path, encoding="latin1").drop_duplicates()
        target_column = "Delivery Status"
        client_column = "Market"
        if frame[target_column].isna().any():
            raise ValueError(f"Missing target labels in {source_path}.")
        client_names = {
            "africa": "Africa",
            "europe": "Europe",
            "latam": "LATAM",
            "pacific_asia": "Pacific Asia",
            "usca": "USCA",
        }
        labels = sorted(frame[target_column].dropna().astype(str).unique().tolist())
        y = frame[target_column].astype(str).map(
            {label: index for index, label in enumerate(labels)}
        ).to_numpy(dtype=np.int64)
    else:
        source_path = Path("ecommerce/Train.csv")
        if not source_path.exists():
            raise FileNotFoundError(
                f"E-Commerce raw source not found: {source_path}"
            )
        frame = pd.read_csv(source_path).drop_duplicates()
        target_column = "Reached.on.Time_Y.N"
        client_column = "Warehouse_block"
        if frame[target_column].isna().any():
            raise ValueError(f"Missing target labels in {source_path}.")
        client_names = {
            client: client
            for client in module3.DATASETS["ecommerce"]["clients"]
        }
        labels = sorted(
            pd.to_numeric(frame[target_column], errors="raise")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        frame[target_column] = pd.to_numeric(
            frame[target_column],
            errors="raise",
        ).astype(int)
        y = frame[target_column].map(
            {label: index for index, label in enumerate(labels)}
        ).to_numpy(dtype=np.int64)

    if len(labels) != (4 if ds_key == "dataco" else 2):
        raise ValueError(
            f"{source_path} has {len(labels)} classes; expected "
            f"{4 if ds_key == 'dataco' else 2}."
        )

    clients = module3.DATASETS[ds_key]["clients"]
    client_values = {
        key: value
        for key, value in client_names.items()
    }
    raw_features = raw_feature_frame(frame, ds_key)
    split_indices = {}

    for client in clients:
        mask = frame[client_column].astype(str) == client_values[client]
        indices = np.flatnonzero(mask.to_numpy())
        if indices.size == 0:
            raise ValueError(f"No source records found for client {client}.")
        client_labels = y[indices]
        class_counts = np.bincount(client_labels)
        stratify = (
            client_labels
            if len(class_counts) > 1 and class_counts.min() >= 3
            else None
        )
        train_indices, held_indices = train_test_split(
            indices,
            test_size=0.40,
            random_state=module3.SEED,
            stratify=stratify,
        )
        held_labels = y[held_indices]
        held_counts = np.bincount(held_labels)
        held_stratify = (
            held_labels
            if len(held_counts) > 1 and held_counts.min() >= 2
            else None
        )
        val_indices, test_indices = train_test_split(
            held_indices,
            test_size=0.50,
            random_state=module3.SEED,
            stratify=held_stratify,
        )
        split_indices[client] = {
            "tr": train_indices,
            "va": val_indices,
            "te": test_indices,
        }
    train_feature_frames = [
        raw_features.iloc[split_indices[client]["tr"]]
        for client in clients
    ]
    train_features = pd.concat(train_feature_frames, ignore_index=True)
    preprocessing = fit_preprocessor(train_features, ds_key)

    transformed = {}
    for client in clients:
        transformed[client] = {}
        for split, indices in split_indices[client].items():
            values = transform_features(
                raw_features.iloc[indices],
                preprocessing,
                ds_key,
            )
            transformed[client][f"X{split}"] = values
            transformed[client][f"y{split}"] = y[indices]

    if ds_key == "ecommerce":
        train_arrays = [
            transformed[client]["Xtr"].astype(np.float64)
            for client in clients
        ]
        count = sum(len(values) for values in train_arrays)
        mean = sum(values.sum(axis=0) for values in train_arrays) / count
        square_mean = sum(
            np.square(values).sum(axis=0)
            for values in train_arrays
        ) / count
        scale = np.sqrt(np.maximum(square_mean - np.square(mean), 0.0))
        scale[scale < 1e-6] = 1.0
        for client in clients:
            for split in ("tr", "va", "te"):
                transformed[client][f"X{split}"] = (
                    (transformed[client][f"X{split}"] - mean) / scale
                ).astype(np.float32)
        preprocessing["federated_mean"] = mean
        preprocessing["federated_scale"] = scale

    tensors = {
        client: {
            key: torch.as_tensor(
                values,
                dtype=(torch.long if key.startswith("y") else torch.float32),
                device=module3.device,
            )
            for key, values in splits.items()
        }
        for client, splits in transformed.items()
    }
    metadata = {
        "source_file": str(source_path),
        "feature_columns": preprocessing["feature_columns"],
        "target_column": target_column,
        "excluded_target_leakage_columns": (
            [
                "Delivery Status",
                "Days for shipping (real)",
                "Late_delivery_risk",
                "Market",
                "Order Status",
                "order date (DateOrders)",
                "shipping date (DateOrders)",
            ]
            if ds_key == "dataco"
            else ["Reached.on.Time_Y.N", "Warehouse_block", "ID"]
        ),
        "fit_scope": "Train samples only: per-client 60/20/20 stratified split.",
        "class_values": [str(value) for value in labels],
        "clients": {
            client: {
                split: len(split_indices[client][split])
                for split in ("tr", "va", "te")
            }
            for client in clients
        },
        "imputation_medians": preprocessing["medians"],
        "imputation_modes": preprocessing["modes"],
        "categorical_values": preprocessing["category_values"],
        "categorical_vocabulary_sha256": {
            column: hashlib.sha256(
                json.dumps(
                    values,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for column, values in preprocessing["category_values"].items()
        },
        "categorical_vocabulary_sizes": {
            column: len(values)
            for column, values in preprocessing["category_values"].items()
        },
        "scaled_columns": preprocessing["scaled_columns"],
        "initial_scaler_mean": preprocessing["initial_scaler_mean"].tolist(),
        "initial_scaler_scale": preprocessing["initial_scaler_scale"].tolist(),
    }
    if "federated_mean" in preprocessing:
        metadata["federated_mean"] = preprocessing["federated_mean"].tolist()
        metadata["federated_scale"] = preprocessing["federated_scale"].tolist()

    return (
        tensors,
        labels,
        len(labels),
        len(preprocessing["feature_columns"]),
        preprocessing["feature_columns"],
        metadata,
    )


def save_round_metrics(dataset_dir, method, history, method_slug=None):
    frame = pd.DataFrame(history)
    frame.insert(0, "method", method)
    filename = method_slug or method.lower().replace(" ", "_")
    frame.to_csv(dataset_dir / f"{filename}_round_metrics.csv", index=False)
    return frame


def clean_comparison_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_suffixes = {".csv", ".json", ".png", ".pth", ".txt"}
    for dataset_key in ("dataco", "ecommerce"):
        dataset_dir = OUTPUT_DIR / dataset_key
        if dataset_dir.exists():
            for path in dataset_dir.iterdir():
                if path.is_file() and path.suffix.lower() in generated_suffixes:
                    path.unlink()
    for filename in ("comparison_results.csv", "comparison_results.json"):
        path = OUTPUT_DIR / filename
        if path.is_file():
            path.unlink()


def run_fedawa(ds_key, tensors, num_classes, input_dim, rounds, alpha, lr,
               batch_size, sample_fraction, mu):
    """FedAWA Algorithm 3: n_m exp(-alpha ||theta_m - theta_g||_2)."""
    config = module3.DATASETS[ds_key]
    clients = config["clients"]
    set_seed(module3.SEED)
    global_model = module3.create_model(ds_key, input_dim, num_classes)
    local_model = module3.create_model(ds_key, input_dim, num_classes)
    X_val, y_val = join_client_split(tensors, clients, "va")
    history = []
    final_state = None
    n_clients = len(clients)

    for round_number in range(1, rounds + 1):
        started = time.time()
        global_state = {
            key: value.detach().clone()
            for key, value in global_model.state_dict().items()
        }
        global_parameters = [
            parameter.detach().clone()
            for parameter in global_model.parameters()
        ]
        local_states = []
        client_sizes = []
        client_drifts = []
        client_train_losses = []

        for client_index, client in enumerate(clients):
            client_data = tensors[client]
            local_model.load_state_dict(global_state)
            seed = module3.SEED + 1000 * round_number + client_index
            rng = np.random.RandomState(seed)
            train_size = len(client_data["ytr"])
            if sample_fraction < 1.0:
                selected = rng.choice(
                    train_size,
                    size=max(int(sample_fraction * train_size), 2),
                    replace=False,
                )
            else:
                selected = np.arange(train_size)

            selected_tensor = torch.as_tensor(
                selected,
                dtype=torch.long,
                device=module3.device,
            )
            X_train = client_data["Xtr"][selected_tensor]
            y_train = client_data["ytr"][selected_tensor]
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loss = module3.local_update(
                local_model,
                global_parameters,
                X_train,
                y_train,
                lr,
                module3.LOCAL_EPOCHS,
                batch_size,
                mu,
                generator,
            )

            squared_drift = sum(
                torch.sum((parameter.detach() - initial) ** 2)
                for parameter, initial in zip(
                    local_model.parameters(),
                    global_parameters,
                )
            )
            client_drifts.append(float(torch.sqrt(squared_drift).item()))
            client_sizes.append(len(y_train))
            client_train_losses.append(train_loss)
            local_states.append(state_to_cpu(local_model.state_dict()))

        log_weights = (
            np.log(np.asarray(client_sizes, dtype=np.float64))
            - alpha * np.asarray(client_drifts, dtype=np.float64)
        )
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= weights.sum()

        fused_state = module3.fuse_states(local_states, weights)
        global_model.load_state_dict(fused_state)
        val_loss, val_accuracy, val_predictions = module3.evaluate(
            global_model,
            X_val,
            y_val,
        )
        val_f1 = f1_score(
            y_val.detach().cpu().numpy(),
            val_predictions.detach().cpu().numpy(),
            average="macro",
            zero_division=0,
        )
        history.append(
            {
                "round": round_number,
                "train_loss_mean": float(np.mean(client_train_losses)),
                "validation_loss": float(val_loss),
                "validation_accuracy": float(val_accuracy),
                "validation_f1_macro": float(val_f1),
                "client_drift_mean": float(np.mean(client_drifts)),
                "client_weights": json.dumps(
                    {
                        client: float(weight)
                        for client, weight in zip(clients, weights)
                    }
                ),
                "round_seconds": float(time.time() - started),
            }
        )
        final_state = state_to_cpu(global_model.state_dict())
        print(
            f"[{config['name']}|FedAWA] round {round_number}/{rounds} "
            f"validation_accuracy={val_accuracy:.4f} "
            f"weights={np.round(weights, 4).tolist()}"
        )

    return history, final_state


def prox_tuning_tensors(tensors, clients):
    tuning = {}
    for index, client in enumerate(clients):
        source = tensors[client]
        y = source["ytr"].detach().cpu().numpy()
        class_counts = np.bincount(y)
        stratify = (
            y
            if len(class_counts) > 1 and class_counts.min() >= 2
            else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            source["Xtr"].detach().cpu().numpy(),
            y,
            test_size=0.20,
            random_state=module3.SEED + index,
            stratify=stratify,
        )
        tuning[client] = {
            "Xtr": torch.as_tensor(
                X_train,
                dtype=torch.float32,
                device=module3.device,
            ),
            "ytr": torch.as_tensor(
                y_train,
                dtype=torch.long,
                device=module3.device,
            ),
            "Xva": torch.as_tensor(
                X_val,
                dtype=torch.float32,
                device=module3.device,
            ),
            "yva": torch.as_tensor(
                y_val,
                dtype=torch.long,
                device=module3.device,
            ),
        }
    return tuning


def evaluate_checkpoint(ds_key, checkpoint_path, tensors, num_classes, input_dim):
    checkpoint = load_checkpoint(checkpoint_path)
    if (
        int(checkpoint.get("input_dim", -1)) != input_dim
        or int(checkpoint.get("num_classes", -1)) != num_classes
    ):
        raise ValueError(
            f"Checkpoint dimensions do not match the {ds_key} client data."
        )
    model = module3.create_model(ds_key, input_dim, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    X_test, y_test = join_client_split(
        tensors,
        module3.DATASETS[ds_key]["clients"],
        "va",
    )
    with torch.inference_mode():
        logits = model(X_test)
        predictions = logits.argmax(dim=1)
    metrics = score_predictions(
        y_test.detach().cpu().numpy(),
        predictions.detach().cpu().numpy(),
        num_classes,
    )
    return metrics, checkpoint


def predict_state(ds_key, state, X, num_classes, input_dim):
    model = module3.create_model(ds_key, input_dim, num_classes)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        return model(X).argmax(dim=1).detach().cpu().numpy()


def evaluate_existing_fedprada(ds_key, dataset_dir):
    config = module3.DATASETS[ds_key]
    checkpoint_path = Path(config["output_dir"]) / "global_model_final.pth"
    legacy_tensors, _, num_classes, input_dim, _ = module3.load_dataset(
        ds_key,
        "natural",
    )
    metrics, checkpoint = evaluate_checkpoint(
        ds_key,
        checkpoint_path,
        legacy_tensors,
        num_classes,
        input_dim,
    )
    reference = {
        "dataset": config["name"],
        "method": "FedPRADA (existing checkpoint; legacy reference only)",
        "source_checkpoint": str(checkpoint_path),
        "checkpoint_round": checkpoint.get("round"),
        "metric_partition": (
            "Original Module 3 validation holdout; not an independent test set."
        ),
        **metrics,
        "comparability": (
            (
                "Not included in the leakage-free comparison table: this DataCo "
                "checkpoint uses the original pre-split preprocessing and "
                "Order Status, which deterministically reveals the Shipping "
                "canceled class. Its holdout also served as Module 3 validation."
            )
            if ds_key == "dataco"
            else (
                "Not included in the leakage-free comparison table: the original "
                "preprocessing was fit before splitting, and this holdout also "
                "served as Module 3 validation."
            )
        ),
    }
    with open(
        dataset_dir / "legacy_fedprada_reference.json",
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(reference, output, indent=2)
    pd.DataFrame(metrics["confusion_matrix"]).to_csv(
        dataset_dir / "legacy_fedprada_confusion_matrix.csv",
        index=False,
        header=False,
    )
    return reference


def fit_centralized(ds_key, tensors, num_classes, input_dim, epochs, batch_size):
    clients = module3.DATASETS[ds_key]["clients"]
    X_train, y_train = join_client_split(tensors, clients, "tr")
    X_val, y_val = join_client_split(tensors, clients, "va")
    model = module3.create_model(ds_key, input_dim, num_classes)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=module3.LEARNING_RATE[ds_key],
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(
            len(y_train),
            device=module3.device,
        )
        epoch_loss = 0.0
        samples_seen = 0
        for start in range(0, len(y_train), batch_size):
            indices = permutation[start:start + batch_size]
            if len(indices) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(X_train[indices]),
                y_train[indices],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), module3.GRAD_CLIP)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
            samples_seen += len(indices)

        validation_loss, validation_accuracy, _ = module3.evaluate(
            model,
            X_val,
            y_val,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(samples_seen, 1),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
    return history, state_to_cpu(model.state_dict()), model


def fit_distributed(ds_key, tensors, num_classes, input_dim, epochs, batch_size):
    clients = module3.DATASETS[ds_key]["clients"]
    history_by_client = {client: [] for client in clients}
    states = {}
    client_models = {}

    for client_index, client in enumerate(clients):
        set_seed(module3.SEED + client_index)
        data = tensors[client]
        model = module3.create_model(ds_key, input_dim, num_classes)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=(2e-4 if ds_key == "ecommerce" else module3.LEARNING_RATE[ds_key]),
        )
        for epoch in range(1, epochs + 1):
            model.train()
            permutation = torch.randperm(
                len(data["ytr"]),
                device=module3.device,
            )
            epoch_loss = 0.0
            samples_seen = 0
            full_batch_size = (
                batch_size
                if batch_size > 0
                else len(data["ytr"])
            )
            for start in range(0, len(permutation), full_batch_size):
                indices = permutation[start:start + full_batch_size]
                if len(indices) < 2:
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(
                    model(data["Xtr"][indices]),
                    data["ytr"][indices],
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    module3.GRAD_CLIP,
                )
                optimizer.step()
                epoch_loss += float(loss.item()) * len(indices)
                samples_seen += len(indices)

            val_loss, val_accuracy, _ = module3.evaluate(
                model,
                data["Xva"],
                data["yva"],
            )
            history_by_client[client].append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss / max(samples_seen, 1),
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy,
                }
            )
        states[client] = state_to_cpu(model.state_dict())
        client_models[client] = model

    pooled_history = []
    for epoch in range(epochs):
        pooled_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(
                    np.mean(
                        [
                            history_by_client[client][epoch]["train_loss"]
                            for client in clients
                        ]
                    )
                ),
                "validation_loss": float(
                    np.mean(
                        [
                            history_by_client[client][epoch]["validation_loss"]
                            for client in clients
                        ]
                    )
                ),
                "validation_accuracy": float(
                    np.mean(
                        [
                            history_by_client[client][epoch]["validation_accuracy"]
                            for client in clients
                        ]
                    )
                ),
            }
        )

    test_labels = []
    test_predictions = []
    for client in clients:
        data = tensors[client]
        model = client_models[client]
        model.eval()
        with torch.inference_mode():
            predictions = model(data["Xte"]).argmax(dim=1)
        test_labels.extend(data["yte"].detach().cpu().numpy().tolist())
        test_predictions.extend(predictions.detach().cpu().numpy().tolist())
    return pooled_history, states, test_labels, test_predictions


def save_method_result(dataset_dir, dataset, method, metrics, state, input_dim,
                       num_classes, settings, round_history, all_round_frames,
                       class_values=None, feature_columns=None):
    method_slug = method.lower().replace(" ", "_")
    metrics_row = {
        "dataset": dataset,
        "method": method,
        **{
            key: value
            for key, value in metrics.items()
            if key != "confusion_matrix"
        },
        "settings": settings,
    }
    metrics_row["confusion_matrix"] = metrics["confusion_matrix"]
    if state is not None:
        checkpoint_path = dataset_dir / f"{method_slug}_final.pth"
        save_checkpoint(
            checkpoint_path,
            state,
            dataset,
            method,
            num_classes,
            input_dim,
            settings,
            class_values,
            feature_columns,
        )
        metrics_row["checkpoint"] = str(checkpoint_path)

    pd.DataFrame(metrics["confusion_matrix"]).to_csv(
        dataset_dir / f"{method_slug}_test_confusion_matrix.csv",
        index=False,
        header=False,
    )
    if round_history:
        frame = save_round_metrics(
            dataset_dir,
            method,
            round_history,
            method_slug,
        )
        all_round_frames.append(frame)
    return metrics_row


def plot_test_accuracy(results, path, dataset):
    figure, axis = plt.subplots(figsize=(10, 5))
    methods = [row["method"] for row in results]
    values = [100 * row["test_accuracy"] for row in results]
    bars = axis.bar(methods, values, color="#4776a8")
    if values:
        best_index = int(np.argmax(values))
        bars[best_index].set_color("#27966f")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title(f"{dataset} - held-out test accuracy")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_convergence(round_frames, path, dataset):
    figure, axis = plt.subplots(figsize=(10, 5))
    colors = {
        "FedAvg": "#4472C4",
        "FedProx": "#ED7D31",
        "FedAWA": "#70AD47",
        "FedPRADA": "#A64D79",
    }
    plotted_methods = set()
    for method in COMPARISON_METHODS:
        frame = next(
            (
                item
                for item in round_frames
                if not item.empty and str(item["method"].iloc[0]) == method
            ),
            None,
        )
        if frame is None:
            continue
        if (
            "round" not in frame.columns
            or "validation_accuracy" not in frame.columns
            or "validation_loss" not in frame.columns
        ):
            raise ValueError(f"{method} history is missing convergence metrics.")
        frame = frame.sort_values("round")
        if frame["round"].duplicated().any() or frame[
            ["validation_accuracy", "validation_loss"]
        ].isna().any().any():
            raise ValueError(f"{method} convergence history is invalid.")
        axis.plot(
            frame["round"],
            frame["validation_accuracy"] * 100,
            label=method,
            color=colors[method],
            linewidth=1.8,
        )
        plotted_methods.add(method)
    if plotted_methods != set(COMPARISON_METHODS):
        raise ValueError(
            f"Cannot plot {dataset} convergence: missing histories for "
            f"{sorted(set(COMPARISON_METHODS) - plotted_methods)}."
        )
    axis.set_xlabel("Communication Round")
    axis.set_ylabel("Validation Accuracy (%)")
    axis.set_title(f"{dataset} - validation accuracy by communication round")
    axis.legend(title="Method")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_convergence_csv(round_frames, path, expected_rounds):
    required = set(COMPARISON_METHODS)
    selected_frames = [
        frame
        for frame in round_frames
        if not frame.empty and str(frame["method"].iloc[0]) in required
    ]
    found = {str(frame["method"].iloc[0]) for frame in selected_frames}
    if found != required:
        raise ValueError(
            f"Convergence histories incomplete. Found {sorted(found)}; "
            f"expected {sorted(required)}."
        )
    combined = pd.concat(selected_frames, ignore_index=True)
    combined = combined[
        ["round", "method", "validation_accuracy", "validation_loss"]
    ].sort_values(["method", "round"])
    for method in COMPARISON_METHODS:
        method_history = combined[combined["method"] == method]
        actual_rounds = method_history["round"].astype(int).tolist()
        expected = list(range(1, expected_rounds + 1))
        if actual_rounds != expected:
            raise ValueError(
                f"{method} has {len(actual_rounds)} rounds, expected "
                f"{expected_rounds} complete sequential rounds."
            )
    combined.to_csv(path, index=False)
    return combined


def verify_saved_methods(dataset_dir, rows, X_test, y_test, num_classes,
                        input_dim, expected_rounds):
    comparable = {
        row["method"]: row
        for row in rows
        if row["method"] in COMPARISON_METHODS
    }
    if set(comparable) != set(COMPARISON_METHODS):
        raise ValueError(
            f"Expected exactly {COMPARISON_METHODS}; found "
            f"{sorted(comparable)}."
        )

    state_signatures = set()
    validation_histories = set()
    for method in COMPARISON_METHODS:
        row = comparable[method]
        checkpoint_path = Path(row["checkpoint"])
        checkpoint = load_checkpoint(checkpoint_path)
        if (
            checkpoint.get("method") != method
            or checkpoint.get("dataset") != row["dataset"]
            or int(checkpoint.get("input_dim", -1)) != input_dim
            or int(checkpoint.get("num_classes", -1)) != num_classes
        ):
            raise ValueError(
                f"{method} checkpoint metadata does not match its result row."
            )
        state = checkpoint["model_state_dict"]
        signature = state_sha256(state)
        if signature != row["state_sha256"] or signature in state_signatures:
            raise ValueError(
                f"{method} checkpoint is stale, mis-mapped, or duplicates "
                "another method's model state."
            )
        state_signatures.add(signature)

        predictions = predict_state(
            "dataco" if row["dataset"] == "DataCo" else "ecommerce",
            state,
            X_test,
            num_classes,
            input_dim,
        )
        measured = score_predictions(y_test, predictions, num_classes)
        for key in (
            "test_accuracy",
            "test_precision_macro",
            "test_recall_macro",
            "test_f1_macro",
            "test_sample_count",
            "confusion_matrix",
        ):
            if measured[key] != row[key]:
                raise ValueError(
                    f"{method} saved metrics do not match its own checkpoint "
                    f"on the held-out test set ({key})."
                )

        history_path = dataset_dir / f"{method.lower()}_round_metrics.csv"
        history = pd.read_csv(history_path)
        if (
            history["method"].nunique() != 1
            or history["method"].iloc[0] != method
            or history["round"].astype(int).tolist()
            != list(range(1, expected_rounds + 1))
        ):
            raise ValueError(
                f"{method} round history is missing, stale, or mapped to "
                "another method."
            )
        history_signature = tuple(
            zip(
                history["validation_accuracy"].round(12),
                history["validation_loss"].round(12),
            )
        )
        if history_signature in validation_histories:
            raise ValueError(
                f"{method} validation history duplicates another method's "
                "round history."
            )
        validation_histories.add(history_signature)


def evaluate_dataset(ds_key, args, combined_rows):
    config = module3.DATASETS[ds_key]
    dataset = config["name"]
    dataset_dir = OUTPUT_DIR / ds_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rounds = args.rounds
    if rounds is None:
        rounds = 100 if ds_key == "dataco" else int(
            np.ceil(200 / module3.LOCAL_EPOCHS)
        )
    (
        tensors,
        class_values,
        num_classes,
        input_dim,
        feature_columns,
        preprocessing_metadata,
    ) = load_leakage_free_dataset(ds_key)
    legacy_reference = evaluate_existing_fedprada(ds_key, dataset_dir)
    settings = {
        "seed": module3.SEED,
        "rounds": rounds,
        "local_epochs_per_round": module3.LOCAL_EPOCHS,
        "equivalent_local_epochs": rounds * module3.LOCAL_EPOCHS,
        "learning_rate": module3.LEARNING_RATE[ds_key],
        "paper_reference": "2.pdf, Section 5.4 and Table 4",
        "paper_budget_note": (
            "Default budget follows the paper's target equivalent epochs: "
            "300 for DataCo and approximately 200 for E-Commerce. Explicit "
            "--rounds overrides this dataset-specific default."
        ),
        "paper_fedawa_learning_rate": 2e-4 if ds_key == "dataco" else 5e-4,
        "paper_distributed_learning_rate": (
            2e-4 if ds_key == "ecommerce" else module3.LEARNING_RATE[ds_key]
        ),
        "batch_size": args.batch_size,
        "round_sample_fraction": args.sample_fraction,
        "proximal_candidates": list(PROX_CANDIDATES),
        "prox_tuning_rounds": args.tune_rounds,
        "fedawa_alpha_candidates": list(FEDAWA_ALPHA_CANDIDATES),
        "fedawa_alpha_selection": (
            "The paper does not specify alpha; fixed at the recorded value "
            "before evaluation and not selected using test data."
        ),
        "fedawa_learning_rate": 2e-4 if ds_key == "dataco" else 5e-4,
        "fedprada_parameters": {
            "lambda_diversity": module3.LAMBDA_DIV["natural"],
            "gamma_stability": module3.GAMMA_STAB,
            "mu_proximal": module3.MU_PROX,
            "stability_window": module3.STAB_WINDOW,
        },
        "client_split": (
            "Per-existing-client stratified 60/20/20 train/validation/test; "
            "seed 42; identical records across all methods."
        ),
        "preprocessing": (
            "Fit imputation, categorical vocabularies, and scaling on training "
            "records only; apply unchanged to validation/test. DataCo excludes "
            "Order Status and both dates because the date difference reproduces "
            "Days for shipping (real), a target proxy."
        ),
        "evaluation_protocol_deviation": (
            "Single seeded stratified split rather than the paper's five-fold "
            "evaluation; no five-fold results are claimed because that would "
            "multiply the full CPU training budget. The original FedPRADA "
            "checkpoint is reported separately as a non-comparable legacy "
            "reference because its holdout was monitored during training and "
            "its preprocessing was fit before splitting."
        ),
        "dataset_source": preprocessing_metadata["source_file"],
        "feature_count": input_dim,
        "feature_columns": feature_columns,
        "class_values": [str(value) for value in class_values],
        "training_only_preprocessing_fit": True,
        "reported_precision_recall_f1_average": "Macro",
        "five_fold_evaluation": False,
        "legacy_fedprada_reference_accuracy": legacy_reference["test_accuracy"],
    }

    result_rows = []
    round_frames = []
    X_test, y_test = join_client_split(tensors, config["clients"], "te")
    y_test_numpy = y_test.detach().cpu().numpy()

    print(f"\n=== {dataset}: {input_dim} features, {num_classes} classes ===")
    print(
        f"Test samples: {len(y_test_numpy)}; class labels: {class_values}; "
        f"features: {input_dim}"
    )
    with open(
        dataset_dir / "preprocessing.json",
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(preprocessing_metadata, output, indent=2)

    def history_from_federated(history):
        return [
            {
                "round": index + 1,
                "train_loss_mean": float(
                    np.mean(history["local_train_loss"][index])
                ),
                "validation_loss": float(history["g_val_loss"][index]),
                "validation_accuracy": float(history["g_val_acc"][index]),
                "validation_f1_macro": float(history["g_val_f1"][index]),
                "round_seconds": float(history["round_time"][index]),
            }
            for index in range(rounds)
        ]

    def register_result(method, metrics, state, method_settings, history):
        row = save_method_result(
            dataset_dir,
            dataset,
            method,
            metrics,
            state,
            input_dim,
            num_classes,
            method_settings,
            history,
            round_frames,
            class_values,
            feature_columns,
        )
        if state is not None:
            row["state_sha256"] = state_sha256(state)
        result_rows.append(row)

    fedprada_history, _, best_round, fedprada_state = module3.run_federated(
        ds_key,
        "fedprada",
        tensors,
        num_classes,
        input_dim,
        rounds,
        module3.LAMBDA_DIV["natural"],
        module3.GAMMA_STAB,
        module3.MU_PROX,
        args.batch_size,
        args.sample_fraction,
        module3.STAB_WINDOW,
        tag="|leakage-free-comparison",
    )
    fedprada_predictions = predict_state(
        ds_key,
        fedprada_state,
        X_test,
        num_classes,
        input_dim,
    )
    fedprada_metrics = score_predictions(
        y_test_numpy,
        fedprada_predictions,
        num_classes,
    )
    register_result(
        "FedPRADA",
        fedprada_metrics,
        fedprada_state,
        {
            **settings,
            "checkpoint_round": rounds,
            "best_validation_round": best_round,
            "checkpoint_selection": "Final aggregated round.",
        },
        history_from_federated(fedprada_history),
    )

    if args.include_cl_dl:
        epochs = rounds * module3.LOCAL_EPOCHS
        cl_history, cl_state, _ = fit_centralized(
            ds_key,
            tensors,
            num_classes,
            input_dim,
            epochs,
            args.batch_size,
        )
        with torch.inference_mode():
            cl_model = module3.create_model(ds_key, input_dim, num_classes)
            cl_model.load_state_dict(cl_state, strict=True)
            cl_model.eval()
            cl_predictions = cl_model(X_test).argmax(dim=1)
        cl_metrics = score_predictions(
            y_test_numpy,
            cl_predictions.detach().cpu().numpy(),
            num_classes,
        )
        cl_history_by_round = [
            {
                "round": round_number,
                "validation_accuracy": cl_history[
                    min(round_number * module3.LOCAL_EPOCHS, epochs) - 1
                ]["validation_accuracy"],
                "validation_loss": cl_history[
                    min(round_number * module3.LOCAL_EPOCHS, epochs) - 1
                ]["validation_loss"],
            }
            for round_number in range(1, rounds + 1)
        ]
        result_rows.append(
            save_method_result(
                dataset_dir,
                dataset,
                "Centralized Learning",
                cl_metrics,
                cl_state,
                input_dim,
                num_classes,
                {**settings, "training_epochs": epochs},
                cl_history_by_round,
                round_frames,
                class_values,
                feature_columns,
            )
        )

        dl_history, dl_states, dl_y, dl_predictions = fit_distributed(
            ds_key,
            tensors,
            num_classes,
            input_dim,
            epochs,
            args.batch_size,
        )
        dl_metrics = score_predictions(dl_y, dl_predictions, num_classes)
        dl_history_by_round = [
            {
                "round": round_number,
                "validation_accuracy": dl_history[
                    min(round_number * module3.LOCAL_EPOCHS, epochs) - 1
                ]["validation_accuracy"],
                "validation_loss": dl_history[
                    min(round_number * module3.LOCAL_EPOCHS, epochs) - 1
                ]["validation_loss"],
            }
            for round_number in range(1, rounds + 1)
        ]
        result_rows.append(
            save_method_result(
                dataset_dir,
                dataset,
                "Distributed Learning",
                dl_metrics,
                None,
                input_dim,
                num_classes,
                {
                    **settings,
                    "training_epochs_per_client": epochs,
                    "evaluation": (
                        "Each local model evaluated on its own client's "
                        "held-out test records; pooled metrics cover the same "
                        "combined test partition as other methods."
                    ),
                },
                dl_history_by_round,
                round_frames,
                class_values,
                feature_columns,
            )
        )
        pd.DataFrame(
            [
                {
                    "client": client,
                    "checkpoint": str(
                        dataset_dir / f"distributed_learning_client_{client}.pth"
                    ),
                }
                for client in config["clients"]
            ]
        ).to_csv(dataset_dir / "distributed_learning_clients.csv", index=False)
        for client, state in dl_states.items():
            save_checkpoint(
                dataset_dir / f"distributed_learning_client_{client}.pth",
                state,
                dataset,
                f"Distributed Learning - {client}",
                num_classes,
                input_dim,
                settings,
                class_values,
                feature_columns,
            )

    fedavg_history, _, _, fedavg_state = module3.run_federated(
        ds_key,
        "fedavg",
        tensors,
        num_classes,
        input_dim,
        rounds,
        0.0,
        module3.GAMMA_STAB,
        0.0,
        args.batch_size,
        args.sample_fraction,
        module3.STAB_WINDOW,
        tag="|baseline",
    )
    fedavg_metrics = score_predictions(
        y_test_numpy,
        predict_state(ds_key, fedavg_state, X_test, num_classes, input_dim),
        num_classes,
    )
    register_result(
        "FedAvg",
        fedavg_metrics,
        fedavg_state,
        {
            **settings,
            "aggregation": "Training sample-size-weighted model average.",
        },
        history_from_federated(fedavg_history),
    )

    tuning_tensors = prox_tuning_tensors(tensors, config["clients"])
    selection = []
    for mu in PROX_CANDIDATES:
        history, _, _, _ = module3.run_federated(
            ds_key,
            "fedavg",
            tuning_tensors,
            num_classes,
            input_dim,
            args.tune_rounds,
            0.0,
            module3.GAMMA_STAB,
            mu,
            args.batch_size,
            args.sample_fraction,
            module3.STAB_WINDOW,
            tag=f"|FedProx-mu={mu:g}-tuning",
        )
        selection.append(
            {
                "mu": mu,
                "validation_accuracy": float(history["g_val_acc"][-1]),
                "validation_loss": float(history["g_val_loss"][-1]),
            }
        )
    selected = max(
        selection,
        key=lambda item: (
            item["validation_accuracy"],
            -item["validation_loss"],
        ),
    )
    with open(
        dataset_dir / "fedprox_hyperparameter_selection.json",
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            {
                "selection_data": (
                    "A seeded 20% validation subset drawn only from each "
                    "client's training partition; test records are excluded."
                ),
                "candidates": selection,
                "selected_mu": selected["mu"],
            },
            output,
            indent=2,
        )
    prox_history, _, _, prox_state = module3.run_federated(
        ds_key,
        "fedavg",
        tensors,
        num_classes,
        input_dim,
        rounds,
        0.0,
        module3.GAMMA_STAB,
        selected["mu"],
        args.batch_size,
        args.sample_fraction,
        module3.STAB_WINDOW,
        tag=f"|FedProx-mu={selected['mu']:g}",
    )
    prox_metrics = score_predictions(
        y_test_numpy,
        predict_state(ds_key, prox_state, X_test, num_classes, input_dim),
        num_classes,
    )
    register_result(
        "FedProx",
        prox_metrics,
        prox_state,
        {
            **settings,
            "selected_mu": selected["mu"],
            "aggregation": "Training sample-size-weighted model average.",
            "local_objective": (
                "Cross entropy + (mu / 2) * squared distance from the "
                "received global parameters."
            ),
        },
        history_from_federated(prox_history),
    )

    fedawa_tuning_tensors = prox_tuning_tensors(tensors, config["clients"])
    fedawa_selection = []
    for alpha in FEDAWA_ALPHA_CANDIDATES:
        alpha_history, _ = run_fedawa(
            ds_key,
            fedawa_tuning_tensors,
            num_classes,
            input_dim,
            args.tune_rounds,
            alpha,
            settings["fedawa_learning_rate"],
            args.batch_size,
            args.sample_fraction,
            0.0,
        )
        fedawa_selection.append(
            {
                "alpha": alpha,
                "validation_accuracy": alpha_history[-1]["validation_accuracy"],
                "validation_loss": alpha_history[-1]["validation_loss"],
            }
        )
    selected_alpha = max(
        fedawa_selection,
        key=lambda item: (
            item["validation_accuracy"],
            -item["validation_loss"],
        ),
    )
    with open(
        dataset_dir / "fedawa_hyperparameter_selection.json",
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            {
                "selection_data": (
                    "Seeded validation subset split from training records only."
                ),
                "candidates": fedawa_selection,
                "selected_alpha": selected_alpha["alpha"],
            },
            output,
            indent=2,
        )
    awa_history, awa_state = run_fedawa(
        ds_key,
        tensors,
        num_classes,
        input_dim,
        rounds,
        selected_alpha["alpha"],
        settings["fedawa_learning_rate"],
        args.batch_size,
        args.sample_fraction,
        0.0,
    )
    awa_metrics = score_predictions(
        y_test_numpy,
        predict_state(ds_key, awa_state, X_test, num_classes, input_dim),
        num_classes,
    )
    register_result(
        "FedAWA",
        awa_metrics,
        awa_state,
        {
            **settings,
            "fedawa_alpha_candidates": list(FEDAWA_ALPHA_CANDIDATES),
            "selected_alpha": selected_alpha["alpha"],
            "fedawa_formula": (
                "w_m proportional to n_m * exp(-alpha * "
                "L2_norm(theta_m - theta_global))"
            ),
        },
        awa_history,
    )

    for row in result_rows:
        row["class_values"] = [str(value) for value in class_values]
        row["experiment_settings"] = row["settings"]
        combined_rows.append(row)

    verify_saved_methods(
        dataset_dir,
        result_rows,
        X_test,
        y_test_numpy,
        num_classes,
        input_dim,
        rounds,
    )
    convergence = save_convergence_csv(
        round_frames,
        dataset_dir / "convergence.csv",
        rounds,
    )
    result_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"confusion_matrix", "experiment_settings"}
            }
            for row in result_rows
        ]
    )
    result_frame.to_csv(dataset_dir / "results.csv", index=False)
    with open(dataset_dir / "results.json", "w", encoding="utf-8") as output:
        json.dump(
            {
                "dataset": dataset,
                "class_values": [str(value) for value in class_values],
                "test_sample_count": int(len(y_test_numpy)),
                "feature_columns": feature_columns,
                "convergence_file": str(dataset_dir / "convergence.csv"),
                "experiment_settings": settings,
                "results": result_rows,
            },
            output,
            indent=2,
        )

    plot_test_accuracy(
        result_rows,
        dataset_dir / "test_accuracy_comparison.png",
        dataset,
    )
    plot_convergence(
        [
            convergence[convergence["method"] == method]
            for method in COMPARISON_METHODS
        ],
        dataset_dir / "convergence.png",
        dataset,
    )
    print(f"Saved {dataset} results to {dataset_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train FedAvg, FedProx, and paper-defined FedAWA baselines."
    )
    parser.add_argument(
        "--dataset",
        choices=("dataco", "ecommerce", "both"),
        default="both",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=(
            "Override the per-dataset paper budget (default: 100 DataCo "
            "rounds, 67 E-Commerce rounds)."
        ),
    )
    parser.add_argument("--tune-rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=module3.LOCAL_BATCH_SIZE)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=module3.ROUND_SAMPLE_FRACTION,
    )
    parser.add_argument(
        "--include-cl-dl",
        action="store_true",
        help="Also run Centralized and Distributed Learning (substantial CPU cost).",
    )
    args = parser.parse_args()
    if (args.rounds is not None and args.rounds < 1) or args.tune_rounds < 1:
        parser.error("--rounds and --tune-rounds must be positive.")
    if not 0 < args.sample_fraction <= 1:
        parser.error("--sample-fraction must be in (0, 1].")
    clean_comparison_outputs()
    combined_rows = []
    datasets = (
        ("dataco", "ecommerce")
        if args.dataset == "both"
        else (args.dataset,)
    )
    for ds_key in datasets:
        evaluate_dataset(ds_key, args, combined_rows)

    summary_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"confusion_matrix", "experiment_settings"}
            }
            for row in combined_rows
        ]
    )
    summary_frame.to_csv(OUTPUT_DIR / "comparison_results.csv", index=False)
    with open(OUTPUT_DIR / "comparison_results.json", "w", encoding="utf-8") as output:
        json.dump(combined_rows, output, indent=2)
    print(f"\nComparison results saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
