import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.model_selection import train_test_split

from fedprada_module3 import (
    DATASETS as FEDPRADA_DATASETS,
    DataCoResNetMLP,
    EcommerceResNetMLP,
    SEED as FEDPRADA_SEED,
    VALIDATION_SIZE as FEDPRADA_VALIDATION_SIZE,
    create_model,
    assign_risk_priority,
)

try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False

from benchmark_dashboard import render_research_performance_benchmark


# =========================================================
# PAGE
# =========================================================
st.set_page_config(
    page_title="Federated Learning Live Dashboard",
    page_icon="📊",
    layout="wide",
)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("Navigation")
nav_selection = st.sidebar.radio(
    "Select View",
    ["FedPRADA Live Dashboard", "Fed-FP3 Novelty", "Research Performance Benchmark"],
    index=0,
    help="Switch between the baseline FedPRADA live dashboard, Fed-FP3 Novelty evaluation, and Research Performance Benchmark."
)

if nav_selection == "FedPRADA Live Dashboard":
    st.title("Federated Learning - Live Training Dashboard")

    st.caption(
        "Live monitoring of the DataCo and E-Commerce local models. "
        "Training metrics are read directly from the files written "
        "by the two local training programs."
    )

    # Refresh every 2 seconds.
    if AUTO_REFRESH_AVAILABLE:
        st_autorefresh(
            interval=2000,
            key="training_dashboard_refresh",
        )
    else:
        st.warning(
            "Dashboard"
        )


# =========================================================
# CONFIGURATION
# =========================================================
BASE = "outputs"

DATASETS = {
    "DataCo": {
        "directory": os.path.join(
            BASE,
            "local_dataco",
        ),
        "global_directory": os.path.join(
            BASE,
            "fedprada_module3_dataco",
        ),
        "clients": [
            "africa",
            "europe",
            "latam",
            "pacific_asia",
            "usca",
        ],
    },

    "E-Commerce": {
        "directory": os.path.join(
            BASE,
            "local_ecommerce",
        ),
        "global_directory": os.path.join(
            BASE,
            "fedprada_module3_ecommerce",
        ),
        "clients": [
            "A",
            "B",
            "C",
            "D",
            "F",
        ],
    },
}


# =========================================================
# HELPERS
# =========================================================
def read_json(path):
    if not os.path.exists(path):
        return None

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)
    except Exception:
        return None


def read_client_status(dataset_dir, client):
    path = os.path.join(
        dataset_dir,
        f"client_{client}",
        "live_status.json",
    )

    return read_json(path)


def read_client_final(dataset_dir, client):
    path = os.path.join(
        dataset_dir,
        f"client_{client}",
        "final_result.json",
    )

    return read_json(path)


def read_summary(dataset_dir):
    return read_json(
        os.path.join(
            dataset_dir,
            "final_summary.json",
        )
    )


# =========================================================
# SHIPMENT PREDICTION HELPERS
# =========================================================
def prediction_feature_columns(dataset_name):
    config = FEDPRADA_DATASETS[
        "dataco" if dataset_name == "DataCo" else "ecommerce"
    ]
    expected_columns = None

    for client in config["clients"]:
        path = os.path.join(
            config["data_dir"],
            f"client_{client}.csv",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Processed {dataset_name} client data not found: {path}"
            )

        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(
                f"{dataset_name} client files have inconsistent feature schemas."
            )

    if expected_columns is None:
        raise ValueError(f"No processed clients are configured for {dataset_name}.")

    target = config["target"]
    if target not in expected_columns:
        raise ValueError(
            f"Expected target column '{target}' is missing from the {dataset_name} "
            "client files."
        )

    return [
        column
        for column in expected_columns
        if column not in config["drop"]
    ]


@st.cache_resource
def load_global_prediction_model(dataset_name, checkpoint_path, checkpoint_mtime):
    del checkpoint_mtime

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Final FedPRADA checkpoint not found: {checkpoint_path}. "
            "Run FedPRADA Module 3 first."
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    state_dict = checkpoint.get("model_state_dict")
    class_values = checkpoint.get("class_values")
    input_dim = checkpoint.get("input_dim")
    num_classes = checkpoint.get("num_classes")
    if (
        not isinstance(state_dict, dict)
        or class_values is None
        or input_dim is None
        or num_classes is None
    ):
        raise ValueError(
            f"The final checkpoint is missing model, feature, or class metadata: "
            f"{checkpoint_path}"
        )

    config = FEDPRADA_DATASETS[
        "dataco" if dataset_name == "DataCo" else "ecommerce"
    ]
    if dataset_name == "DataCo":
        model = DataCoResNetMLP(
            input_dim=int(input_dim),
            num_classes=int(num_classes),
            hidden_dim=config["hidden_dim"],
            num_blocks=config["num_blocks"],
            dropout=config["dropout"],
        )
    else:
        model = EcommerceResNetMLP(
            input_dim=int(input_dim),
            num_classes=int(num_classes),
            hidden_dim=config["hidden_dim"],
            num_blocks=config["num_blocks"],
            dropout=config["dropout"],
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, list(class_values), int(input_dim), int(num_classes)


@st.cache_data
def ecommerce_federated_scaler(data_signature):
    del data_signature

    config = FEDPRADA_DATASETS["ecommerce"]
    class_values = sorted(
        {
            value
            for client in config["clients"]
            for value in pd.read_csv(
                os.path.join(
                    config["data_dir"],
                    f"client_{client}.csv",
                ),
                usecols=[config["target"]],
            )[config["target"]].dropna().unique().tolist()
        }
    )
    class_to_index = {
        value: index
        for index, value in enumerate(class_values)
    }

    feature_sum = None
    feature_square_sum = None
    sample_count = 0

    for client in config["clients"]:
        path = os.path.join(
            config["data_dir"],
            f"client_{client}.csv",
        )
        frame = pd.read_csv(path)
        labels = frame[config["target"]].to_numpy()
        try:
            y = np.asarray(
                [class_to_index[value] for value in labels],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                f"Unrecognized E-Commerce target label in {path}: {exc}"
            ) from exc

        features = frame.drop(
            columns=[
                column
                for column in config["drop"]
                if column in frame.columns
            ]
        )
        features = (
            features.apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        X = features.to_numpy(dtype=np.float32)

        class_counts = np.bincount(y)
        stratify = (
            y
            if len(class_counts) > 1 and class_counts.min() >= 2
            else None
        )
        X_train, _, _, _ = train_test_split(
            X,
            y,
            test_size=FEDPRADA_VALIDATION_SIZE,
            random_state=FEDPRADA_SEED,
            stratify=stratify,
        )

        X_train_64 = X_train.astype(np.float64)
        current_sum = X_train_64.sum(axis=0)
        current_square_sum = np.square(X_train_64).sum(axis=0)
        if feature_sum is None:
            feature_sum = current_sum
            feature_square_sum = current_square_sum
        else:
            feature_sum += current_sum
            feature_square_sum += current_square_sum
        sample_count += len(X_train)

    if sample_count == 0 or feature_sum is None or feature_square_sum is None:
        raise ValueError("Could not derive E-Commerce training preprocessing.")

    mean = feature_sum / sample_count
    scale = np.sqrt(
        np.maximum(feature_square_sum / sample_count - np.square(mean), 0.0)
    )
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


@st.cache_data
def dataco_delivery_status_labels(source_signature):
    del source_signature

    target = FEDPRADA_DATASETS["dataco"]["target"]
    path = os.path.join("DataCo", "DataCoSupplyChainDataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DataCo label mapping source not found: {path}"
        )

    labels = pd.read_csv(
        path,
        usecols=[target],
        encoding="latin1",
    )[target]
    return sorted(labels.dropna().astype(str).unique().tolist())


def predict_shipment(dataset_name, checkpoint_path, feature_columns, values):
    if len(feature_columns) != len(values):
        raise ValueError("The supplied values do not match the model feature schema.")

    try:
        input_values = np.asarray(
            [float(values[column]) for column in feature_columns],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"All shipment features must be numeric: {exc}") from exc

    if not np.isfinite(input_values).all():
        raise ValueError("Shipment feature values must all be finite numbers.")

    model_path = (
        (os.path.getmtime(checkpoint_path), os.path.getsize(checkpoint_path))
        if os.path.exists(checkpoint_path)
        else (0, 0)
    )
    model, class_values, input_dim, num_classes = load_global_prediction_model(
        dataset_name,
        checkpoint_path,
        model_path,
    )
    if len(feature_columns) != input_dim:
        raise ValueError(
            f"Feature schema has {len(feature_columns)} columns, but the final "
            f"{dataset_name} checkpoint expects {input_dim}."
        )
    if len(class_values) != num_classes:
        raise ValueError("Checkpoint class mapping does not match its output layer.")

    if dataset_name == "DataCo":
        if num_classes != 4:
            raise ValueError(
                f"DataCo checkpoint must have four classes; found {num_classes}."
            )
        label_source = os.path.join(
            "DataCo",
            "DataCoSupplyChainDataset.csv",
        )
        label_signature = (
            (os.path.getmtime(label_source), os.path.getsize(label_source))
            if os.path.exists(label_source)
            else (0, 0)
        )
        labels = dataco_delivery_status_labels(label_signature)
        if len(labels) != num_classes:
            raise ValueError(
                "DataCo Delivery Status labels do not match the checkpoint classes."
            )
        # FedPRADA class values follow the sorted LabelEncoder class order.
        output_labels = labels
    else:
        if num_classes != 2:
            raise ValueError(
                f"E-Commerce checkpoint must have two classes; found {num_classes}."
            )
        label_names = {
            0: "On-time",
            1: "Not on-time",
        }
        output_labels = []
        for value in class_values:
            numeric_value = int(value)
            if numeric_value not in label_names or float(value) != numeric_value:
                raise ValueError(
                    f"Unsupported E-Commerce class value in checkpoint: {value}"
                )
            output_labels.append(label_names[numeric_value])

        config = FEDPRADA_DATASETS["ecommerce"]
        data_signature = tuple(
            (
                os.path.getmtime(
                    os.path.join(
                        config["data_dir"],
                        f"client_{client}.csv",
                    )
                ),
                os.path.getsize(
                    os.path.join(
                        config["data_dir"],
                        f"client_{client}.csv",
                    )
                ),
            )
            for client in config["clients"]
        )
        mean, scale = ecommerce_federated_scaler(data_signature)
        if len(mean) != input_dim or len(scale) != input_dim:
            raise ValueError(
                "E-Commerce preprocessing dimensions do not match the checkpoint."
            )
        input_values = (input_values - mean) / scale

    with torch.inference_mode():
        logits = model(torch.from_numpy(input_values.reshape(1, -1)))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    if (
        len(probabilities) != num_classes
        or not np.isfinite(probabilities).all()
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)
    ):
        raise ValueError("The model returned invalid class probabilities.")

    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_label": output_labels[predicted_index],
        "labels": output_labels,
        "probabilities": probabilities.tolist(),
    }


# =========================================================
# GLOBAL MODEL (FedPRADA Module 3) HELPERS
# =========================================================
def read_global_summary(global_dir):
    """
    Reads module3_summary.json produced by FedPRADA Module 3.
    """
    return read_json(
        os.path.join(
            global_dir,
            "module3_summary.json",
        )
    )


def read_global_accuracy(global_dir):
    """Read the final global-model accuracy saved by FedPRADA Module 3."""
    summary = read_global_summary(global_dir)

    if summary:
        # Support the current Module 3 summary format and the older format.
        final = summary.get("final") or {}
        for value in (
            final.get("accuracy"),
            summary.get("final_global_val_acc"),
            summary.get("best_global_val_acc"),
        ):
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    # Fallback: use the most recent row in global_metrics.csv.
    metrics_path = os.path.join(global_dir, "global_metrics.csv")
    if os.path.exists(metrics_path):
        try:
            metrics = pd.read_csv(metrics_path)
            for column in ("global_val_acc", "accuracy", "val_accuracy"):
                if column in metrics.columns and not metrics[column].dropna().empty:
                    return float(metrics[column].dropna().iloc[-1])
        except Exception:
            pass

    return None


def read_client_history(dataset_dir, client):
    path = os.path.join(
        dataset_dir,
        f"client_{client}",
        "training_metrics.csv",
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def status_text(status):
    if not status:
        return "Not started"
    value = status.get(
        "status",
        "unknown",
    )
    if value == "completed":
        return "Completed"
    if value == "training":
        return "Training"
    if value == "starting":
        return "Starting"
    return value.capitalize()


def fmt_percent(value):
    if value is None:
        return "-"

    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def fmt_number(value):
    if value is None:
        return "-"

    try:
        return f"{float(value):.4f}"
    except Exception:
        return "-"


def dataset_results(dataset_name, config):
    dataset_dir = config["directory"]
    completed = []
    current = []
    for client in config["clients"]:
        status = read_client_status(
            dataset_dir,
            client,
        )
        if status:
            if status.get("val_accuracy") is not None:
                current.append(
                    float(status["val_accuracy"])
                )
        final = read_client_final(
            dataset_dir,
            client,
        )
        if final and final.get("final_accuracy") is not None:
            completed.append(
                float(final["final_accuracy"])
            )
    return current, completed


def progress_percent(status):
    if not status:
        return 0.0

    total = max(
        int(status.get("total_epochs", 1)),
        1,
    )

    current = int(
        status.get("current_epoch", 0)
    )

    return min(
        current / total,
        1.0,
    )



# =========================================================
# =========================================================
# FED-FP3 REAL-TIME INFERENCE HELPERS & PREPROCESSING SCALERS
# =========================================================
ECOMM_RAW_NUM_MEANS = {
    'Customer_care_calls': 4.054459496317847,
    'Customer_rating': 2.9905445949631786,
    'Cost_of_the_Product': 210.19683607600692,
    'Prior_purchases': 3.5675970542776616,
    'Discount_offered': 13.37321574688608,
    'Weight_in_gms': 3634.016728793527,
}
ECOMM_RAW_NUM_STDS = {
    'Customer_care_calls': 1.1414377553154325,
    'Customer_rating': 1.4135389093311594,
    'Cost_of_the_Product': 48.061086809072016,
    'Prior_purchases': 1.5227908136052688,
    'Discount_offered': 16.204790382056665,
    'Weight_in_gms': 1635.3029076240832,
}
ECOMM_FED_MEANS = np.array([
    1.5260884392406502, 0.0009879051848807308, 0.003392004310014208,
    0.0034518828074559713, -0.002250766546140547, 1.347732181425486,
    0.49698760941229964, 0.002467991491383526, 0.0027586191728135936
], dtype=np.float32)
ECOMM_FED_STDS = np.array([
    0.7499204385971818, 1.0026306354456698, 1.0020911706502251,
    0.999158060826681, 0.9989634807465694, 0.630879732517342,
    0.49999092542059914, 1.0040214227400783, 1.0008695682855901
], dtype=np.float32)

DATACO_RAW_STATS = {
    'Days for shipment (scheduled)': (3.1815, 1.1950),
    'Benefit per order': (21.1951, 106.3162),
    'Sales per customer': (178.3665, 117.8950),
    'Order Item Quantity': (2.4437, 1.5555),
    'Order Item Product Price': (120.0350, 128.9237),
    'Order Item Discount Rate': (0.1034, 0.0705),
    'Product Price': (120.0350, 128.9237),
}
DATACO_CATEGORICAL_MAPS = {
    'Shipping Mode': {'First Class': 0, 'Same Day': 1, 'Second Class': 2, 'Standard Class': 3},
    'Type': {'CASH': 0, 'DEBIT': 1, 'PAYMENT': 2, 'TRANSFER': 3},
    'Customer Segment': {'Consumer': 0, 'Corporate': 1, 'Home Office': 2},
}
DATACO_CATEGORICAL_STATS = {
    'Shipping Mode': (2.4175, 0.9822),
    'Type': (1.4400, 1.0003),
    'Customer Segment': (0.6798, 0.7254),
}


def ecommerce_federated_scaler(_=None):
    return ECOMM_FED_MEANS, ECOMM_FED_STDS


@st.cache_resource
def load_fed_fp3_prediction_model(dataset_key):
    """
    Loads and caches the trained Fed-FP3 global model checkpoint for real-time inference.
    """
    ckpt_path = os.path.join(BASE, "fed_fp3", dataset_key, "global_model_final.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(BASE, "fed_fp3", dataset_key, "global_model_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Trained Fed-FP3 checkpoint not found at {ckpt_path}. Run Fed-FP3 Module 3 first.")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint["num_classes"])
    model = create_model(dataset_key, input_dim, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, input_dim, num_classes


@st.cache_data
def get_dataco_feature_metadata():
    """
    Extracts the exact 53 input feature names, training medians, and value ranges from DataCo client data.
    """
    cfg = FEDPRADA_DATASETS["dataco"]
    drop_cols = cfg["drop"]
    sample_frames = []
    for cl in cfg["clients"]:
        p = os.path.join(cfg["data_dir"], f"client_{cl}.csv")
        if os.path.exists(p):
            sample_frames.append(pd.read_csv(p, nrows=300))
    if not sample_frames:
        raise FileNotFoundError("DataCo client training data files not found in processed/clients/dataco/")

    combined = pd.concat(sample_frames, ignore_index=True)
    features = [c for c in combined.columns if c not in drop_cols]
    medians = combined[features].median().to_dict()
    mins = combined[features].min().to_dict()
    maxs = combined[features].max().to_dict()
    return features, medians, mins, maxs


def run_fed_fp3_inference(dataset_key, raw_inputs, shipment_id=None):
    """
    Executes real-time inference on new shipment input using the trained Fed-FP3 model.
    Validates input schemas, applies exact training-time encoders and scalers,
    computes calibrated softmax probabilities, and assigns risk priority.
    """
    model, input_dim, num_classes = load_fed_fp3_prediction_model(dataset_key)

    if dataset_key == "ecommerce":
        mode_mapping = {"Flight": 0, "Road": 1, "Ship": 2}
        imp_mapping = {"high": 0, "low": 1, "medium": 2}
        gender_mapping = {"F": 0, "M": 1}

        mode_val = raw_inputs.get("Mode_of_Shipment")
        if mode_val not in mode_mapping:
            raise ValueError(f"Invalid Mode of Shipment: '{mode_val}'. Must be one of {list(mode_mapping.keys())}")

        imp_val = raw_inputs.get("Product_importance")
        if imp_val not in imp_mapping:
            raise ValueError(f"Invalid Product Importance: '{imp_val}'. Must be one of {list(imp_mapping.keys())}")

        gen_val = raw_inputs.get("Gender")
        if gen_val not in gender_mapping:
            raise ValueError(f"Invalid Gender: '{gen_val}'. Must be one of {list(gender_mapping.keys())}")

        try:
            calls = float(raw_inputs["Customer_care_calls"])
            rating = float(raw_inputs["Customer_rating"])
            cost = float(raw_inputs["Cost_of_the_Product"])
            prior = float(raw_inputs["Prior_purchases"])
            discount = float(raw_inputs["Discount_offered"])
            weight = float(raw_inputs["Weight_in_gms"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Missing or invalid numerical feature: {e}")

        if not (1 <= calls <= 25):
            raise ValueError(f"Customer care calls ({calls}) is outside realistic range [1, 25]")
        if not (1 <= rating <= 5):
            raise ValueError(f"Customer rating ({rating}) must be between 1 and 5")
        if cost <= 0:
            raise ValueError(f"Cost of the product (${cost}) must be strictly positive")
        if prior < 0:
            raise ValueError(f"Prior purchases ({prior}) cannot be negative")
        if not (0 <= discount <= 100):
            raise ValueError(f"Discount offered ({discount}%) must be between 0% and 100%")
        if weight <= 0:
            raise ValueError(f"Weight in grams ({weight}g) must be strictly positive")

        # Step 1: Standardize raw numerical features
        s_calls = (calls - ECOMM_RAW_NUM_MEANS['Customer_care_calls']) / ECOMM_RAW_NUM_STDS['Customer_care_calls']
        s_rating = (rating - ECOMM_RAW_NUM_MEANS['Customer_rating']) / ECOMM_RAW_NUM_STDS['Customer_rating']
        s_cost = (cost - ECOMM_RAW_NUM_MEANS['Cost_of_the_Product']) / ECOMM_RAW_NUM_STDS['Cost_of_the_Product']
        s_prior = (prior - ECOMM_RAW_NUM_MEANS['Prior_purchases']) / ECOMM_RAW_NUM_STDS['Prior_purchases']
        s_discount = (discount - ECOMM_RAW_NUM_MEANS['Discount_offered']) / ECOMM_RAW_NUM_STDS['Discount_offered']
        s_weight = (weight - ECOMM_RAW_NUM_MEANS['Weight_in_gms']) / ECOMM_RAW_NUM_STDS['Weight_in_gms']

        unscaled_vec = np.array([
            mode_mapping[mode_val],
            s_calls,
            s_rating,
            s_cost,
            s_prior,
            imp_mapping[imp_val],
            gender_mapping[gen_val],
            s_discount,
            s_weight
        ], dtype=np.float32)

        # Step 2: Federated standardization across 9 dimensions
        scaled_vec = (unscaled_vec - ECOMM_FED_MEANS) / ECOMM_FED_STDS

        with torch.no_grad():
            logits = model(torch.tensor(scaled_vec, dtype=torch.float32).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).numpy()[0]

        class_names = ["On-time", "Not on-time"]
        pred_idx = int(probs.argmax())
        conf = float(probs[pred_idx])
        pred_label = class_names[pred_idx]

        prio, action, risk_score = assign_risk_priority("ecommerce", pred_label, conf)

        if prio == "CRITICAL":
            explanation = (f"Predicted 'Not on-time' with critical confidence ({conf:.2%} >= 75%). "
                           f"High discount rate ({discount:.0f}%) or parcel weight ({weight:,.0f}g) significantly "
                           f"increases fulfillment bottleneck probability, triggering immediate emergency rerouting.")
        elif prio == "HIGH":
            explanation = (f"Predicted 'Not on-time' with high confidence ({conf:.2%} in [60%, 75%)). "
                           f"Elevated shipment lead time requires expedited fulfillment center handling.")
        elif prio == "MEDIUM":
            explanation = (f"Marginal delivery delay uncertainty (confidence {conf:.2%} < 60%). "
                           f"Requires carrier tracking check to verify truck/flight handoff.")
        elif prio == "LOW-MEDIUM":
            explanation = (f"Predicted 'On-time' but with moderate confidence ({conf:.2%} < 70%). "
                           f"Monitor initial milestone scans.")
        else:
            explanation = (f"High confidence ({conf:.2%} >= 70%) for on-time delivery. "
                           f"Proceed with standard logistics handling.")

        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        input_summary = (f"Mode: {mode_val} | Importance: {imp_val} | Cost: ${cost:.0f} | "
                         f"Discount: {discount:.0f}% | Weight: {weight:,.0f}g | Calls: {int(calls)}")

    else: # dataco
        features, medians, mins, maxs = get_dataco_feature_metadata()

        vec_vals = []
        for feat in features:
            if feat in raw_inputs and raw_inputs[feat] is not None:
                val_raw = raw_inputs[feat]
                if feat in DATACO_CATEGORICAL_MAPS and isinstance(val_raw, str):
                    cat_map = DATACO_CATEGORICAL_MAPS[feat]
                    enc_val = cat_map.get(val_raw, 0)
                    mean_val, std_val = DATACO_CATEGORICAL_STATS[feat]
                    val = (enc_val - mean_val) / std_val
                elif feat in DATACO_RAW_STATS and not raw_inputs.get("_is_scaled_input", False):
                    mean_val, std_val = DATACO_RAW_STATS[feat]
                    try:
                        val = (float(val_raw) - mean_val) / std_val
                    except Exception:
                        val = medians.get(feat, 0.0)
                else:
                    try:
                        val = float(val_raw)
                        if not np.isfinite(val):
                            raise ValueError(f"Feature '{feat}' must be finite")
                    except (ValueError, TypeError):
                        val = medians.get(feat, 0.0)
                vec_vals.append(val)
            else:
                vec_vals.append(medians.get(feat, 0.0))

        scaled_vec = np.array(vec_vals, dtype=np.float32)

        with torch.no_grad():
            logits = model(torch.tensor(scaled_vec, dtype=torch.float32).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).numpy()[0]

        class_names = ["Advance shipping", "Late delivery", "Shipping canceled", "Shipping on time"]
        pred_idx = int(probs.argmax())
        conf = float(probs[pred_idx])
        pred_label = class_names[pred_idx]

        prio, action, risk_score = assign_risk_priority("dataco", pred_label, conf)

        if prio == "CRITICAL":
            explanation = (f"High-confidence delay predicted: '{pred_label}' at {conf:.2%} confidence (>= 70%). "
                           f"Tight scheduled transit window or origin processing latency necessitates immediate carrier tier escalation.")
        elif prio == "HIGH":
            explanation = (f"Predicted '{pred_label}' with confidence {conf:.2%}. "
                           f"Fulfillment cancellation risk or delay requires warehouse inventory check and dispatch monitoring.")
        elif prio == "MEDIUM":
            explanation = (f"Moderate delivery uncertainty ({conf:.2%} confidence). "
                           f"Audit local logistics hub handover logs.")
        elif prio == "LOW-MEDIUM":
            explanation = (f"Predicted on-track status but with marginal confidence ({conf:.2%} < 70%). "
                           f"Routine telemetry check recommended.")
        else:
            explanation = (f"Confirmed on-time or early arrival prediction ({conf:.2%} >= 70%). "
                           f"Standard supply chain dispatch schedule progressing normally.")

        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        sched_days = raw_inputs.get("Days for shipment (scheduled)", 3)
        sales = raw_inputs.get("Sales per customer", 150)
        qty = raw_inputs.get("Order Item Quantity", 2)
        smode = raw_inputs.get("Shipping Mode", "Standard Class")
        input_summary = f"Sched Days: {sched_days} | Sales: ${float(sales):.2f} | Qty: {qty} | Mode: {smode}"

    sid = shipment_id or f"{'ECOMM' if dataset_key == 'ecommerce' else 'DC'}-PRED-{datetime.now().strftime('%H%M%S')}"

    return {
        "Shipment_ID": sid,
        "Dataset": "E-Commerce Shipping" if dataset_key == "ecommerce" else "DataCo Supply Chain",
        "Dataset_Key": dataset_key,
        "Predicted_Status": pred_label,
        "Confidence": conf,
        "Confidence_Percentage": f"{conf * 100:.2f}%",
        "Risk_Priority": prio,
        "Recommended_Action": action,
        "Composite_Risk_Score": risk_score,
        "Class_Probabilities": prob_dict,
        "Explanation": explanation,
        "Input_Summary": input_summary,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def run_fed_fp3_batch_inference(dataset_key, batch_df):
    """
    Executes vectorized batch inference for uploaded CSV files across either dataset.
    Returns prioritized dataframe sorted by Composite_Risk_Score descending.
    """
    model, input_dim, num_classes = load_fed_fp3_prediction_model(dataset_key)
    df_clean = batch_df.copy()
    rows = []

    if dataset_key == "ecommerce":
        mode_mapping = {"Flight": 0, "Road": 1, "Ship": 2}
        imp_mapping = {"high": 0, "low": 1, "medium": 2}
        gender_mapping = {"F": 0, "M": 1}
        class_names = ["On-time", "Not on-time"]

        # Check if already preprocessed numeric features
        is_raw = any(isinstance(v, str) for v in df_clean.get("Mode_of_Shipment", [0]))

        X_list = []
        for _, row in df_clean.iterrows():
            if is_raw:
                m_val = mode_mapping.get(row.get("Mode_of_Shipment", "Road"), 1)
                imp_val = imp_mapping.get(row.get("Product_importance", "medium"), 2)
                gen_val = gender_mapping.get(row.get("Gender", "M"), 1)
                calls = float(row.get("Customer_care_calls", 4))
                rating = float(row.get("Customer_rating", 3))
                cost = float(row.get("Cost_of_the_Product", 210))
                prior = float(row.get("Prior_purchases", 3))
                disc = float(row.get("Discount_offered", 10))
                weight = float(row.get("Weight_in_gms", 3500))

                s_calls = (calls - ECOMM_RAW_NUM_MEANS['Customer_care_calls']) / ECOMM_RAW_NUM_STDS['Customer_care_calls']
                s_rating = (rating - ECOMM_RAW_NUM_MEANS['Customer_rating']) / ECOMM_RAW_NUM_STDS['Customer_rating']
                s_cost = (cost - ECOMM_RAW_NUM_MEANS['Cost_of_the_Product']) / ECOMM_RAW_NUM_STDS['Cost_of_the_Product']
                s_prior = (prior - ECOMM_RAW_NUM_MEANS['Prior_purchases']) / ECOMM_RAW_NUM_STDS['Prior_purchases']
                s_disc = (disc - ECOMM_RAW_NUM_MEANS['Discount_offered']) / ECOMM_RAW_NUM_STDS['Discount_offered']
                s_weight = (weight - ECOMM_RAW_NUM_MEANS['Weight_in_gms']) / ECOMM_RAW_NUM_STDS['Weight_in_gms']

                vec = np.array([m_val, s_calls, s_rating, s_cost, s_prior, imp_val, gen_val, s_disc, s_weight], dtype=np.float32)
            else:
                drop_t = ["Reached.on.Time_Y.N", "ID", "Warehouse_block"]
                vec = row.drop([c for c in drop_t if c in row.index]).values.astype(np.float32)[:9]

            scaled = (vec - ECOMM_FED_MEANS) / ECOMM_FED_STDS
            X_list.append(scaled)

        X_batch = torch.tensor(np.array(X_list), dtype=torch.float32)
        with torch.no_grad():
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).numpy()

        for idx, row in df_clean.iterrows():
            pred_idx = int(probs[idx].argmax())
            conf = float(probs[idx, pred_idx])
            pred_label = class_names[pred_idx]
            prio, action, risk_score = assign_risk_priority("ecommerce", pred_label, conf)
            sid = str(row.get("ID", row.get("Shipment_ID", f"ECOMM-BATCH-{idx+1}")))
            rows.append({
                "Shipment_ID": sid,
                "Dataset": "E-Commerce Shipping",
                "Predicted_Status": pred_label,
                "Confidence": conf,
                "Confidence_Percentage": f"{conf * 100:.2f}%",
                "Risk_Priority": prio,
                "Recommended_Action": action,
                "Composite_Risk_Score": risk_score,
                "Prob_On-time": float(probs[idx, 0]),
                "Prob_Not on-time": float(probs[idx, 1]),
            })

    else: # dataco
        features, medians, _, _ = get_dataco_feature_metadata()
        class_names = ["Advance shipping", "Late delivery", "Shipping canceled", "Shipping on time"]
        drop_cols = ["Delivery Status", "Days for shipping (real)", "Late_delivery_risk", "Market", "Order Id"]

        X_list = []
        for _, row in df_clean.iterrows():
            vec = []
            for feat in features:
                if feat in row.index:
                    val = row[feat]
                    try:
                        vec.append(float(val))
                    except Exception:
                        vec.append(medians.get(feat, 0.0))
                else:
                    vec.append(medians.get(feat, 0.0))
            X_list.append(vec)

        X_batch = torch.tensor(np.array(X_list), dtype=torch.float32)
        with torch.no_grad():
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).numpy()

        for idx, row in df_clean.iterrows():
            pred_idx = int(probs[idx].argmax())
            conf = float(probs[idx, pred_idx])
            pred_label = class_names[pred_idx]
            prio, action, risk_score = assign_risk_priority("dataco", pred_label, conf)
            sid = str(row.get("Order Id", row.get("Shipment_ID", f"DC-BATCH-{idx+1}")))
            res_row = {
                "Shipment_ID": sid,
                "Dataset": "DataCo Supply Chain",
                "Predicted_Status": pred_label,
                "Confidence": conf,
                "Confidence_Percentage": f"{conf * 100:.2f}%",
                "Risk_Priority": prio,
                "Recommended_Action": action,
                "Composite_Risk_Score": risk_score,
            }
            for c_i, c_name in enumerate(class_names):
                res_row[f"Prob_{c_name}"] = float(probs[idx, c_i])
            rows.append(res_row)

    res_df = pd.DataFrame(rows).sort_values("Composite_Risk_Score", ascending=False).reset_index(drop=True)
    res_df.insert(0, "Rank", range(1, len(res_df) + 1))
    return res_df


# =========================================================
# FED-FP3 NOVELTY DASHBOARD RENDERER
# =========================================================
def render_fed_fp3_dashboard():
    st.title("Fed-FP3: Federated Feature-Pertinence & Prediction-based Prioritization")
    st.caption(
        "Proposed Novelty Framework: Modulates federated aggregation weights via client "
        "feature distribution divergence (FD_m) and feature-pertinence scores (FP_m), "
        "and produces calibrated shipment delivery risk prioritization rankings."
    )

    # -----------------------------------------------------
    # SECTION 1: DATASET SELECTOR
    # -----------------------------------------------------
    st.header("1. Target Dataset Selection")
    col_sel1, col_sel2 = st.columns([2, 3])
    with col_sel1:
        selected_ds = st.radio(
            "Select Dataset",
            ["DataCo Supply Chain", "E-Commerce Shipping"],
            horizontal=True,
            key="fed_fp3_dataset_radio"
        )
    ds_key = "dataco" if selected_ds == "DataCo Supply Chain" else "ecommerce"
    fp3_dir = os.path.join(BASE, "fed_fp3", ds_key)
    fedprada_dir = DATASETS["DataCo"]["global_directory"] if ds_key == "dataco" else DATASETS["E-Commerce"]["global_directory"]

    with col_sel2:
        st.info(
            f"Active Dataset: **{selected_ds}** | Fed-FP3 Directory: `{fp3_dir}` | "
            f"Baseline Directory: `{fedprada_dir}`"
        )

    st.divider()

    # Check if Fed-FP3 outputs exist
    summary_path = os.path.join(fp3_dir, "fed_fp3_summary.json")
    eval_metrics_path = os.path.join(fp3_dir, "evaluation_metrics.json")
    prio_path = os.path.join(fp3_dir, "shipment_risk_prioritization.csv")
    fusion_log_path = os.path.join(fp3_dir, "fusion_log.csv")
    global_metrics_path = os.path.join(fp3_dir, "global_metrics.csv")

    fp3_summary = read_json(summary_path)
    eval_metrics = read_json(eval_metrics_path)

    if not fp3_summary or not eval_metrics:
        st.warning(
            f"Fed-FP3 outputs for {selected_ds} are not fully generated yet or training is currently running. "
            f"Expected summary at `{summary_path}`."
        )
        return

    # -----------------------------------------------------
    # SECTION 2: FED-FP3 PERFORMANCE METRICS
    # -----------------------------------------------------
    st.header(f"2. Fed-FP3 Performance Metrics ({selected_ds})")
    st.caption("Comprehensive multi-class / binary classification evaluation on holdout validation data.")

    m1, m2, m3, m4, m5 = st.columns(5)
    acc = eval_metrics.get("accuracy", 0.0)
    auc_val = eval_metrics.get("auc", 0.0)
    prec_m = eval_metrics.get("precision_macro", 0.0)
    prec_w = eval_metrics.get("precision_weighted", 0.0)
    rec_m = eval_metrics.get("recall_macro_sensitivity", 0.0)
    rec_w = eval_metrics.get("recall_weighted", 0.0)
    f1_m = eval_metrics.get("f1_macro", 0.0)
    f1_w = eval_metrics.get("f1_weighted", 0.0)
    spec_m = eval_metrics.get("specificity_macro", 0.0)
    val_loss = eval_metrics.get("validation_loss", 0.0)

    m1.metric("Validation Accuracy", fmt_percent(acc))
    m1.metric("ROC-AUC Score", fmt_number(auc_val))

    m2.metric("Precision (Macro)", fmt_percent(prec_m))
    m2.metric("Precision (Weighted)", fmt_percent(prec_w))

    m3.metric("Recall / Sensitivity (Macro)", fmt_percent(rec_m))
    m3.metric("Recall (Weighted)", fmt_percent(rec_w))

    m4.metric("F1-Score (Macro)", fmt_percent(f1_m))
    m4.metric("F1-Score (Weighted)", fmt_percent(f1_w))

    m5.metric("Specificity (Macro)", fmt_percent(spec_m))
    m5.metric("Validation Loss", fmt_number(val_loss))

    # Class-level specificity breakdown
    class_labels = eval_metrics.get("class_labels", [])
    class_specs = eval_metrics.get("class_specificity", [])
    if class_labels and class_specs and len(class_labels) == len(class_specs):
        with st.expander("Per-Class Specificity & Sensitivity Breakdown", expanded=False):
            spec_rows = [
                {"Class Label": lbl, "Specificity (%)": f"{sp * 100:.2f}%", "Index": i}
                for i, (lbl, sp) in enumerate(zip(class_labels, class_specs))
            ]
            st.dataframe(pd.DataFrame(spec_rows), use_container_width=True, hide_index=True)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm_csv_path = os.path.join(fp3_dir, "confusion_matrix.csv")
    cm_png_path = os.path.join(fp3_dir, "confusion_matrix.png")
    cm_col1, cm_col2 = st.columns([1, 1])

    with cm_col1:
        if os.path.exists(cm_csv_path):
            try:
                cm_df = pd.read_csv(cm_csv_path, index_col=0)
                st.write("**Confusion Matrix Contingency Table**")
                st.dataframe(cm_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load confusion matrix CSV: {e}")

    with cm_col2:
        if os.path.exists(cm_png_path):
            st.image(cm_png_path, caption=f"{selected_ds} - Fed-FP3 Confusion Matrix Heatmap", use_container_width=True)

    st.divider()

    # -----------------------------------------------------
    # SECTION 3: BASELINE COMPARISON TABLE
    # -----------------------------------------------------
    st.header(f"3. Baseline Comparison Table (Existing FedPRADA vs. Fed-FP3)")
    st.caption("Side-by-side empirical comparison between the baseline FedPRADA model and proposed Fed-FP3 novelty.")

    fedprada_summary = read_global_summary(fedprada_dir) or {}
    fedprada_best_acc = fedprada_summary.get("best_global_val_acc")
    fedprada_final_acc = fedprada_summary.get("final_global_val_acc")
    fedprada_loss = fedprada_summary.get("final_global_val_loss")
    fedprada_f1 = fedprada_summary.get("final_global_val_f1")

    fp3_best_acc = fp3_summary.get("best_global_val_acc")
    fp3_final_acc = fp3_summary.get("final_global_val_acc")
    fp3_loss = fp3_summary.get("final_global_val_loss")
    fp3_f1 = fp3_summary.get("final_global_val_f1")

    comp_rows = [
        {
            "Metric": "Best Validation Accuracy",
            "Existing FedPRADA": fmt_percent(fedprada_best_acc),
            "Proposed Fed-FP3": fmt_percent(fp3_best_acc),
            "Difference (Delta)": f"{(float(fp3_best_acc) - float(fedprada_best_acc))*100:+.2f} pp" if fp3_best_acc and fedprada_best_acc else "-",
            "Interpretation": "Peak generalization reached across communication rounds"
        },
        {
            "Metric": "Final Validation Accuracy",
            "Existing FedPRADA": fmt_percent(fedprada_final_acc),
            "Proposed Fed-FP3": fmt_percent(fp3_final_acc),
            "Difference (Delta)": f"{(float(fp3_final_acc) - float(fedprada_final_acc))*100:+.2f} pp" if fp3_final_acc and fedprada_final_acc else "-",
            "Interpretation": "Accuracy at final communication round (round 100)"
        },
        {
            "Metric": "Final Validation Loss",
            "Existing FedPRADA": fmt_number(fedprada_loss),
            "Proposed Fed-FP3": fmt_number(fp3_loss),
            "Difference (Delta)": f"{float(fp3_loss) - float(fedprada_loss):+.4f}" if fp3_loss and fedprada_loss else "-",
            "Interpretation": "Cross-entropy loss on global validation set"
        },
        {
            "Metric": "Final Macro F1-Score",
            "Existing FedPRADA": fmt_percent(fedprada_f1),
            "Proposed Fed-FP3": fmt_percent(fp3_f1),
            "Difference (Delta)": f"{(float(fp3_f1) - float(fedprada_f1))*100:+.2f} pp" if fp3_f1 and fedprada_f1 else "-",
            "Interpretation": "Harmonic mean of precision & recall across all classes"
        },
        {
            "Metric": "Feature Distribution Divergence (FD_m)",
            "Existing FedPRADA": "Not considered (Assumes uniform feature alignment)",
            "Proposed Fed-FP3": "Explicitly quantified via continuous moment discrepancy",
            "Difference (Delta)": "Novel Capability",
            "Interpretation": "Measures covariate shift between client and global distribution"
        },
        {
            "Metric": "Client Feature-Pertinence Weighting (FP_m)",
            "Existing FedPRADA": "Disabled (Weight depends only on n_m, D_m, S_m)",
            "Proposed Fed-FP3": "Enabled (w_m scaled by FP_m^alpha)",
            "Difference (Delta)": "Novel Capability",
            "Interpretation": "Prevents distorted feature distributions from skewing global model"
        },
        {
            "Metric": "Delivery Risk Prioritization Output",
            "Existing FedPRADA": "Standard class probabilities only",
            "Proposed Fed-FP3": "Automated 5-tier calibrated shipment risk ranking",
            "Difference (Delta)": "Novel Capability",
            "Interpretation": "Actionable operational triage for delayed and critical shipments"
        },
    ]
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------------------------------
    # SECTION 4: CONVERGENCE GRAPH
    # -----------------------------------------------------
    st.header(f"4. Convergence & Learning Trajectory ({selected_ds})")
    st.caption("Round-by-round trajectory of global model validation accuracy and loss.")

    if os.path.exists(global_metrics_path):
        try:
            fp3_metrics_df = pd.read_csv(global_metrics_path)
            fedprada_metrics_csv = os.path.join(fedprada_dir, "global_metrics.csv")
            fedprada_metrics_df = pd.read_csv(fedprada_metrics_csv) if os.path.exists(fedprada_metrics_csv) else None

            chart_c1, chart_c2 = st.columns(2)
            with chart_c1:
                st.write("**Validation Accuracy Progression**")
                acc_dict = {"Fed-FP3 Accuracy": fp3_metrics_df["global_val_acc"].values}
                if fedprada_metrics_df is not None and "global_val_acc" in fedprada_metrics_df.columns:
                    min_len = min(len(fp3_metrics_df), len(fedprada_metrics_df))
                    acc_dict = {
                        "Fed-FP3 Accuracy": fp3_metrics_df["global_val_acc"].iloc[:min_len].values,
                        "FedPRADA Accuracy": fedprada_metrics_df["global_val_acc"].iloc[:min_len].values,
                    }
                    rounds_idx = fp3_metrics_df["round"].iloc[:min_len].values
                else:
                    rounds_idx = fp3_metrics_df["round"].values

                acc_plot_df = pd.DataFrame(acc_dict, index=rounds_idx)
                st.line_chart(acc_plot_df, use_container_width=True)

            with chart_c2:
                st.write("**Validation Loss Progression**")
                loss_dict = {"Fed-FP3 Loss": fp3_metrics_df["global_val_loss"].values}
                if fedprada_metrics_df is not None and "global_val_loss" in fedprada_metrics_df.columns:
                    min_len = min(len(fp3_metrics_df), len(fedprada_metrics_df))
                    loss_dict = {
                        "Fed-FP3 Loss": fp3_metrics_df["global_val_loss"].iloc[:min_len].values,
                        "FedPRADA Loss": fedprada_metrics_df["global_val_loss"].iloc[:min_len].values,
                    }
                    rounds_idx = fp3_metrics_df["round"].iloc[:min_len].values
                else:
                    rounds_idx = fp3_metrics_df["round"].values

                loss_plot_df = pd.DataFrame(loss_dict, index=rounds_idx)
                st.line_chart(loss_plot_df, use_container_width=True)

        except Exception as e:
            st.warning(f"Error plotting convergence: {e}")

    perf_png = os.path.join(fp3_dir, "global_model_performance.png")
    if os.path.exists(perf_png):
        with st.expander(f"Detailed Global Performance Plot ({selected_ds})", expanded=False):
            st.image(perf_png, caption="Fed-FP3 Global Model Performance & Learning Curves", use_container_width=True)

    st.divider()

    # -----------------------------------------------------
    # SECTION 5: FEATURE-PERTINENCE ANALYSIS
    # -----------------------------------------------------
    st.header(f"5. Feature-Pertinence Analysis ($FP_m$) & Aggregation Weights")
    st.caption("How client feature distribution divergence modulates fusion weights relative to FedPRADA and FedAvg.")

    if os.path.exists(fusion_log_path):
        try:
            flog = pd.read_csv(fusion_log_path)
            max_round = flog["round"].max()
            final_flog = flog[flog["round"] == max_round].copy()

            st.subheader(f"Final Communication Round (Round {max_round}) Client Fusion Decomposition")
            display_cols = [
                "client", "n_m", "KL_D_m", "stability_S_m",
                "feature_divergence_FD_m", "feature_pertinence_FP_m",
                "weight_fedavg", "weight_fedprada", "weight_fed_fp3"
            ]
            available_cols = [c for c in display_cols if c in final_flog.columns]
            table_display = final_flog[available_cols].rename(columns={
                "client": "Client",
                "n_m": "Samples (n_m)",
                "KL_D_m": "Label Divergence (D_m)",
                "stability_S_m": "Stability (S_m)",
                "feature_divergence_FD_m": "Feature Divergence (FD_m)",
                "feature_pertinence_FP_m": "Feature Pertinence (FP_m)",
                "weight_fedavg": "FedAvg Weight",
                "weight_fedprada": "FedPRADA Weight",
                "weight_fed_fp3": "Fed-FP3 Weight",
            })
            st.dataframe(table_display.style.format({
                "Label Divergence (D_m)": "{:.6f}",
                "Stability (S_m)": "{:.4f}",
                "Feature Divergence (FD_m)": "{:.6f}",
                "Feature Pertinence (FP_m)": "{:.4f}",
                "FedAvg Weight": "{:.4f}",
                "FedPRADA Weight": "{:.4f}",
                "Fed-FP3 Weight": "{:.4f}",
            }), use_container_width=True, hide_index=True)

            # Weight comparison bar chart
            st.write("**Aggregation Weight Distribution Comparison**")
            weight_chart_df = final_flog[["client", "weight_fedavg", "weight_fedprada", "weight_fed_fp3"]].copy()
            weight_chart_df.columns = ["Client", "FedAvg", "FedPRADA", "Fed-FP3"]
            st.bar_chart(weight_chart_df.set_index("Client"), use_container_width=True)

        except Exception as e:
            st.warning(f"Error parsing fusion log: {e}")

    fp_col1, fp_col2 = st.columns(2)
    with fp_col1:
        fp_analysis_png = os.path.join(fp3_dir, "feature_pertinence_analysis.png")
        if os.path.exists(fp_analysis_png):
            st.image(fp_analysis_png, caption="Feature Divergence & Pertinence across Rounds", use_container_width=True)

    with fp_col2:
        weights_comp_png = os.path.join(fp3_dir, "weights_comparison.png")
        if os.path.exists(weights_comp_png):
            st.image(weights_comp_png, caption="Comparison of Client Aggregation Weights", use_container_width=True)

    with st.expander("Mathematical Formulation of Feature-Pertinence Weighting", expanded=True):
        st.markdown(r"""
        **1. Continuous Feature Divergence ($FD_m$):**
        Quantifies the 2-Wasserstein moment discrepancy between client $m$'s feature distribution $(\mu_{m,j}, \sigma_{m,j})$ and global feature reference $(\mu_{g,j}, \sigma_{g,j})$ across all $d$ input features:
        $$FD_m = \frac{1}{d} \sum_{j=1}^d \frac{(\mu_{m,j} - \mu_{g,j})^2 + (\sigma_{m,j} - \sigma_{g,j})^2}{\sigma_{g,j}^2 + \epsilon}$$

        **2. Client Relative Divergence & Feature-Pertinence Score ($FP_m$):**
        Normalized relative to peer clients and converted via exponential sensitivity decay ($\beta_{\text{FP}} = 1.0$):
        $$\widetilde{FD}_m = \frac{FD_m}{\frac{1}{M}\sum_{k=1}^M FD_k + \epsilon}, \qquad FP_m = \exp(-\beta_{\text{FP}} \cdot \widetilde{FD}_m) \in (0, 1]$$

        **3. Tri-Factor Fed-FP3 Aggregation Weight ($w_m$):**
        Modulates FedPRADA's sample size, label KL-divergence, and loss stability with the feature-pertinence factor:
        $$w_m^{\text{raw}} = n_m \cdot \exp(-\lambda_{\text{div}} D_m) \cdot S_m^\gamma \cdot FP_m^{\alpha_{\text{FP}}}, \qquad w_m = \frac{w_m^{\text{raw}}}{\sum_{k=1}^M w_k^{\text{raw}}}$$
        """)

    st.divider()

    # -----------------------------------------------------
    # SECTION 6: DELIVERY RISK PRIORITY LIST
    # -----------------------------------------------------
    st.header(f"6. Shipment Delivery Risk Priority List ({selected_ds})")
    st.caption("Ranked risk triage derived directly from Fed-FP3 global model prediction probabilities.")

    if os.path.exists(prio_path):
        try:
            prio_df = pd.read_csv(prio_path)
            total_shipments = len(prio_df)

            # Summary KPIs
            k1, k2, k3, k4, k5 = st.columns(5)
            crit_count = int((prio_df["Risk_Priority"] == "CRITICAL").sum())
            high_count = int((prio_df["Risk_Priority"] == "HIGH").sum())
            med_count = int((prio_df["Risk_Priority"] == "MEDIUM").sum())
            low_med_count = int((prio_df["Risk_Priority"] == "LOW-MEDIUM").sum())
            low_count = int((prio_df["Risk_Priority"] == "LOW").sum())

            k1.metric("Total Shipments", f"{total_shipments:,}")
            k2.metric("CRITICAL Risk", f"{crit_count:,}", f"{crit_count/total_shipments*100:.1f}%")
            k3.metric("HIGH Risk", f"{high_count:,}", f"{high_count/total_shipments*100:.1f}%")
            k4.metric("MEDIUM Risk", f"{med_count:,}", f"{med_count/total_shipments*100:.1f}%")
            k5.metric("LOW / LOW-MED Risk", f"{low_count + low_med_count:,}", f"{(low_count + low_med_count)/total_shipments*100:.1f}%")

            # Interactive Filter Controls
            st.subheader("Filter and Sort Shipments")
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)

            with fcol1:
                all_priorities = ["CRITICAL", "HIGH", "MEDIUM", "LOW-MEDIUM", "LOW"]
                selected_priorities = st.multiselect(
                    "Priority Level",
                    options=all_priorities,
                    default=all_priorities,
                    key=f"prio_filter_{ds_key}"
                )

            with fcol2:
                all_clients = sorted(prio_df["Client_Partition"].unique().tolist())
                selected_clients = st.multiselect(
                    "Client Partition",
                    options=all_clients,
                    default=all_clients,
                    key=f"client_filter_{ds_key}"
                )

            with fcol3:
                min_conf = st.slider(
                    "Min Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key=f"conf_filter_{ds_key}"
                )

            with fcol4:
                sort_option = st.selectbox(
                    "Sort By",
                    ["Composite_Risk_Score (High to Low)", "Confidence (High to Low)", "Shipment_ID"],
                    key=f"sort_filter_{ds_key}"
                )

            # Apply filters
            filtered_df = prio_df[
                (prio_df["Risk_Priority"].isin(selected_priorities)) &
                (prio_df["Client_Partition"].isin(selected_clients)) &
                (prio_df["Confidence"] >= min_conf)
            ].copy()

            if "Composite_Risk_Score" in sort_option:
                filtered_df = filtered_df.sort_values("Composite_Risk_Score", ascending=False)
            elif "Confidence" in sort_option:
                filtered_df = filtered_df.sort_values("Confidence", ascending=False)
            else:
                filtered_df = filtered_df.sort_values("Shipment_ID")

            st.caption(f"Showing **{len(filtered_df):,}** of **{total_shipments:,}** evaluated shipments.")

            # Display table
            display_cols = [
                "Shipment_ID", "Client_Partition", "Predicted_Status", "Actual_Status",
                "Confidence_Percentage", "Risk_Priority", "Recommended_Action",
                "Composite_Risk_Score"
            ]
            display_table = filtered_df[display_cols]

            # Style table with priority color mapping
            def highlight_priority(row):
                val = row["Risk_Priority"]
                if val == "CRITICAL":
                    return ["background-color: #f8d7da; color: #721c24; font-weight: bold;"] * len(row)
                elif val == "HIGH":
                    return ["background-color: #fff3cd; color: #856404; font-weight: bold;"] * len(row)
                elif val == "MEDIUM":
                    return ["background-color: #d1ecf1; color: #0c5460;"] * len(row)
                elif val in ("LOW-MEDIUM", "LOW"):
                    return ["background-color: #d4edda; color: #155724;"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display_table.style.apply(highlight_priority, axis=1).format({
                    "Composite_Risk_Score": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=420
            )

            # CSV Download Button
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Download Filtered Shipment Risk List ({len(filtered_df)} Shipments CSV)",
                data=csv_data,
                file_name=f"{ds_key}_shipment_risk_prioritization.csv",
                mime="text/csv",
                key=f"download_csv_{ds_key}"
            )

            with st.expander("Operational Playbook Guidelines", expanded=False):
                st.markdown(r"""
                - **CRITICAL Priority**: High-confidence late delivery / non-delivery prediction ($\ge 70\%$). Immediate dispatch of expedited replacement shipment, emergency courier rerouting, and active customer notification.
                - **HIGH Priority**: Predicted delivery delay or cancellation risk ($50\% - 70\%$). Immediate warehouse dispatch priority escalation, priority carrier handoff, and contingency inventory allocation.
                - **MEDIUM Priority**: Moderate delivery risk or lower confidence delay prediction. Automated carrier route tracking, proactive transit monitoring, and safety stock audit.
                - **LOW / LOW-MEDIUM Priority**: Confirmed on-time or early arrival prediction. Standard logistics processing, regular batch fulfillment, and automated milestone notification.
                """)

        except Exception as e:
            st.error(f"Error loading shipment prioritization list: {e}")

    st.divider()

    # -----------------------------------------------------
    # SECTION 7: RESULTS AND PERSISTENCE
    # -----------------------------------------------------
    st.header(f"7. Results & Artifacts Persistence ({selected_ds})")
    st.caption("Verifiable record of saved checkpoints, convergence logs, and configuration metadata.")

    files_info = []
    if os.path.exists(fp3_dir):
        for fname in sorted(os.listdir(fp3_dir)):
            fpath = os.path.join(fp3_dir, fname)
            if os.path.isfile(fpath):
                fstat = os.stat(fpath)
                mtime_str = datetime.fromtimestamp(fstat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                files_info.append({
                    "Artifact Name": fname,
                    "Size (KB)": f"{fstat.st_size / 1024:.2f}",
                    "Last Modified": mtime_str,
                    "Type": "Checkpoint" if fname.endswith(".pth") else ("Data / Metrics" if fname.endswith(".csv") else ("Metadata" if fname.endswith(".json") else "Plot")),
                    "Full Path": fpath
                })

    if files_info:
        st.dataframe(pd.DataFrame(files_info), use_container_width=True, hide_index=True)

    with st.expander("Inspect Raw Fed-FP3 Summary JSON", expanded=False):
        st.json(fp3_summary)

    with st.expander("Inspect Raw Evaluation Metrics JSON", expanded=False):
        st.json(eval_metrics)

    st.divider()

    # -----------------------------------------------------
    # SECTION 8: PREDICT NEW SHIPMENT & REAL-TIME PRIORITIZATION
    # -----------------------------------------------------
    st.header(f"8. Predict New Shipment & Real-Time Prioritization")
    st.caption(
        "Interactive decision-support interface: Enter individual shipment attributes or upload batch manifests "
        "to obtain real-time federated model predictions, confidence estimation, calibrated composite risk scoring, "
        "and prioritized dispatch queue rankings."
    )

    if "fed_fp3_prediction_history" not in st.session_state:
        st.session_state["fed_fp3_prediction_history"] = []

    pred_tab_single, pred_tab_batch = st.tabs(["🚀 Single Shipment Prediction", "📁 Batch Manifest CSV Upload"])

    # -----------------------------------------------------
    # TAB 1: SINGLE SHIPMENT PREDICTION
    # -----------------------------------------------------
    with pred_tab_single:
        st.subheader("Single Shipment Attributes & Scenario Simulation")

        col_p1, col_p2 = st.columns([2, 3])
        with col_p1:
            pred_ds_choice = st.radio(
                "Inference Pipeline / Dataset Model",
                ["E-Commerce Shipping", "DataCo Supply Chain"],
                horizontal=True,
                key="pred_target_ds_radio"
            )
        pred_key = "ecommerce" if pred_ds_choice == "E-Commerce Shipping" else "dataco"

        with col_p2:
            st.info(
                f"Using trained Fed-FP3 checkpoint: `outputs/fed_fp3/{pred_key}/global_model_final.pth` | "
                f"Target Classes: {2 if pred_key == 'ecommerce' else 4} classes"
            )

        # Preset scenarios definitions
        if pred_key == "ecommerce":
            ecomm_presets = {
                "Custom Entry": {},
                "🔴 Standard Road - Promo Order (Critical Delay Risk)": {
                    "mode": "Road", "calls": 6, "rating": 2, "cost": 265.0, "prior": 4,
                    "importance": "low", "gender": "M", "discount": 45.0, "weight": 1450.0
                },
                "🟢 Express Flight - High-Priority Urgent Parcel (On-Time Dispatch)": {
                    "mode": "Flight", "calls": 2, "rating": 5, "cost": 180.0, "prior": 3,
                    "importance": "high", "gender": "F", "discount": 5.0, "weight": 4200.0
                },
                "🟡 Maritime Cargo - Medium Discount Standard Transit (Routine Handling)": {
                    "mode": "Ship", "calls": 4, "rating": 3, "cost": 210.0, "prior": 3,
                    "importance": "medium", "gender": "M", "discount": 12.0, "weight": 5200.0
                },
                "🟠 Road Delivery - Heavyweight Bulky Parcel (Elevated Delay Risk)": {
                    "mode": "Road", "calls": 5, "rating": 2, "cost": 280.0, "prior": 5,
                    "importance": "medium", "gender": "F", "discount": 15.0, "weight": 6800.0
                }
            }
            preset_name = st.selectbox(
                "🎯 Preset Scenarios (Pre-populates realistic logistical cases)",
                list(ecomm_presets.keys()),
                key="ecomm_preset_select"
            )
            ps_vals = ecomm_presets[preset_name]

            # Shipment Tracking ID
            default_sid = f"ECOMM-TRK-{datetime.now().strftime('%H%M%S')}"
            ship_id = st.text_input("Shipment ID / Tracking Ref", value=default_sid, key="ecomm_sid_input")

            # Form fields layout
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.markdown("##### 🚚 Shipment Logistics & Package Attributes")
                mode_opts = ["Flight", "Road", "Ship"]
                default_mode = ps_vals.get("mode", "Road")
                in_mode = st.selectbox("Mode of Shipment", mode_opts, index=mode_opts.index(default_mode) if default_mode in mode_opts else 1, key="in_mode")

                in_cost = st.number_input(
                    "Cost of Product ($)",
                    min_value=1.0, max_value=1500.0,
                    value=float(ps_vals.get("cost", 210.0)),
                    step=5.0, format="%.2f", key="in_cost"
                )
                in_weight = st.number_input(
                    "Package Weight (grams)",
                    min_value=100.0, max_value=15000.0,
                    value=float(ps_vals.get("weight", 3500.0)),
                    step=50.0, format="%.1f", key="in_weight"
                )
                in_discount = st.slider(
                    "Promotional Discount Offered (%)",
                    min_value=0.0, max_value=75.0,
                    value=float(ps_vals.get("discount", 10.0)),
                    step=1.0, key="in_discount"
                )

            with col_f2:
                st.markdown("##### 👤 Customer Profile & Service Interactions")
                in_calls = st.number_input(
                    "Customer Care Calls",
                    min_value=1, max_value=20,
                    value=int(ps_vals.get("calls", 4)),
                    step=1, key="in_calls"
                )
                in_rating = st.slider(
                    "Customer Service Rating (1 = Poor, 5 = Excellent)",
                    min_value=1, max_value=5,
                    value=int(ps_vals.get("rating", 3)),
                    step=1, key="in_rating"
                )
                in_prior = st.number_input(
                    "Prior Purchases Count",
                    min_value=0, max_value=25,
                    value=int(ps_vals.get("prior", 3)),
                    step=1, key="in_prior"
                )
                imp_opts = ["low", "medium", "high"]
                default_imp = ps_vals.get("importance", "medium")
                in_imp = st.selectbox("Product Importance Tier", imp_opts, index=imp_opts.index(default_imp) if default_imp in imp_opts else 1, key="in_imp")

                gen_opts = ["F", "M"]
                default_gen = ps_vals.get("gender", "M")
                in_gender = st.selectbox("Customer Gender", gen_opts, index=gen_opts.index(default_gen) if default_gen in gen_opts else 1, key="in_gender")

            raw_input_dict = {
                "Mode_of_Shipment": in_mode,
                "Customer_care_calls": in_calls,
                "Customer_rating": in_rating,
                "Cost_of_the_Product": in_cost,
                "Prior_purchases": in_prior,
                "Product_importance": in_imp,
                "Gender": in_gender,
                "Discount_offered": in_discount,
                "Weight_in_gms": in_weight,
            }

        else: # DataCo Supply Chain
            dataco_presets = {
                "Custom Entry": {},
                "🔴 Standard Class - Tight Transit Schedule (Late Delivery Risk)": {
                    "mode": "Standard Class", "type": "DEBIT", "days": 2, "sales": 310.0,
                    "qty": 3, "price": 110.0, "disc": 0.15, "profit": -15.0, "segment": "Consumer"
                },
                "🟢 First Class Express - Dedicated Rush Dispatch (On-Time Arrival)": {
                    "mode": "First Class", "type": "TRANSFER", "days": 4, "sales": 480.0,
                    "qty": 2, "price": 240.0, "disc": 0.05, "profit": 95.0, "segment": "Corporate"
                },
                "🟠 Promotional Clearance - High Discount (Cancellation Risk)": {
                    "mode": "Same Day", "type": "PAYMENT", "days": 1, "sales": 85.0,
                    "qty": 1, "price": 95.0, "disc": 0.25, "profit": -50.0, "segment": "Consumer"
                },
                "🔵 Bulk Commercial Wholesale - Extended Lead Time (Advance Arrival)": {
                    "mode": "Second Class", "type": "TRANSFER", "days": 4, "sales": 1250.0,
                    "qty": 5, "price": 260.0, "disc": 0.08, "profit": 310.0, "segment": "Corporate"
                }
            }
            preset_name = st.selectbox(
                "🎯 Preset Scenarios (Pre-populates realistic logistical cases)",
                list(dataco_presets.keys()),
                key="dataco_preset_select"
            )
            ps_vals = dataco_presets[preset_name]

            # Shipment Tracking ID
            default_sid = f"DC-TRK-{datetime.now().strftime('%H%M%S')}"
            ship_id = st.text_input("Shipment ID / Tracking Ref", value=default_sid, key="dataco_sid_input")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("##### 🚚 Order & Shipping Specifications")
                mode_opts = ["Standard Class", "First Class", "Second Class", "Same Day"]
                default_mode = ps_vals.get("mode", "Standard Class")
                in_smode = st.selectbox("Carrier Shipping Mode", mode_opts, index=mode_opts.index(default_mode) if default_mode in mode_opts else 0, key="in_smode")

                in_days = st.number_input(
                    "Scheduled Shipping Transit Window (Days)",
                    min_value=0, max_value=6,
                    value=int(ps_vals.get("days", 3)),
                    step=1, key="in_days"
                )
                type_opts = ["DEBIT", "TRANSFER", "PAYMENT", "CASH"]
                default_type = ps_vals.get("type", "DEBIT")
                in_type = st.selectbox("Payment Transaction Type", type_opts, index=type_opts.index(default_type) if default_type in type_opts else 0, key="in_type")

                seg_opts = ["Consumer", "Corporate", "Home Office"]
                default_seg = ps_vals.get("segment", "Consumer")
                in_segment = st.selectbox("Customer Segment", seg_opts, index=seg_opts.index(default_seg) if default_seg in seg_opts else 0, key="in_segment")

            with col_d2:
                st.markdown("##### 💰 Financial, Pricing & Product Details")
                in_sales = st.number_input(
                    "Sales per Customer ($)",
                    min_value=5.0, max_value=3000.0,
                    value=float(ps_vals.get("sales", 180.0)),
                    step=10.0, format="%.2f", key="in_sales"
                )
                in_qty = st.number_input(
                    "Order Item Quantity",
                    min_value=1, max_value=10,
                    value=int(ps_vals.get("qty", 2)),
                    step=1, key="in_qty"
                )
                in_price = st.number_input(
                    "Product Item Price ($)",
                    min_value=5.0, max_value=2000.0,
                    value=float(ps_vals.get("price", 120.0)),
                    step=5.0, format="%.2f", key="in_price"
                )
                in_disc = st.slider(
                    "Order Item Discount Rate",
                    min_value=0.0, max_value=0.35,
                    value=float(ps_vals.get("disc", 0.10)),
                    step=0.01, key="in_disc"
                )
                in_profit = st.number_input(
                    "Estimated Benefit / Profit per Order ($)",
                    min_value=-500.0, max_value=1500.0,
                    value=float(ps_vals.get("profit", 30.0)),
                    step=10.0, format="%.2f", key="in_profit"
                )

            raw_input_dict = {
                "Shipping Mode": in_smode,
                "Days for shipment (scheduled)": in_days,
                "Type": in_type,
                "Customer Segment": in_segment,
                "Sales per customer": in_sales,
                "Order Item Quantity": in_qty,
                "Order Item Product Price": in_price,
                "Product Price": in_price,
                "Order Item Discount Rate": in_disc,
                "Benefit per order": in_profit,
            }

            with st.expander("⚙️ Advanced: Inspect / Override 53-Feature Vector (DataCo)", expanded=False):
                st.caption("Training median baselines automatically pre-fill non-logistical demographic and categorical attributes.")
                features_all, medians_all, _, _ = get_dataco_feature_metadata()
                adv_cols = st.columns(3)
                for i_feat, feat_name in enumerate(features_all):
                    if feat_name not in raw_input_dict:
                        col_idx = i_feat % 3
                        val_default = medians_all.get(feat_name, 0.0)
                        adv_val = adv_cols[col_idx].number_input(
                            f"{feat_name}",
                            value=float(val_default),
                            key=f"adv_{feat_name}"
                        )
                        raw_input_dict[feat_name] = adv_val

        # Prediction CTA
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Predict Delivery Risk & Assign Priority", type="primary", use_container_width=True)

        if predict_btn:
            try:
                with st.spinner("Executing Fed-FP3 calibrated inference..."):
                    pred_res = run_fed_fp3_inference(pred_key, raw_input_dict, shipment_id=ship_id)

                # Store in session history
                st.session_state["fed_fp3_prediction_history"].append(pred_res)

                st.success("✅ Real-time Fed-FP3 inference completed successfully!")

                # Result presentation
                prio_tier = pred_res["Risk_Priority"]
                color_map = {
                    "CRITICAL": "#d32f2f",
                    "HIGH": "#f57c00",
                    "MEDIUM": "#fbc02d",
                    "LOW-MEDIUM": "#0288d1",
                    "LOW": "#388e3c"
                }
                badge_color = color_map.get(prio_tier, "#333333")

                # Prediction Summary Card
                st.markdown(
                    f"""
                    <div style="border: 2px solid {badge_color}; border-radius: 10px; padding: 20px; background-color: rgba(255, 255, 255, 0.05); margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span style="font-size: 1.25rem; font-weight: bold; color: #ffffff;">📦 Shipment: <code>{pred_res['Shipment_ID']}</code> ({pred_res['Dataset']})</span>
                            <span style="background-color: {badge_color}; color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 1.05rem; letter-spacing: 0.5px;">
                                {prio_tier} PRIORITY
                            </span>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 15px;">
                            <div>
                                <small style="color: #aaaaaa;">Predicted Status</small><br>
                                <strong style="font-size: 1.15rem; color: #ffffff;">{pred_res['Predicted_Status']}</strong>
                            </div>
                            <div>
                                <small style="color: #aaaaaa;">Model Confidence</small><br>
                                <strong style="font-size: 1.15rem; color: #ffffff;">{pred_res['Confidence_Percentage']}</strong>
                            </div>
                            <div>
                                <small style="color: #aaaaaa;">Composite Risk Score</small><br>
                                <strong style="font-size: 1.15rem; color: {badge_color};">{pred_res['Composite_Risk_Score']:.4f}</strong>
                            </div>
                            <div>
                                <small style="color: #aaaaaa;">Decision Impact</small><br>
                                <strong style="font-size: 1.05rem; color: #ffffff;">{'Immediate Escalation' if prio_tier in ['CRITICAL', 'HIGH'] else 'Standard Routine'}</strong>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Recommended Action Banner
                if prio_tier == "CRITICAL":
                    st.error(f"🚨 **RECOMMENDED OPERATIONAL ACTION**: {pred_res['Recommended_Action']}")
                elif prio_tier == "HIGH":
                    st.warning(f"⚠️ **RECOMMENDED OPERATIONAL ACTION**: {pred_res['Recommended_Action']}")
                elif prio_tier == "MEDIUM":
                    st.info(f"ℹ️ **RECOMMENDED OPERATIONAL ACTION**: {pred_res['Recommended_Action']}")
                else:
                    st.success(f"✅ **RECOMMENDED OPERATIONAL ACTION**: {pred_res['Recommended_Action']}")

                # Class Probability Breakdown
                st.markdown("##### 📊 Class Probability Breakdown")
                prob_cols = st.columns(len(pred_res["Class_Probabilities"]))
                for col_pb, (c_name, c_prob) in zip(prob_cols, pred_res["Class_Probabilities"].items()):
                    is_top = (c_name == pred_res["Predicted_Status"])
                    with col_pb:
                        st.metric(
                            label=f"{'⭐ ' if is_top else ''}{c_name}",
                            value=f"{c_prob * 100:.2f}%"
                        )
                        st.progress(float(c_prob))

                # Operational Reasoning Explanation
                with st.expander("🔍 Risk Prioritization Explanation & Contributing Factors", expanded=True):
                    st.write(f"**Operational Rationale**: {pred_res['Explanation']}")
                    st.caption(f"**Input Parameter Summary**: `{pred_res['Input_Summary']}`")
                    st.caption(
                        "**Decision Rule Policy**: Threshold conditions adhere to Fed-FP3 multi-factor risk prioritization: "
                        "Critical delays trigger proactive rerouting; on-time high-confidence trajectories proceed on standard cadence."
                    )

                st.caption(
                    "⚠️ *Operational Disclaimer: Model predictions and risk scores are probabilistic assessments generated "
                    "by the trained Fed-FP3 model to support logistics queue dispatch prioritization. Not an absolute guarantee of shipment transit time.*"
                )

            except Exception as e:
                st.error(f"❌ Input Validation or Inference Error: {e}")

        # -----------------------------------------------------
        # SESSION PRIORITIZATION QUEUE TABLE
        # -----------------------------------------------------
        st.divider()
        st.subheader("📋 Session Prioritization Queue & Risk Ranking")
        st.caption(
            "Continuously accumulated shipments evaluated during this session, automatically ranked "
            "by Composite Risk Score (highest operational risk at Rank #1)."
        )

        history_items = st.session_state.get("fed_fp3_prediction_history", [])

        if history_items:
            hist_df = pd.DataFrame(history_items).sort_values("Composite_Risk_Score", ascending=False).reset_index(drop=True)
            hist_df.insert(0, "Rank", range(1, len(hist_df) + 1))

            col_q1, col_q2 = st.columns([3, 1])
            with col_q1:
                tier_filter = st.multiselect(
                    "Filter Queue by Priority Tier",
                    ["CRITICAL", "HIGH", "MEDIUM", "LOW-MEDIUM", "LOW"],
                    default=["CRITICAL", "HIGH", "MEDIUM", "LOW-MEDIUM", "LOW"],
                    key="queue_tier_filter"
                )
            with col_q2:
                st.metric("Total Queued Shipments", len(hist_df))

            filtered_hist = hist_df[hist_df["Risk_Priority"].isin(tier_filter)].copy()

            show_cols = [
                "Rank", "Shipment_ID", "Dataset", "Predicted_Status",
                "Confidence_Percentage", "Risk_Priority", "Composite_Risk_Score",
                "Recommended_Action", "Timestamp"
            ]
            avail_show = [c for c in show_cols if c in filtered_hist.columns]

            def highlight_queue(row):
                val = row.get("Risk_Priority", "")
                if val == "CRITICAL":
                    return ["background-color: #f8d7da; color: #721c24; font-weight: bold;"] * len(row)
                elif val == "HIGH":
                    return ["background-color: #fff3cd; color: #856404; font-weight: bold;"] * len(row)
                elif val == "MEDIUM":
                    return ["background-color: #d1ecf1; color: #0c5460;"] * len(row)
                elif val in ("LOW-MEDIUM", "LOW"):
                    return ["background-color: #d4edda; color: #155724;"] * len(row)
                return [""] * len(row)

            st.dataframe(
                filtered_hist[avail_show].style.apply(highlight_queue, axis=1).format({
                    "Composite_Risk_Score": "{:.4f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            col_act1, col_act2 = st.columns([2, 1])
            with col_act1:
                hist_csv = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Export Prioritization Queue ({len(hist_df)} Shipments CSV)",
                    data=hist_csv,
                    file_name=f"fed_fp3_session_prioritization_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_queue_csv"
                )
            with col_act2:
                if st.button("🗑️ Clear Session History", key="clear_history_btn"):
                    st.session_state["fed_fp3_prediction_history"] = []
                    st.rerun()

        else:
            st.info("No shipments predicted in this session yet. Submit a shipment above or upload a batch manifest to generate the prioritized queue.")

    # -----------------------------------------------------
    # TAB 2: BATCH CSV UPLOAD
    # -----------------------------------------------------
    with pred_tab_batch:
        st.subheader("Batch Manifest Upload & Prioritization")
        st.caption("Upload a CSV file containing multiple shipments for rapid vectorized inference and global risk queue ranking.")

        batch_ds_choice = st.radio(
            "Batch Dataset Schema",
            ["E-Commerce Shipping", "DataCo Supply Chain"],
            horizontal=True,
            key="batch_ds_choice_radio"
        )
        batch_key = "ecommerce" if batch_ds_choice == "E-Commerce Shipping" else "dataco"

        uploaded_manifest = st.file_uploader(
            f"Upload CSV Manifest for {batch_ds_choice}",
            type=["csv"],
            key=f"uploader_{batch_key}"
        )

        if uploaded_manifest is not None:
            try:
                batch_raw_df = pd.read_csv(uploaded_manifest)
                st.write(f"**Uploaded Manifest Preview** ({len(batch_raw_df)} rows, {len(batch_raw_df.columns)} columns):")
                st.dataframe(batch_raw_df.head(5), use_container_width=True)

                if st.button("🚀 Process Batch Predictions & Build Prioritization Queue", type="primary", key="run_batch_btn"):
                    with st.spinner(f"Running vectorized Fed-FP3 inference on {len(batch_raw_df)} shipments..."):
                        t_start = time.time()
                        batch_res_df = run_fed_fp3_batch_inference(batch_key, batch_raw_df)
                        elapsed_s = time.time() - t_start

                    st.success(f"✅ Processed {len(batch_res_df)} shipments in {elapsed_s:.2f} seconds ({elapsed_s / max(len(batch_res_df), 1) * 1000:.1f} ms/shipment)!")

                    # Metrics summary
                    m_b1, m_b2, m_b3, m_b4 = st.columns(4)
                    crit_count = (batch_res_df["Risk_Priority"] == "CRITICAL").sum()
                    high_count = (batch_res_df["Risk_Priority"] == "HIGH").sum()
                    med_count = (batch_res_df["Risk_Priority"] == "MEDIUM").sum()
                    low_count = batch_res_df["Risk_Priority"].isin(["LOW", "LOW-MEDIUM"]).sum()

                    m_b1.metric("Critical Risk Shipments", f"{crit_count} ({crit_count / len(batch_res_df):.1%})")
                    m_b2.metric("High Risk Shipments", f"{high_count} ({high_count / len(batch_res_df):.1%})")
                    m_b3.metric("Medium Risk Shipments", f"{med_count} ({med_count / len(batch_res_df):.1%})")
                    m_b4.metric("Low Risk (On-Time)", f"{low_count} ({low_count / len(batch_res_df):.1%})")

                    # Display Prioritized Batch Table
                    st.dataframe(
                        batch_res_df.style.apply(highlight_queue, axis=1).format({
                            "Composite_Risk_Score": "{:.4f}",
                            "Confidence": "{:.4f}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Export batch CSV
                    b_csv = batch_res_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"📥 Download Prioritized Batch Manifest ({len(batch_res_df)} Shipments CSV)",
                        data=b_csv,
                        file_name=f"fed_fp3_batch_prioritization_{batch_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_batch_csv_btn"
                    )

            except Exception as e:
                st.error(f"❌ Error processing batch CSV manifest: {e}")



# =========================================================
# ROUTING CHECK
# =========================================================
if nav_selection == "Fed-FP3 Novelty":
    render_fed_fp3_dashboard()
    st.stop()

if nav_selection == "Research Performance Benchmark":
    render_research_performance_benchmark(is_embedded=False)
    st.stop()


# =========================================================
# TOP-LEVEL MODEL SUMMARY
# =========================================================
st.header("Final Accuracy of the Two Models")

summary_cols = st.columns(2)

for col, (dataset_name, config) in zip(
    summary_cols,
    DATASETS.items(),
):
    current, completed = dataset_results(
        dataset_name,
        config,
    )

    summary = read_summary(
        config["directory"]
    )

    if summary and summary.get("average_final_accuracy") is not None:
        final_accuracy = float(
            summary["average_final_accuracy"]
        )
        label = "Final Average Accuracy"
    elif completed:
        final_accuracy = sum(completed) / len(completed)
        label = "Final Average Accuracy"
    elif current:
        final_accuracy = sum(current) / len(current)
        label = "Current Average Validation Accuracy"
    else:
        final_accuracy = None
        label = "Final Accuracy"

    with col:
        st.metric(
            f"{dataset_name}",
            fmt_percent(final_accuracy),
            help=(
                "For a completed model this is the average of the "
                "best validation accuracies from its five local clients."
            ),
        )

        if summary:
            st.caption(
                f"Completed clients: "
                f"{summary.get('clients_completed', 0)} / "
                f"{len(config['clients'])}"
            )
        else:
            st.caption(label)


# =========================================================
# FINAL GLOBAL MODEL ACCURACY (FedPRADA Module 3)
# =========================================================
st.header("Final Global Model Accuracy")

global_cols = st.columns(2)

for col, (dataset_name, config) in zip(
    global_cols,
    DATASETS.items(),
):

    global_accuracy = read_global_accuracy(
        config["global_directory"]
    )

    global_summary = read_global_summary(
        config["global_directory"]
    )

    with col:
        st.metric(
            f"{dataset_name} - Global Model",
            fmt_percent(global_accuracy),
        )

        if global_summary:
            f = global_summary.get("final") or {}
            round_value = f.get("round", global_summary.get("best_round", "-"))
            f1_value = f.get("f1", global_summary.get("final_global_val_f1"))
            loss_value = f.get("loss", global_summary.get("final_global_val_loss"))
            precision_value = f.get("precision")
            recall_value = f.get("recall")

            st.caption(
                f"Round: {round_value} | "
                f"F1: {fmt_percent(f1_value)} | "
                f"Precision: {fmt_percent(precision_value)} | "
                f"Recall: {fmt_percent(recall_value)} | "
                f"Loss: {fmt_number(loss_value)}"
            )
        else:
            st.caption(
                "Run FedPRADA Module 3 to generate the global-model accuracy files."
            )


st.divider()


# =========================================================
# MODEL COMPARISON
# =========================================================
comparison_rows = []

for dataset_name, config in DATASETS.items():
    current, completed = dataset_results(
        dataset_name,
        config,
    )

    summary = read_summary(
        config["directory"]
    )

    if summary and summary.get("average_final_accuracy") is not None:
        accuracy = float(
            summary["average_final_accuracy"]
        )
        source = "Final"
    elif completed:
        accuracy = float(
            np.mean(completed)
        )
        source = "Completed clients"
    elif current:
        accuracy = float(
            np.mean(current)
        )
        source = "Current validation"
    else:
        accuracy = None
        source = "No data"

    global_accuracy = read_global_accuracy(
        config["global_directory"]
    )

    comparison_rows.append(
        {
            "Model": dataset_name,
            "Accuracy": (
                f"{accuracy * 100:.2f}%"
                if accuracy is not None
                else "-"
            ),
            "Status": source,
            "Global Model Accuracy": (
                f"{global_accuracy * 100:.2f}%"
                if global_accuracy is not None
                else "-"
            ),
        }
    )

st.subheader("Model Accuracy Comparison")

st.dataframe(
    pd.DataFrame(comparison_rows),
    use_container_width=True,
    hide_index=True,
)

st.divider()


# =========================================================
# DATASET DASHBOARDS
# =========================================================
for dataset_name, config in DATASETS.items():

    st.header(
        f"{dataset_name} Model Training"
    )

    dataset_dir = config["directory"]

    rows = []
    histories = {}

    for client in config["clients"]:

        status = read_client_status(
            dataset_dir,
            client,
        )

        history = read_client_history(
            dataset_dir,
            client,
        )

        histories[client] = history

        final = read_client_final(
            dataset_dir,
            client,
        )

        if status:
            rows.append(
                {
                    "Client": client,
                    "Status": status_text(status),
                    "Epoch": (
                        f"{status.get('current_epoch', 0)} / "
                        f"{status.get('total_epochs', 0)}"
                    ),
                    "Train Accuracy": fmt_percent(
                        status.get("train_accuracy")
                    ),
                    "Validation Accuracy": fmt_percent(
                        status.get("val_accuracy")
                    ),
                    "Best Accuracy": fmt_percent(
                        status.get("best_val_accuracy")
                    ),
                    "Val Loss": fmt_number(
                        status.get("val_loss")
                    ),
                    "Learning Rate": (
                        f"{status.get('learning_rate', 0):.2e}"
                        if status.get("learning_rate") is not None
                        else "-"
                    ),
                }
            )

        elif final:
            rows.append(
                {
                    "Client": client,
                    "Status": "Completed",
                    "Epoch": str(
                        final.get("total_epochs_run", "-")
                    ),
                    "Train Accuracy": "-",
                    "Validation Accuracy": fmt_percent(
                        final.get("final_accuracy")
                    ),
                    "Best Accuracy": fmt_percent(
                        final.get("final_accuracy")
                    ),
                    "Val Loss": "-",
                    "Learning Rate": "-",
                }
            )

        else:
            rows.append(
                {
                    "Client": client,
                    "Status": "Not started",
                    "Epoch": "-",
                    "Train Accuracy": "-",
                    "Validation Accuracy": "-",
                    "Best Accuracy": "-",
                    "Val Loss": "-",
                    "Learning Rate": "-",
                }
            )


    # -----------------------------------------------------
    # Dataset metrics
    # -----------------------------------------------------
    current_values = []
    best_values = []

    for client in config["clients"]:

        status = read_client_status(
            dataset_dir,
            client,
        )

        if status:

            if status.get("val_accuracy") is not None:
                current_values.append(
                    float(status["val_accuracy"])
                )

            if status.get("best_val_accuracy") is not None:
                best_values.append(
                    float(status["best_val_accuracy"])
                )


    final_values = []

    for client in config["clients"]:

        final = read_client_final(
            dataset_dir,
            client,
        )

        if final and final.get("final_accuracy") is not None:
            final_values.append(
                float(final["final_accuracy"])
            )


    if final_values:

        model_accuracy = float(
            sum(final_values) / len(final_values)
        )

    elif best_values:

        model_accuracy = float(
            sum(best_values) / len(best_values)
        )

    else:

        model_accuracy = None


    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Final Model Accuracy",
        fmt_percent(model_accuracy),
    )

    m2.metric(
        "Best Client Accuracy",
        fmt_percent(
            max(final_values)
            if final_values
            else (
                max(best_values)
                if best_values
                else None
            )
        ),
    )

    m3.metric(
        "Clients Reporting",
        len(
            current_values
        ),
    )

    active_count = sum(
        1
        for client in config["clients"]
        if (
            read_client_status(
                dataset_dir,
                client,
            )
            or {}
        ).get("status") == "training"
    )

    m4.metric(
        "Active Clients",
        active_count,
    )

    m5.metric(
        "Global Model Accuracy",
        fmt_percent(
            read_global_accuracy(
                config["global_directory"]
            )
        ),
    )


    # -----------------------------------------------------
    # Client table
    # -----------------------------------------------------
    st.subheader(
        f"{dataset_name} - Client Metrics"
    )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------
    st.subheader(
        f"{dataset_name} - Client Progress"
    )

    progress_cols = st.columns(
        len(config["clients"])
    )

    for col, client in zip(
        progress_cols,
        config["clients"],
    ):

        status = read_client_status(
            dataset_dir,
            client,
        )

        with col:

            st.write(
                f"**Client {client}**"
            )

            st.progress(
                progress_percent(status)
            )

            if status:

                st.caption(
                    f"{status_text(status)} | "
                    f"Epoch "
                    f"{status.get('current_epoch', 0)}/"
                    f"{status.get('total_epochs', 0)}"
                )

            else:

                st.caption(
                    "Waiting for training..."
                )


    # -----------------------------------------------------
    # Accuracy graph
    # -----------------------------------------------------
    st.subheader(
        f"{dataset_name} - Training Accuracy"
    )

    chart_frames = []

    for client, history in histories.items():

        if history.empty:
            continue

        required = {
            "epoch",
            "train_accuracy",
            "val_accuracy",
        }

        if not required.issubset(
            history.columns
        ):
            continue

        temp = history[
            [
                "epoch",
                "train_accuracy",
                "val_accuracy",
            ]
        ].copy()

        temp["Client"] = client

        chart_frames.append(
            temp
        )

    if chart_frames:

        chart_df = pd.concat(
            chart_frames,
            ignore_index=True,
        )

        train_df = chart_df.pivot(
            index="epoch",
            columns="Client",
            values="train_accuracy",
        )

        val_df = chart_df.pivot(
            index="epoch",
            columns="Client",
            values="val_accuracy",
        )

        st.write("Training accuracy")

        st.line_chart(
            train_df,
            use_container_width=True,
        )

        st.write("Validation accuracy")

        st.line_chart(
            val_df,
            use_container_width=True,
        )

    else:

        st.info(
            "Start local training to see the accuracy curves."
        )


    # -----------------------------------------------------
    # Loss graph
    # -----------------------------------------------------
    st.subheader(
        f"{dataset_name} - Training / Validation Loss"
    )

    loss_frames = []

    for client, history in histories.items():

        if history.empty:
            continue

        required = {
            "epoch",
            "train_loss",
            "val_loss",
        }

        if not required.issubset(
            history.columns
        ):
            continue

        temp = history[
            [
                "epoch",
                "train_loss",
                "val_loss",
            ]
        ].copy()

        temp["Client"] = client

        loss_frames.append(
            temp
        )

    if loss_frames:

        loss_df = pd.concat(
            loss_frames,
            ignore_index=True,
        )

        train_loss_df = loss_df.pivot(
            index="epoch",
            columns="Client",
            values="train_loss",
        )

        val_loss_df = loss_df.pivot(
            index="epoch",
            columns="Client",
            values="val_loss",
        )

        st.write("Training loss")

        st.line_chart(
            train_loss_df,
            use_container_width=True,
        )

        st.write("Validation loss")

        st.line_chart(
            val_loss_df,
            use_container_width=True,
        )

    else:

        st.info(
            "Loss data will appear during training."
        )


    # -----------------------------------------------------
    # Precision / Recall / F1
    # -----------------------------------------------------
    st.subheader(
        f"{dataset_name} - Validation Metrics"
    )

    metric_frames = []

    for client, history in histories.items():

        if history.empty:
            continue

        required = {
            "epoch",
            "val_precision",
            "val_recall",
            "val_f1",
        }

        if not required.issubset(
            history.columns
        ):
            continue

        temp = history[
            [
                "epoch",
                "val_precision",
                "val_recall",
                "val_f1",
            ]
        ].copy()

        temp["Client"] = client

        metric_frames.append(
            temp
        )

    if metric_frames:

        metric_df = pd.concat(
            metric_frames,
            ignore_index=True,
        )

        precision_df = metric_df.pivot(
            index="epoch",
            columns="Client",
            values="val_precision",
        )

        recall_df = metric_df.pivot(
            index="epoch",
            columns="Client",
            values="val_recall",
        )

        f1_df = metric_df.pivot(
            index="epoch",
            columns="Client",
            values="val_f1",
        )

        st.write("Validation precision")

        st.line_chart(
            precision_df,
            use_container_width=True,
        )

        st.write("Validation recall")

        st.line_chart(
            recall_df,
            use_container_width=True,
        )

        st.write("Validation F1")

        st.line_chart(
            f1_df,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # Confusion matrices / saved graphs
    # -----------------------------------------------------
    with st.expander(
        f"{dataset_name} - Saved training outputs"
    ):

        for client in config["clients"]:

            client_dir = os.path.join(
                dataset_dir,
                f"client_{client}",
            )

            cm_path = os.path.join(
                client_dir,
                "confusion_matrix.png",
            )

            if os.path.exists(cm_path):

                st.write(
                    f"Client {client} confusion matrix"
                )

                st.image(
                    cm_path,
                    use_container_width=True,
                )

        global_cm_path = os.path.join(
            config["global_directory"],
            "final_global_confusion_matrix.png",
        )

        if os.path.exists(global_cm_path):

            st.write(
                f"{dataset_name} - Global model confusion matrix"
            )

            st.image(
                global_cm_path,
                use_container_width=True,
            )

        # FedPRADA Module 3 output graphs (DataCo and E-Commerce).
        st.subheader(f"{dataset_name} - FedPRADA Module 3 Graphs")
        global_graphs = [
            ("global_model_performance.png", "Global model performance"),
            ("kl_divergence_per_round.png", "KL divergence per round"),
            ("stability_per_round.png", "Client reliability / stability per round"),
            ("client_validation_per_round.png", "Client validation performance per round"),
            ("weighted_fusion_weights.png", "Adaptive weighted-fusion weights"),
            ("weighted_fusion_breakdown.png", "Weighted-fusion breakdown"),
            ("fusion_parameter_movement.png", "Global parameter movement during fusion"),
        ]

        graph_found = False
        for filename, graph_title in global_graphs:
            graph_path = os.path.join(config["global_directory"], filename)
            if os.path.exists(graph_path):
                graph_found = True
                st.write(graph_title)
                st.image(graph_path, use_container_width=True)

        if not graph_found:
            st.info(
                "FedPRADA graphs will appear here after Module 3 creates its PNG output files. "
                f"Expected folder: {config['global_directory']}"
            )

        # Show round-by-round global model metrics when available.
        global_metrics_path = os.path.join(
            config["global_directory"], "global_metrics.csv"
        )
        if os.path.exists(global_metrics_path):
            st.subheader(f"{dataset_name} - Global Model Metrics by Round")
            try:
                global_metrics_df = pd.read_csv(global_metrics_path)
                st.dataframe(global_metrics_df, use_container_width=True, hide_index=True)
                if "round" in global_metrics_df.columns and "global_val_acc" in global_metrics_df.columns:
                    plot_df = global_metrics_df.set_index("round")[["global_val_acc"]].rename(
                        columns={"global_val_acc": "Global validation accuracy"}
                    )
                    st.line_chart(plot_df, use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not read global_metrics.csv: {exc}")

        fusion_log_path = os.path.join(config["global_directory"], "fusion_log.csv")
        if os.path.exists(fusion_log_path):
            with st.expander(f"{dataset_name} - FedPRADA fusion log"):
                try:
                    st.dataframe(pd.read_csv(fusion_log_path), use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.warning(f"Could not read fusion_log.csv: {exc}")

    st.divider()


# =========================================================
# NEW SHIPMENT PREDICTION
# =========================================================
st.header("New Shipment Prediction")
st.caption(
    "Enter numeric feature values using each model's processed client-file "
    "representation. DataCo values are already encoded and standardized. "
    "E-Commerce values receive the same federated training-split "
    "standardization used by FedPRADA Module 3."
)

prediction_configs = (
    (
        "DataCo",
        os.path.join(
            DATASETS["DataCo"]["global_directory"],
            "global_model_final.pth",
        ),
        "Delivery Status",
    ),
    (
        "E-Commerce",
        os.path.join(
            DATASETS["E-Commerce"]["global_directory"],
            "global_model_final.pth",
        ),
        "On-time delivery status",
    ),
)

for dataset_name, checkpoint_path, prediction_target in prediction_configs:
    output_title = (
        "DataCo Prediction"
        if dataset_name == "DataCo"
        else "E-Commerce Prediction"
    )
    st.subheader(f"{dataset_name} Shipment Inputs")

    try:
        feature_columns = prediction_feature_columns(dataset_name)
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"{dataset_name} prediction is unavailable: {exc}")
        st.subheader(output_title)
        continue

    st.caption(
        f"Features: {len(feature_columns)} | Prediction target: {prediction_target}"
    )
    input_values = {}
    with st.expander(f"{dataset_name} feature form", expanded=True):
        with st.form(f"{dataset_name.lower()}_shipment_prediction_form"):
            input_columns = st.columns(2)
            for index, feature in enumerate(feature_columns):
                with input_columns[index % len(input_columns)]:
                    input_values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        step=0.1,
                        format="%.6f",
                        key=f"{dataset_name}_{feature}_prediction_input",
                    )

            submitted = st.form_submit_button("Predict")

    result_key = f"new_shipment_prediction_{dataset_name}"
    if submitted:
        st.session_state.pop(result_key, None)
        try:
            st.session_state[result_key] = predict_shipment(
                dataset_name,
                checkpoint_path,
                feature_columns,
                input_values,
            )
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            st.error(f"{dataset_name} prediction failed: {exc}")

    st.subheader(output_title)
    result = st.session_state.get(result_key)
    if result is not None:
        st.success(f"Predicted class: **{result['predicted_label']}**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Class label": result["labels"],
                    "Probability / Confidence": [
                        f"{probability:.2%}"
                        for probability in result["probabilities"]
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()


# =========================================================
# FEDERATED MODEL COMPARISON
# =========================================================
st.header("Federated Model Comparison")
st.caption(
    "Tables compare models trained on identical, leakage-free train/validation/"
    "test partitions with preprocessing fitted on training samples only. The "
    "original FedPRADA checkpoints are shown separately as legacy references."
)

render_research_performance_benchmark(is_embedded=True)

st.markdown("#### Legacy Baseline Checkpoints & Convergence Plots")

comparison_root = os.path.join(
    BASE,
    "fedprada_baseline_comparison",
)
comparison_datasets = (
    ("dataco", "DataCo"),
    ("ecommerce", "E-Commerce Shipping"),
)
comparison_available = False

for dataset_key, dataset_title in comparison_datasets:
    result_dir = os.path.join(comparison_root, dataset_key)
    csv_path = os.path.join(result_dir, "results.csv")
    json_path = os.path.join(result_dir, "results.json")

    st.subheader(dataset_title)
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        st.info(
            f"No saved {dataset_title} comparison results yet. "
            "Run the isolated experiment script to populate this section."
        )
        continue

    try:
        comparison_results = pd.read_csv(csv_path)
        with open(json_path, "r", encoding="utf-8") as comparison_file:
            comparison_metadata = json.load(comparison_file)
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        st.error(f"Could not load {dataset_title} comparison results: {exc}")
        continue

    required_columns = {
        "method",
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
        "test_sample_count",
    }
    if not required_columns.issubset(comparison_results.columns):
        st.error(
            f"{dataset_title} comparison results are missing required "
            f"columns: {sorted(required_columns - set(comparison_results.columns))}"
        )
        continue

    if comparison_results.empty:
        st.warning(f"The saved {dataset_title} comparison contains no results.")
        continue
    comparison_results = comparison_results[
        comparison_results["method"].isin(
            ("FedAvg", "FedProx", "FedAWA", "FedPRADA")
        )
    ].copy()
    required_methods = ("FedAvg", "FedProx", "FedAWA", "FedPRADA")
    if set(comparison_results["method"]) != set(required_methods) or (
        comparison_results["method"].duplicated().any()
    ):
        st.error(
            f"{dataset_title} comparison must contain exactly one result for "
            "FedAvg, FedProx, FedAWA, and FedPRADA."
        )
        continue
    comparison_results["method"] = pd.Categorical(
        comparison_results["method"],
        categories=required_methods,
        ordered=True,
    )
    comparison_results = comparison_results.sort_values("method")

    settings = (
        comparison_metadata.get("experiment_settings")
        if isinstance(comparison_metadata, dict)
        else None
    )
    if not isinstance(settings, dict) or not settings.get(
        "training_only_preprocessing_fit"
    ):
        st.warning(
            f"The saved {dataset_title} results predate the leakage-free "
            "comparison pipeline. Rerun run_baseline_comparison.py before "
            "using these metrics."
        )
        continue

    if not comparison_results[list(required_columns - {"method"})].notna().all().all():
        st.error(f"{dataset_title} comparison results contain missing metrics.")
        continue

    test_count = int(comparison_metadata.get("test_sample_count", -1))
    if test_count < 1 or not (
        comparison_results["test_sample_count"] == test_count
    ).all():
        st.error(f"{dataset_title} test sample counts do not match the saved metadata.")
        continue

    convergence_path = os.path.join(result_dir, "convergence.csv")
    try:
        convergence_data = pd.read_csv(convergence_path)
    except (OSError, pd.errors.ParserError) as exc:
        st.error(f"Could not load {dataset_title} convergence history: {exc}")
        continue
    required_convergence_columns = {
        "round",
        "method",
        "validation_accuracy",
        "validation_loss",
    }
    if not required_convergence_columns.issubset(convergence_data.columns):
        st.error(
            f"{dataset_title} convergence CSV is missing required columns: "
            f"{sorted(required_convergence_columns - set(convergence_data.columns))}"
        )
        continue
    expected_rounds = int(settings.get("rounds", 0))
    history_is_valid = (
        expected_rounds > 0
        and set(convergence_data["method"]) == set(required_methods)
        and len(convergence_data)
        == expected_rounds * len(required_methods)
        and convergence_data[
            ["round", "validation_accuracy", "validation_loss"]
        ].notna().all().all()
    )
    if history_is_valid:
        history_is_valid = all(
            convergence_data.loc[
                convergence_data["method"] == method,
                "round",
            ].astype(int).tolist()
            == list(range(1, expected_rounds + 1))
            for method in required_methods
        )
    if not history_is_valid:
        st.error(
            f"{dataset_title} convergence history is incomplete, duplicated, "
            "or inconsistent with the configured communication rounds."
        )
        continue

    comparison_available = True
    st.caption(f"Test samples: {test_count:,}")

    fedprada_rows = comparison_results[
        comparison_results["method"] == "FedPRADA"
    ]
    if len(fedprada_rows) != 1:
        st.error(
            f"{dataset_title} results must contain exactly one FedPRADA "
            "reference row to compute accuracy differences."
        )
        continue
    fedprada_accuracy = float(fedprada_rows["test_accuracy"].iloc[0])

    table = pd.DataFrame(
        {
            "Method": comparison_results["method"],
            "Final Test Accuracy (%)": comparison_results["test_accuracy"] * 100,
            "Precision (%)": comparison_results["test_precision_macro"] * 100,
            "Recall (%)": comparison_results["test_recall_macro"] * 100,
            "F1-score (%)": comparison_results["test_f1_macro"] * 100,
            "Difference vs FedPRADA (pp)": (
                comparison_results["test_accuracy"] - fedprada_accuracy
            )
            * 100,
        }
    )
    best_accuracy = table["Final Test Accuracy (%)"].max()
    styled_table = table.style.apply(
        lambda column: [
            "background-color: #c6efce" if value == best_accuracy else ""
            for value in column
        ],
        subset=["Final Test Accuracy (%)"],
    ).format(
        {
            "Final Test Accuracy (%)": "{:.2f}",
            "Precision (%)": "{:.2f}",
            "Recall (%)": "{:.2f}",
            "F1-score (%)": "{:.2f}",
            "Difference vs FedPRADA (pp)": "{:+.2f}",
        }
    )
    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Highest measured test accuracy: "
        f"{table.loc[table['Final Test Accuracy (%)'].idxmax(), 'Method']} "
        f"({best_accuracy:.2f}%). Differences are percentage points "
        "relative to FedPRADA."
    )

    with st.expander(f"{dataset_title} experiment settings"):
        st.json(settings)

    legacy_path = os.path.join(
        result_dir,
        "legacy_fedprada_reference.json",
    )
    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, "r", encoding="utf-8") as legacy_file:
                legacy_reference = json.load(legacy_file)
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not load the {dataset_title} legacy reference: {exc}")
        else:
            with st.expander(f"{dataset_title} existing FedPRADA checkpoint (legacy)"):
                st.warning(
                    legacy_reference.get(
                        "comparability",
                        "This is a historical checkpoint score, not part of the "
                        "leakage-free comparison.",
                    )
                )
                st.caption(
                    "Original validation-holdout accuracy (not an independent "
                    f"test score): "
                    f"{float(legacy_reference['test_accuracy']) * 100:.2f}%"
                )
                st.write(
                    f"Source checkpoint: "
                    f"`{legacy_reference.get('source_checkpoint', 'unknown')}`"
                )

    accuracy_graph = os.path.join(
        result_dir,
        "test_accuracy_comparison.png",
    )
    convergence_graph = os.path.join(result_dir, "convergence.png")
    if os.path.exists(accuracy_graph):
        st.image(
            accuracy_graph,
            caption=f"{dataset_title} test accuracy comparison",
            use_container_width=True,
        )
    else:
        st.warning(f"Test accuracy graph is missing: {accuracy_graph}")

    if os.path.exists(convergence_graph):
        st.image(
            convergence_graph,
            caption=(
                f"{dataset_title} validation accuracy by communication round "
                f"({expected_rounds} measured rounds)"
            ),
            use_container_width=True,
        )
    else:
        st.warning(f"Convergence graph is missing: {convergence_graph}")

if not comparison_available:
    st.code(
        "python run_baseline_comparison.py --dataset both --tune-rounds 5\n"
        "streamlit run dashboard.py",
        language="powershell",
    )

st.divider()


# =========================================================
# FOOTER
# =========================================================
st.caption(
    "Last dashboard refresh: "
    + datetime.now().strftime("%H:%M:%S")
)

with st.expander("How to run"):

    st.code(
        """pip install streamlit streamlit-autorefresh pandas matplotlib torch scikit-learn

# Terminal 0 - run the paper-budget baseline comparison (DataCo 100 rounds,
# E-Commerce 67 rounds; optional CL/DL adds considerable CPU time)
python run_baseline_comparison.py --dataset both --tune-rounds 5 --include-cl-dl

# Terminal 1
python local_dataco.py

# Terminal 2
python local_ecommerce.py

# Terminal 3
streamlit run live_dashboard.py
""",
        language="bash",
    )

    st.write(
        "The dashboard updates every 2 seconds while the "
        "two local training programs are running."
    )