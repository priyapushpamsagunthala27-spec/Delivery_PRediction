import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================

os.makedirs("processed", exist_ok=True)
os.makedirs("processed/clients", exist_ok=True)


print("========================================")
print("       DATA PREPROCESSING")
print("========================================")


# =========================================================
# 1. DATACO SUPPLY CHAIN DATASET
# =========================================================

print("\n========================================")
print("      DATACO SUPPLY CHAIN DATASET")
print("========================================")


# ---------------------------------------------------------
# LOAD DATACO DATASET
# ---------------------------------------------------------

file_path = "DataCo/DataCoSupplyChainDataset.csv"

df = pd.read_csv(
    file_path,
    encoding="latin1"
)

print("\nOriginal DataCo dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ---------------------------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------------------------

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:")
print(df.shape)


# ---------------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------------

print("\nMissing values before preprocessing:")
print(df.isnull().sum().sum())

# Fill numerical missing values with median
numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )


# Fill categorical missing values with mode
categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


print("\nMissing values after preprocessing:")
print(df.isnull().sum().sum())


# ---------------------------------------------------------
# CONVERT DATE COLUMNS
# ---------------------------------------------------------

date_columns = [
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]

for column in date_columns:

    if column in df.columns:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        # Extract useful date information
        df[column + "_year"] = df[column].dt.year
        df[column + "_month"] = df[column].dt.month
        df[column + "_day"] = df[column].dt.day


# ---------------------------------------------------------
# DROP ORIGINAL DATE COLUMNS
# ---------------------------------------------------------

for column in date_columns:

    if column in df.columns:
        df = df.drop(column, axis=1)


# ---------------------------------------------------------
# ENCODE CATEGORICAL DATA
# ---------------------------------------------------------

print("\nEncoding categorical columns...")

categorical_columns = df.select_dtypes(
    include="object"
).columns

label_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

    label_encoders[column] = encoder


print("Categorical encoding completed.")


# ---------------------------------------------------------
# CONVERT INF VALUES
# ---------------------------------------------------------

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.fillna(0)


# ---------------------------------------------------------
# DISPLAY PREPROCESSED DATA
# ---------------------------------------------------------

print("\nDataCo shape after preprocessing:")
print(df.shape)

print("\nFirst 5 rows after preprocessing:")
print(df.head())


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

scaler_dataco = StandardScaler()

df_scaled = scaler_dataco.fit_transform(df)

df_scaled = pd.DataFrame(
    df_scaled,
    columns=df.columns
)


print("\nDataCo normalization completed.")


# ---------------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------------

train_dataco, test_dataco = train_test_split(
    df_scaled,
    test_size=0.20,
    random_state=42
)


print("\nDataCo training shape:")
print(train_dataco.shape)

print("DataCo testing shape:")
print(test_dataco.shape)


# ---------------------------------------------------------
# SAVE DATACO DATA
# ---------------------------------------------------------

train_dataco.to_csv(
    "processed/dataco_train.csv",
    index=False
)

test_dataco.to_csv(
    "processed/dataco_test.csv",
    index=False
)

df_scaled.to_csv(
    "processed/dataco_processed.csv",
    index=False
)


print("\nDataCo preprocessing completed.")
print("DataCo processed files saved.")


# =========================================================
# NEW: SPLIT DATACO INTO 5 FEDERATED CLIENTS
# =========================================================

print("\n========================================")
print("   SPLITTING DATACO INTO 5 CLIENTS")
print("========================================")


# ---------------------------------------------------------
# CREATE DATACO CLIENT FOLDER
# ---------------------------------------------------------

os.makedirs(
    "processed/clients/dataco",
    exist_ok=True
)


# ---------------------------------------------------------
# CHECK MARKET COLUMN
# ---------------------------------------------------------

if "Market" in df.columns:

    print("\nDataCo Market column found.")

    # Get the encoder used for the Market column
    market_encoder = label_encoders["Market"]

    market_names = [
        "Africa",
        "Europe",
        "LATAM",
        "Pacific Asia",
        "USCA"
    ]

    for market in market_names:

        # Convert original market name into
        # its encoded numerical value
        encoded_market = market_encoder.transform(
            [market]
        )[0]

        # IMPORTANT:
        # Use the ORIGINAL pre-normalized df to
        # identify the client rows.
        market_mask = df["Market"] == encoded_market

        # Take the SAME rows from the normalized dataset
        client_df = df_scaled.loc[
            market_mask
        ].copy()

        # Reset index
        client_df = client_df.reset_index(drop=True)

        # Output path
        output_path = (
            f"processed/clients/dataco/"
            f"client_{market.lower().replace(' ', '_')}.csv"
        )

        # Save client dataset
        client_df.to_csv(
            output_path,
            index=False
        )

        print(
            f"Client {market}: "
            f"{client_df.shape[0]} rows -> "
            f"{output_path}"
        )

else:

    print("\nERROR: 'Market' column was not found in DataCo dataset.")

    print("Available columns:")
    print(df.columns.tolist())


# =========================================================
# 2. E-COMMERCE DATASET
# =========================================================

print("\n========================================")
print("          E-COMMERCE DATASET")
print("========================================")


# ---------------------------------------------------------
# LOAD E-COMMERCE DATASET
# ---------------------------------------------------------

file_path = "ecommerce/Train.csv"

ecommerce = pd.read_csv(
    file_path
)

print("\nOriginal E-commerce dataset shape:")
print(ecommerce.shape)

print("\nFirst 5 rows:")
print(ecommerce.head())


# ---------------------------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------------------------

print("\nDuplicate rows:")
print(ecommerce.duplicated().sum())

ecommerce = ecommerce.drop_duplicates()

print("\nShape after removing duplicates:")
print(ecommerce.shape)


# ---------------------------------------------------------
# DROP ID COLUMN
# ---------------------------------------------------------

if "ID" in ecommerce.columns:
    ecommerce = ecommerce.drop(columns=["ID"])


# ---------------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------------

print("\nMissing values before preprocessing:")
print(ecommerce.isnull().sum().sum())


# Numerical columns
numeric_columns = ecommerce.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    ecommerce[column] = ecommerce[column].fillna(
        ecommerce[column].median()
    )


# Categorical columns (Warehouse_block kept as text for now, used for client split)
categorical_columns = ecommerce.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    if ecommerce[column].isnull().sum() > 0:

        ecommerce[column] = ecommerce[column].fillna(
            ecommerce[column].mode()[0]
        )


print("\nMissing values after preprocessing:")
print(ecommerce.isnull().sum().sum())


# ---------------------------------------------------------
# ENCODE CATEGORICAL COLUMNS (except Warehouse_block, needed for client split)
# ---------------------------------------------------------

print("\nEncoding E-commerce categorical columns...")

categorical_columns = [
    c for c in ecommerce.select_dtypes(include="object").columns
    if c != "Warehouse_block"
]

ecommerce_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    ecommerce[column] = encoder.fit_transform(
        ecommerce[column].astype(str)
    )

    ecommerce_encoders[column] = encoder


print("E-commerce categorical encoding completed.")


# ---------------------------------------------------------
# HANDLE INFINITE VALUES
# ---------------------------------------------------------

ecommerce = ecommerce.replace(
    [np.inf, -np.inf],
    np.nan
)

ecommerce[numeric_columns] = ecommerce[numeric_columns].fillna(0)


# ---------------------------------------------------------
# DISPLAY PREPROCESSED DATA
# ---------------------------------------------------------

print("\nE-commerce shape after preprocessing:")
print(ecommerce.shape)

print("\nFirst 5 rows after preprocessing:")
print(ecommerce.head())


# ---------------------------------------------------------
# NORMALIZATION (numeric feature columns only)
# ---------------------------------------------------------

num_cols = [
    "Customer_care_calls", "Customer_rating", "Cost_of_the_Product",
    "Prior_purchases", "Discount_offered", "Weight_in_gms"
]

num_cols = [c for c in num_cols if c in ecommerce.columns]

scaler_ecommerce = StandardScaler()

ecommerce[num_cols] = scaler_ecommerce.fit_transform(
    ecommerce[num_cols]
)


print("\nE-commerce normalization completed.")


# ---------------------------------------------------------
# TRAIN TEST SPLIT (full processed e-commerce data)
# ---------------------------------------------------------

ecommerce_for_split = ecommerce.drop(columns=["Warehouse_block"])

train_ecommerce, test_ecommerce = train_test_split(
    ecommerce_for_split,
    test_size=0.20,
    random_state=42
)


print("\nE-commerce training shape:")
print(train_ecommerce.shape)

print("E-commerce testing shape:")
print(test_ecommerce.shape)


# ---------------------------------------------------------
# SAVE E-COMMERCE DATA
# ---------------------------------------------------------

train_ecommerce.to_csv(
    "processed/ecommerce_train.csv",
    index=False
)

test_ecommerce.to_csv(
    "processed/ecommerce_test.csv",
    index=False
)

ecommerce_for_split.to_csv(
    "processed/ecommerce_processed.csv",
    index=False
)


print("\nE-commerce preprocessing completed.")
print("E-commerce processed files saved.")


# ---------------------------------------------------------
# SPLIT E-COMMERCE INTO 5 CLIENTS (A, B, C, D, F) BY Warehouse_block
# ---------------------------------------------------------

print("\nSplitting E-commerce dataset into 5 clients (A, B, C, D, F)...")

for block in ["A", "B", "C", "D", "F"]:

    client_df = ecommerce[ecommerce["Warehouse_block"] == block]
    client_df = client_df.drop(columns=["Warehouse_block"])
    client_df = client_df.reset_index(drop=True)

    out_path = f"processed/clients/client_{block}.csv"
    client_df.to_csv(out_path, index=False)

    print(f"Client {block}: {client_df.shape[0]} rows -> {out_path}")


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n========================================")
print("   ALL PREPROCESSING COMPLETED")
print("========================================")

print("\nGenerated files:")

print("\nDataCo:")
print("  processed/dataco_train.csv")
print("  processed/dataco_test.csv")
print("  processed/dataco_processed.csv")

print("\nDataCo clients:")
print("  processed/clients/dataco/client_africa.csv")
print("  processed/clients/dataco/client_europe.csv")
print("  processed/clients/dataco/client_latam.csv")
print("  processed/clients/dataco/client_pacific_asia.csv")
print("  processed/clients/dataco/client_usca.csv")

print("\nE-commerce:")
print("  processed/ecommerce_train.csv")
print("  processed/ecommerce_test.csv")
print("  processed/ecommerce_processed.csv")

print("\nE-commerce clients:")
print("  processed/clients/client_A.csv")
print("  processed/clients/client_B.csv")
print("  processed/clients/client_C.csv")
print("  processed/clients/client_D.csv")
print("  processed/clients/client_F.csv")

print("\n========================================")
print("          PREPROCESSING DONE")
print("========================================")