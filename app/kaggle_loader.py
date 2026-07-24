import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple

def load_and_merge_kaggle_datasets(data_dir: str = "/Users/sterlingsuman/Desktop/projectx/data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Universal Kaggle Auto-Ingestion Engine.
    Scans data directory for any Kaggle CSV files (IBM AML, PaySim, Synthetic, or Custom),
    automatically detects schemas, normalizes headers, and outputs standardized relational tables.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    df_tx_list = []

    for fpath in csv_files:
        fname = os.path.basename(fpath).lower()
        if fname in ["customers.csv", "customers_processed.csv"]:
            continue

        try:
            print(f"Scanning & Inspecting Kaggle CSV feed: {fname}...")
            # Read first few rows to inspect schema
            sample_df = pd.read_csv(fpath, nrows=5)
            cols = [c.lower().strip() for c in sample_df.columns]

            # Case A: Kaggle PaySim Format (step, type, amount, nameOrig, nameDest, isFraud)
            if "nameorig" in cols or "namedest" in cols:
                print(f" Detected Kaggle PaySim Schema in {fname}")
                df_raw = pd.read_csv(fpath)
                col_map = {c: c.lower().strip() for c in df_raw.columns}
                df_raw = df_raw.rename(columns=col_map)
                
                base_date = datetime(2026, 6, 1)
                timestamps = [base_date + timedelta(hours=int(s)) for s in df_raw.get("step", range(len(df_raw)))]

                df_clean = pd.DataFrame({
                    "transaction_id": [f"PS-TX-{i+1:07d}" for i in range(len(df_raw))],
                    "customer_id": df_raw["nameorig"].astype(str),
                    "timestamp": timestamps,
                    "amount": df_raw["amount"].astype(float),
                    "transaction_type": df_raw["type"].astype(str),
                    "channel": "Online",
                    "destination_account": df_raw["namedest"].astype(str),
                    "country_code": "US"
                })
                df_tx_list.append(df_clean)

            # Case B: Kaggle IBM Transactions Format (Timestamp, Account, Account.1, Amount Paid, Payment Format)
            elif "amount paid" in cols or "payment format" in cols or "from bank" in cols:
                print(f" Detected Kaggle IBM Transactions Schema in {fname}")
                df_raw = pd.read_csv(fpath)
                col_map = {c: c.lower().strip() for c in df_raw.columns}
                df_raw = df_raw.rename(columns=col_map)

                df_clean = pd.DataFrame({
                    "transaction_id": [f"IBM-TX-{i+1:07d}" for i in range(len(df_raw))],
                    "customer_id": df_raw["account"].astype(str),
                    "timestamp": pd.to_datetime(df_raw["timestamp"]),
                    "amount": df_raw["amount paid"].astype(float),
                    "transaction_type": df_raw["payment format"].astype(str),
                    "channel": "Branch",
                    "destination_account": df_raw.get("account.1", df_raw["account"]).astype(str),
                    "country_code": "US"
                })
                df_tx_list.append(df_clean)

            # Case C: Standard or Previously Ingested Schema
            elif "customer_id" in cols and "amount" in cols:
                if fname == "transactions.csv" and len(csv_files) > 1:
                    # Skip existing transactions.csv if raw Kaggle feeds exist alongside
                    continue
                df_raw = pd.read_csv(fpath)
                col_map = {c: c.lower().strip() for c in df_raw.columns}
                df_raw = df_raw.rename(columns=col_map)
                
                # Fill missing schema attributes cleanly
                if "transaction_id" not in df_raw.columns:
                    df_raw["transaction_id"] = [f"TX-{i+1:07d}" for i in range(len(df_raw))]
                if "channel" not in df_raw.columns:
                    df_raw["channel"] = "Online"
                if "destination_account" not in df_raw.columns:
                    df_raw["destination_account"] = "ACC-EXTERNAL"
                if "country_code" not in df_raw.columns:
                    df_raw["country_code"] = "US"

                df_tx_list.append(df_raw)

        except Exception as e:
            print(f"⚠️ Warning: Could not process {fname}: {e}")

    # Fallback / Merged Result Assembly
    if df_tx_list:
        df_transactions = pd.concat(df_tx_list, ignore_index=True)
    else:
        # Load default transactions.csv if available
        tx_path = os.path.join(data_dir, "transactions.csv")
        cust_path = os.path.join(data_dir, "customers.csv")
        if os.path.exists(tx_path) and os.path.exists(cust_path):
            return pd.read_csv(tx_path), pd.read_csv(cust_path)
        else:
            raise FileNotFoundError("No valid Kaggle CSV dataset found in data directory.")

    # Extract unique customer profiles dynamically
    unique_cust_ids = df_transactions["customer_id"].unique()
    
    # Preserve existing customers.csv metadata if present
    cust_path = os.path.join(data_dir, "customers.csv")
    if os.path.exists(cust_path):
        df_existing_cust = pd.read_csv(cust_path)
        existing_ids = set(df_existing_cust["customer_id"].astype(str))
        missing_ids = [cid for cid in unique_cust_ids if str(cid) not in existing_ids]
        
        if missing_ids:
            occupations = ["Software Engineer", "Consultant", "Retail Business", "Import/Export", "Real Estate", "Student", "Accountant"]
            countries = ["US", "CA", "GB", "DE", "FR", "SG", "AE", "KY", "PA"]
            new_custs = []
            for cid in missing_ids:
                new_custs.append({
                    "customer_id": cid,
                    "customer_name": f"Customer_{cid}",
                    "risk_rating": "Medium",
                    "account_opened_date": "2024-01-01",
                    "kyc_status": "Verified",
                    "occupation": np.random.choice(occupations),
                    "country": np.random.choice(countries)
                })
            df_customers = pd.concat([df_existing_cust, pd.DataFrame(new_custs)], ignore_index=True)
        else:
            df_customers = df_existing_cust
    else:
        occupations = ["Software Engineer", "Consultant", "Retail Business", "Import/Export", "Real Estate", "Student", "Accountant"]
        countries = ["US", "CA", "GB", "DE", "FR", "SG", "AE", "KY", "PA"]
        customers = []
        for cid in unique_cust_ids:
            customers.append({
                "customer_id": cid,
                "customer_name": f"Customer_{cid}",
                "risk_rating": "Low",
                "account_opened_date": "2023-01-15",
                "kyc_status": "Verified",
                "occupation": np.random.choice(occupations),
                "country": np.random.choice(countries)
            })
        df_customers = pd.DataFrame(customers)

    # Save standardized relational tables
    df_transactions.to_csv(os.path.join(data_dir, "transactions.csv"), index=False)
    df_customers.to_csv(os.path.join(data_dir, "customers.csv"), index=False)

    print(f"✅ Universal Ingestion Complete: Loaded {len(df_transactions):,} transactions and {len(df_customers):,} customer profiles.")
    return df_transactions, df_customers
