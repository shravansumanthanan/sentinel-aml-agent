import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional


def load_and_merge_kaggle_datasets(
    data_dir: str = "data",
    max_rows: int = 500_000,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Universal Dataset Loader for AML Transaction Analysis.

    Supports multiple data sources in priority order:
    1. Local CSV files in data/ directory (transactions.csv, customers.csv)
    2. Any Kaggle-format CSVs (PaySim or IBM AML schema auto-detected)
       with chunked reading & 100% label-preserving sampling.
    3. Synthetic generation fallback for demo/hackathon mode

    Returns
    -------
    df_transactions : pd.DataFrame
    df_customers    : pd.DataFrame
    customer_labels : Optional pd.Series indexed by customer_id
                      (1 = laundering confirmed, 0 = clean) — None if unavailable.
    """
    os.makedirs(data_dir, exist_ok=True)
    # Priority 1: Auto-detect newly added Kaggle/PaySim/Custom CSVs in data/ (excluding default transactions.csv/customers.csv)
    csv_files = [
        f for f in glob.glob(os.path.join(data_dir, "*.csv"))
        if os.path.basename(f).lower() not in ["transactions.csv", "customers.csv", "customers_processed.csv"]
    ]
    df_tx_list = []

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        try:
            sample_df = pd.read_csv(fpath, nrows=10).dropna(how="all")
            if sample_df.empty:
                print(f"⚠️ [Ingestion] Skipping empty file: {fname}")
                continue
            cols = [c.lower().strip() for c in sample_df.columns]

            if "nameorig" in cols or "namedest" in cols:
                # PaySim schema — Chunked reading with max_rows cap
                chunks = []
                total_read = 0
                for chunk in pd.read_csv(fpath, chunksize=100_000):
                    chunks.append(chunk)
                    total_read += len(chunk)
                    if total_read >= max_rows:
                        print(f"⚡ [Ingestion] PaySim CSV capped at {max_rows:,} rows")
                        break
                df_raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
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
                    "channel": np.random.choice(["Online", "Mobile", "ATM", "Branch"], len(df_raw)),
                    "destination_account": df_raw["namedest"].astype(str),
                    "country_code": np.random.choice(["US", "CA", "GB", "KY", "PA", "AE"], len(df_raw), p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
                })
                if "isfraud" in df_raw.columns:
                    df_clean["is_laundering"] = df_raw["isfraud"].astype(int).values
                df_tx_list.append(df_clean)

            elif "amount paid" in cols or "payment format" in cols or "from bank" in cols:
                # IBM AML schema (ealtman2019 dataset) — Chunked & 100% Label-Preserving Sampling
                chunks = []
                for chunk in pd.read_csv(fpath, chunksize=100_000):
                    chunks.append(chunk)
                df_raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                df_raw.columns = [c.lower().strip() for c in df_raw.columns]

                # Identify laundering label column if present
                laundering_col = next((c for c in df_raw.columns if "launder" in c.lower()), None)
                
                # Label-preserving sampling if dataset size exceeds max_rows
                if len(df_raw) > max_rows:
                    if laundering_col:
                        pos_mask = df_raw[laundering_col].astype(int) == 1
                        pos_df = df_raw[pos_mask]
                        neg_df = df_raw[~pos_mask]
                        n_neg_needed = max(0, max_rows - len(pos_df))
                        neg_sampled = neg_df.sample(n=min(n_neg_needed, len(neg_df)), random_state=42)
                        df_raw = pd.concat([pos_df, neg_sampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)
                        print(f"⚡ [Ingestion] IBM AML Dataset sampled to {len(df_raw):,} rows (Preserved 100% of {len(pos_df):,} laundering positives)")
                    else:
                        df_raw = df_raw.sample(n=max_rows, random_state=42).reset_index(drop=True)

                acct_col = "account" if "account" in df_raw.columns else "from bank"
                dest_col = "account.1" if "account.1" in df_raw.columns else ("to bank" if "to bank" in df_raw.columns else acct_col)
                amount_col = "amount paid" if "amount paid" in df_raw.columns else "amount"
                type_col = "payment format" if "payment format" in df_raw.columns else "transaction type"

                laundering_series = None
                if laundering_col:
                    laundering_series = pd.Series(
                        df_raw[laundering_col].astype(int).values,
                        name="is_laundering",
                    )
                    print(f"✅ [IBM AML] Found '{laundering_col}' labels — "
                          f"{int(laundering_series.sum())} laundering / {len(laundering_series)} total rows")

                df_clean = pd.DataFrame({
                    "transaction_id": [f"IBM-TX-{i+1:07d}" for i in range(len(df_raw))],
                    "customer_id": df_raw[acct_col].astype(str),
                    "timestamp": pd.to_datetime(df_raw["timestamp"], errors="coerce"),
                    "amount": pd.to_numeric(df_raw[amount_col], errors="coerce").fillna(0.0),
                    "transaction_type": df_raw[type_col].astype(str),
                    "channel": np.random.choice(["Branch", "Wire", "Online"], len(df_raw)),
                    "destination_account": df_raw[dest_col].astype(str),
                    "country_code": np.random.choice(["US", "CA", "GB", "KY", "PA", "AE"], len(df_raw), p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]),
                })
                if laundering_series is not None:
                    df_clean["is_laundering"] = laundering_series.values
                df_tx_list.append(df_clean)

        except Exception as e:
            print(f"Notice: Skipping file {fname}: {e}")

    if df_tx_list:
        df_transactions = pd.concat(df_tx_list, ignore_index=True)
    else:
        tx_path = os.path.join(data_dir, "transactions.csv")
        cust_path = os.path.join(data_dir, "customers.csv")
        if os.path.exists(tx_path) and os.path.exists(cust_path):
            print("✅ Loading pre-existing transactions.csv and customers.csv from data/")
            df_tx = pd.read_csv(tx_path)
            df_cust = pd.read_csv(cust_path)
            defaults = {
                "channel": lambda n: np.random.choice(["Online", "Branch", "ATM", "Mobile"], n),
                "destination_account": lambda n: "ACC-EXTERNAL",
                "country_code": lambda n: np.random.choice(["US", "CA", "GB", "KY", "PA", "AE", "DE"], n, p=[0.5, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05])
            }
            n_rows = len(df_tx)
            for col in ["transaction_id", "customer_id", "timestamp", "amount", "transaction_type", "channel", "destination_account", "country_code"]:
                if col not in df_tx.columns:
                    df_tx[col] = defaults[col](n_rows) if col in defaults else ""
            customer_labels = _extract_customer_labels(df_tx)
            return df_tx, df_cust, customer_labels
        else:
            # Priority 3: Generate synthetic demo data
            print("⚠️ No dataset found. Generating synthetic AML demo data...")
            df_transactions = _generate_synthetic_data()

    # Aggregate per-customer labels from transaction-level labels (if present)
    customer_labels = _extract_customer_labels(df_transactions)

    # Generate customer metadata from transaction data
    unique_cust_ids = df_transactions["customer_id"].unique()
    occupations = ["Software Engineer", "Consultant", "Retail Business", "Import/Export",
                   "Real Estate", "Student", "Accountant", "Physician", "Attorney"]
    countries = ["US", "CA", "GB", "DE", "FR", "SG", "AE", "KY", "PA"]
    risk_ratings = ["Low", "Medium", "High"]

    np.random.seed(42)
    customers = []
    for cid in unique_cust_ids:
        cid_str = str(cid)
        display_name = f"Customer_{cid_str.replace('CUST-', '') if cid_str.startswith('CUST-') else cid_str}"
        customers.append({
            "customer_id": cid_str,
            "customer_name": display_name,
            "risk_rating": np.random.choice(risk_ratings, p=[0.6, 0.25, 0.15]),
            "account_opened_date": f"20{np.random.randint(20,26):02d}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
            "kyc_status": np.random.choice(["Verified", "Pending", "Expired"], p=[0.7, 0.2, 0.1]),
            "occupation": np.random.choice(occupations),
            "country": np.random.choice(countries),
        })
    df_customers = pd.DataFrame(customers)

    return df_transactions, df_customers, customer_labels


def _extract_customer_labels(df_tx: pd.DataFrame) -> Optional[pd.Series]:
    """
    Derive a per-customer binary label (1 = launderer) from a transaction-level
    `is_laundering` column, if it exists.

    A customer is labeled 1 if ANY of their transactions is marked as laundering.
    Returns None if the column is absent.
    """
    if "is_laundering" not in df_tx.columns:
        return None
    labels = (
        df_tx.groupby("customer_id")["is_laundering"]
        .max()  # 1 if any transaction is laundering
        .astype(int)
    )
    n_pos = int(labels.sum())
    print(f"✅ [Labels] {n_pos} laundering customers / {len(labels)} total customers")
    return labels


def _generate_synthetic_data() -> pd.DataFrame:
    """Generates a realistic synthetic AML transaction dataset for demo purposes."""
    np.random.seed(42)
    n_customers = 500
    n_transactions = 5000
    
    customer_ids = [f"CUST-{i:04d}" for i in range(1, n_customers + 1)]
    tx_types = ["Deposit", "Withdrawal", "Wire", "Transfer"]
    channels = ["Online", "Branch", "ATM", "Mobile"]
    countries = ["US", "CA", "GB", "DE", "FR", "SG", "AE", "KY", "PA"]
    country_probs = [0.40, 0.10, 0.10, 0.05, 0.05, 0.05, 0.10, 0.10, 0.05]
    
    base_date = datetime(2026, 1, 1)
    
    records = []
    for i in range(n_transactions):
        cid = np.random.choice(customer_ids)
        amt = np.random.lognormal(mean=7.5, sigma=1.5)
        
        # Inject structuring patterns for ~5% of customers
        if int(cid.split("-")[1]) < 25:
            amt = np.random.uniform(8500, 9999)
        
        records.append({
            "transaction_id": f"TX-{i+1:07d}",
            "customer_id": cid,
            "timestamp": (base_date + timedelta(hours=np.random.randint(0, 4320))).isoformat(),
            "amount": round(amt, 2),
            "transaction_type": np.random.choice(tx_types),
            "channel": np.random.choice(channels),
            "destination_account": f"ACC-{np.random.randint(10000, 99999)}",
            "country_code": np.random.choice(countries, p=country_probs)
        })
    
    return pd.DataFrame(records)
