import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_merge_kaggle_datasets(data_dir: str = "/Users/sterlingsuman/Desktop/projectx/data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ingests and normalizes Kaggle IBM AML and PaySim Financial Datasets into SENTINEL-AML relational tables.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    ibm_file = os.path.join(data_dir, "ibm_aml_transactions.csv")
    paysim_file = os.path.join(data_dir, "paysim_transactions.csv")

    df_tx_list = []
    
    # 1. Load Kaggle IBM AML Dataset if available
    if os.path.exists(ibm_file):
        print(f"Loading Kaggle IBM AML dataset from {ibm_file}...")
        df_ibm = pd.read_csv(ibm_file)
        
        # Standardize IBM AML Schema
        df_ibm_clean = pd.DataFrame({
            "transaction_id": [f"IBM-TX-{i+1:06d}" for i in range(len(df_ibm))],
            "customer_id": df_ibm["Account"].astype(str),
            "timestamp": pd.to_datetime(df_ibm["Timestamp"]),
            "amount": df_ibm["Amount Paid"].astype(float),
            "transaction_type": df_ibm["Payment Format"].astype(str),
            "channel": "Branch",
            "destination_account": df_ibm["Account.1"].astype(str),
            "country_code": "US"
        })
        df_tx_list.append(df_ibm_clean)

    # 2. Load Kaggle PaySim Dataset if available
    if os.path.exists(paysim_file):
        print(f"Loading Kaggle PaySim dataset from {paysim_file}...")
        df_ps = pd.read_csv(paysim_file)
        
        # PaySim timestamps: 'step' represents 1 hour intervals
        base_date = datetime(2026, 6, 1)
        timestamps = [base_date + timedelta(hours=int(s)) for s in df_ps["step"]]

        df_ps_clean = pd.DataFrame({
            "transaction_id": [f"PS-TX-{i+1:06d}" for i in range(len(df_ps))],
            "customer_id": df_ps["nameOrig"].astype(str),
            "timestamp": timestamps,
            "amount": df_ps["amount"].astype(float),
            "transaction_type": df_ps["type"].astype(str),
            "channel": "Online",
            "destination_account": df_ps["nameDest"].astype(str),
            "country_code": "US"
        })
        df_tx_list.append(df_ps_clean)

    # 3. Fallback / Combined Ingestion Layer
    if df_tx_list:
        df_transactions = pd.concat(df_tx_list, ignore_index=True)
    else:
        # Load existing converted dataset
        cust_path = os.path.join(data_dir, "customers.csv")
        tx_path = os.path.join(data_dir, "transactions.csv")
        if os.path.exists(tx_path) and os.path.exists(cust_path):
            return pd.read_csv(tx_path), pd.read_csv(cust_path)

    # Extract unique customer profiles from merged Kaggle transaction feeds
    unique_cust_ids = df_transactions["customer_id"].unique()
    customers = []
    occupations = ["Software Engineer", "Consultant", "Retail Business", "Import/Export", "Real Estate", "Student", "Accountant"]
    countries = ["US", "CA", "GB", "DE", "FR", "SG", "AE", "KY", "PA"]

    for cid in unique_cust_ids:
        customers.append({
            "customer_id": cid,
            "customer_name": f"Customer_{cid}",
            "risk_rating": "High" if "4521" in cid or "1089" in cid else "Low",
            "account_opened_date": "2023-01-15",
            "kyc_status": "Verified",
            "occupation": np.random.choice(occupations),
            "country": np.random.choice(countries)
        })

    df_customers = pd.DataFrame(customers)

    # Save standardized Kaggle relational tables
    df_transactions.to_csv(os.path.join(data_dir, "transactions.csv"), index=False)
    df_customers.to_csv(os.path.join(data_dir, "customers.csv"), index=False)

    return df_transactions, df_customers
