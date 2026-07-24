import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_aml_dataset(data_dir="/Users/sterlingsuman/Desktop/projectx/data", num_customers=5000, num_transactions=50000):
    """
    Generates large-scale Kaggle-standard synthetic AML dataset containing 5,000 customers 
    and 50,000 transactions with embedded money laundering patterns (Structuring, Rapid Velocity, Smurfing).
    """
    os.makedirs(data_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    print(f"Generating Kaggle-scale dataset ({num_customers} customers, {num_transactions} transactions)...")

    # 1. Generate Customers
    occupations = ["Software Engineer", "Consultant", "Retail Business", "Import/Export", "Real Estate", "Student", "Retired", "Accountant", "Medical Doctor", "Financial Analyst"]
    countries = ["US", "CA", "GB", "DE", "FR", "SG", "AE", "KY", "PA"]
    kyc_statuses = ["Verified", "Verified", "Verified", "Pending", "Enhanced"]
    
    customers = []
    for i in range(1, num_customers + 1):
        cid = f"CUST-{i:04d}"
        customers.append({
            "customer_id": cid,
            "customer_name": f"Customer_{i}",
            "risk_rating": random.choice(["Low", "Low", "Low", "Medium", "High"]),
            "account_opened_date": (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 700))).strftime("%Y-%m-%d"),
            "kyc_status": random.choice(kyc_statuses),
            "occupation": random.choice(occupations),
            "country": random.choice(countries)
        })
    
    df_customers = pd.DataFrame(customers)

    transactions = []
    base_date = datetime(2026, 6, 1)

    # 2. Generate baseline transaction ledger (49,500 transactions)
    tx_id_counter = 10000
    for i in range(num_transactions - 500):
        cust = random.choice(customers)
        tx_time = base_date + timedelta(days=random.randint(0, 50), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        
        amount = round(float(np.random.lognormal(mean=5.2, sigma=1.4)), 2)
        amount = max(5.0, min(amount, 95000.0))
        
        tx_type = random.choice(["Deposit", "Transfer", "Withdrawal", "Wire"])
        channel = random.choice(["Online", "ATM", "Branch", "Mobile"])
        dest_acc = f"ACC-{random.randint(1000, 9999)}"
        c_code = cust["country"]

        transactions.append({
            "transaction_id": f"TX-{tx_id_counter}",
            "customer_id": cust["customer_id"],
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "transaction_type": tx_type,
            "channel": channel,
            "destination_account": dest_acc,
            "country_code": c_code
        })
        tx_id_counter += 1

    # 3. Inject In-Depth Structuring Clusters (Multiple subjects across dataset)
    structuring_subjects = ["CUST-4521", "CUST-0420", "CUST-0899", "CUST-1250", "CUST-3300"]
    for s_id in structuring_subjects:
        structuring_start = base_date + timedelta(days=random.randint(10, 40))
        n_deposits = random.randint(10, 20)
        for k in range(n_deposits):
            tx_time = structuring_start + timedelta(days=k // 4, hours=(k % 4) * 2 + random.randint(0, 1))
            amount = round(random.uniform(9100.0, 9950.0), 2)
            transactions.append({
                "transaction_id": f"TX-{tx_id_counter}",
                "customer_id": s_id,
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": amount,
                "transaction_type": "Deposit",
                "channel": "Branch",
                "destination_account": f"ACC-{s_id}-SELF",
                "country_code": "US"
            })
            tx_id_counter += 1

    # 4. Inject Rapid Cash-Out Velocity Spikes
    velocity_subjects = ["CUST-1089", "CUST-0150", "CUST-2200", "CUST-4100"]
    for v_id in velocity_subjects:
        velocity_start = base_date + timedelta(days=random.randint(15, 45), hours=random.randint(8, 14))
        wire_amount = round(random.uniform(150000.0, 500000.0), 2)
        transactions.append({
            "transaction_id": f"TX-{tx_id_counter}",
            "customer_id": v_id,
            "timestamp": velocity_start.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": wire_amount,
            "transaction_type": "Wire",
            "channel": "Online",
            "destination_account": f"ACC-{v_id}-IN",
            "country_code": random.choice(["KY", "PA", "AE"])
        })
        tx_id_counter += 1

        n_withdrawals = random.randint(4, 8)
        for m in range(n_withdrawals):
            tx_time = velocity_start + timedelta(minutes=15 * (m + 1))
            transactions.append({
                "transaction_id": f"TX-{tx_id_counter}",
                "customer_id": v_id,
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": round(wire_amount / n_withdrawals * 0.95, 2),
                "transaction_type": "Withdrawal",
                "channel": "Branch",
                "destination_account": f"ACC-SHELL-{m}",
                "country_code": "PA"
            })
            tx_id_counter += 1

    df_transactions = pd.DataFrame(transactions)
    df_transactions["timestamp"] = pd.to_datetime(df_transactions["timestamp"])
    df_transactions = df_transactions.sort_values(by="timestamp").reset_index(drop=True)
    df_transactions["timestamp"] = df_transactions["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV
    cust_path = os.path.join(data_dir, "customers.csv")
    tx_path = os.path.join(data_dir, "transactions.csv")
    df_customers.to_csv(cust_path, index=False)
    df_transactions.to_csv(tx_path, index=False)

    print(f"✅ Generated {len(df_customers):,} customers at {cust_path}")
    print(f"✅ Generated {len(df_transactions):,} transactions at {tx_path}")

if __name__ == "__main__":
    generate_aml_dataset()
