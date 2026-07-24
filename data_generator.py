import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_aml_dataset(data_dir="/Users/sterlingsuman/Desktop/projectx/data", num_customers=500, num_transactions=5000):
    os.makedirs(data_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    # 1. Generate Customers
    occupations = ["Software Engineer", "Consultant", "Retail Business", "Import/Export", "Real Estate", "Student", "Retired", "Accountant"]
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

    # Specific Target Customers for Injected AML Patterns:
    # Customer CUST-4521 -> Structuring / Smurfing Pattern
    # Customer CUST-1089 -> Rapid Cash-Out / Velocity Spike Pattern
    # Customer CUST-0088 -> Normal baseline

    transactions = []
    base_date = datetime(2026, 6, 1)

    # Generate baseline transactions for general customers
    tx_id_counter = 10000
    for i in range(num_transactions - 100):
        cust = random.choice(customers)
        tx_time = base_date + timedelta(days=random.randint(0, 50), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        
        # Standard transaction amounts (log-normal distribution around $150)
        amount = round(float(np.random.lognormal(mean=4.8, sigma=1.2)), 2)
        amount = max(5.0, min(amount, 25000.0))
        
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

    # INJECT PATTERN 1: Structuring / Smurfing for CUST-4521
    # 14 cash deposits between $9,100 and $9,950 within a 5-day window to stay below the $10,000 CTR limit
    structuring_start = base_date + timedelta(days=35)
    for k in range(14):
        tx_time = structuring_start + timedelta(days=k // 3, hours=(k % 3) * 3 + random.randint(0, 1))
        amount = round(random.uniform(9100.0, 9950.0), 2)
        transactions.append({
            "transaction_id": f"TX-{tx_id_counter}",
            "customer_id": "CUST-4521",
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "transaction_type": "Deposit",
            "channel": "Branch",
            "destination_account": "ACC-4521-SELF",
            "country_code": "US"
        })
        tx_id_counter += 1

    # Ensure CUST-4521 exists in customer table with specific attributes
    if "CUST-4521" not in df_customers["customer_id"].values:
        df_customers.loc[0, "customer_id"] = "CUST-4521"
        df_customers.loc[0, "customer_name"] = "Target Structuring Subject"
        df_customers.loc[0, "risk_rating"] = "High"

    # INJECT PATTERN 2: Rapid Cash-Out / Velocity Spike for CUST-1089
    # Huge incoming wire ($250,000) followed immediately by 5 rapid withdrawals totaling $245,000 within 2 hours
    velocity_start = base_date + timedelta(days=40, hours=10)
    transactions.append({
        "transaction_id": f"TX-{tx_id_counter}",
        "customer_id": "CUST-1089",
        "timestamp": velocity_start.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": 250000.00,
        "transaction_type": "Wire",
        "channel": "Online",
        "destination_account": "ACC-1089-IN",
        "country_code": "KY" # Cayman Islands
    })
    tx_id_counter += 1

    for m in range(5):
        tx_time = velocity_start + timedelta(minutes=15 * (m + 1))
        transactions.append({
            "transaction_id": f"TX-{tx_id_counter}",
            "customer_id": "CUST-1089",
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": 49000.00,
            "transaction_type": "Withdrawal",
            "channel": "Branch",
            "destination_account": f"ACC-SHELL-{m}",
            "country_code": "PA"
        })
        tx_id_counter += 1

    if "CUST-1089" not in df_customers["customer_id"].values:
        df_customers.loc[1, "customer_id"] = "CUST-1089"
        df_customers.loc[1, "customer_name"] = "Target Velocity Subject"
        df_customers.loc[1, "risk_rating"] = "High"

    df_transactions = pd.DataFrame(transactions)
    df_transactions["timestamp"] = pd.to_datetime(df_transactions["timestamp"])
    df_transactions = df_transactions.sort_values(by="timestamp").reset_index(drop=True)
    df_transactions["timestamp"] = df_transactions["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV
    cust_path = os.path.join(data_dir, "customers.csv")
    tx_path = os.path.join(data_dir, "transactions.csv")
    df_customers.to_csv(cust_path, index=False)
    df_transactions.to_csv(tx_path, index=False)

    print(f"Generated {len(df_customers)} customers at {cust_path}")
    print(f"Generated {len(df_transactions)} transactions at {tx_path}")

if __name__ == "__main__":
    generate_aml_dataset()
