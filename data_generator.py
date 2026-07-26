import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_aml_dataset(data_dir=None, num_customers=1000, num_transactions=15000):
    """
    Generates an institutional-grade synthetic AML dataset containing realistic
    customer profiles and transaction ledgers with embedded money laundering typologies:
      1. Structuring / Smurfing (Cash deposits under $10k CTR threshold)
      2. Rapid Cash-out Velocity Spikes (Wire in → instant multi-part withdrawal)
      3. Offshore FATF High-Risk Jurisdiction Funneling (KY, PA, AE)
    
    Includes ground-truth `is_laundering` labels (0 = Legitimate, 1 = Suspicious).
    """
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")

    os.makedirs(data_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    print(f"⚡ Generating institutional AML dataset ({num_customers:,} customers, {num_transactions:,} transactions)...")

    # 1. Realistic Customer Name Generators
    first_names = ["Marcus", "Elena", "Sophia", "David", "Aisha", "Chen", "Carlos", "Fatima", "Viktor", "Amara", "Dmitri", "Sarah", "Oliver", "Maya", "Alexander", "Isabella", "Lucas", "Zara", "Julian", "Hannah"]
    last_names = ["Vance", "Rostova", "Chen", "Al-Mansoor", "Silva", "Kowalski", "O'Connor", "Patel", "Tanaka", "Dubois", "Müller", "Siddiqui", "Zhang", "Novak", "Wright", "Kovacs", "Rossi", "Benali", "Volkov", "Nakamoto"]
    corp_prefixes = ["Apex Global", "Vanguard", "Horizon", "Caspian", "Pinnacle", "Titanium", "Meridian", "Atlas", "Omega", "Biscayne"]
    corp_suffixes = ["Trading LLC", "Logistics Inc", "Holdings", "Capital Partners", "Import Export Ltd", "Enterprises", "Financial Co", "Investments Group"]

    occupations = [
        "Software Engineer", "Consultant", "Retail Business", "Import/Export",
        "Real Estate", "Student", "Retired", "Accountant", "Medical Doctor",
        "Financial Analyst", "Attorney", "Architect", "Logistics Manager"
    ]
    countries = ["US", "CA", "GB", "DE", "FR", "SG", "AE", "KY", "PA"]
    kyc_statuses = ["Verified", "Verified", "Verified", "Verified", "Enhanced", "Pending"]
    
    customers = []
    suspicious_customer_ids = set()

    for i in range(1, num_customers + 1):
        cid = f"CUST-{i:04d}"
        
        # 15% corporate entities, 85% individual customers
        if random.random() < 0.15:
            name = f"{random.choice(corp_prefixes)} {random.choice(corp_suffixes)}"
            occ = "Import/Export" if "Import" in name or "Logistics" in name else "Financial Services"
        else:
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            occ = random.choice(occupations)

        # Baseline initial risk distribution
        r_val = random.random()
        if r_val < 0.70:
            init_risk = "Low"
        elif r_val < 0.90:
            init_risk = "Medium"
        else:
            init_risk = "High"

        open_date = (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 900))).strftime("%Y-%m-%d")
        
        customers.append({
            "customer_id": cid,
            "customer_name": name,
            "risk_rating": init_risk,
            "account_opened_date": open_date,
            "kyc_status": random.choice(kyc_statuses),
            "occupation": occ,
            "country": random.choice(countries),
            "is_laundering": 0 # Default, updated after typology injection
        })

    df_customers = pd.DataFrame(customers)

    transactions = []
    base_date = datetime(2026, 6, 1)
    tx_id_counter = 10000

    # 2. Baseline Legitimate Transactions (~92% of ledger)
    num_baseline = int(num_transactions * 0.92)
    for _ in range(num_baseline):
        cust = random.choice(customers)
        tx_time = base_date + timedelta(days=random.randint(0, 60), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        
        # Lognormal distribution for natural consumer & business payments
        if "LLC" in cust["customer_name"] or "Inc" in cust["customer_name"] or "Ltd" in cust["customer_name"]:
            amount = round(float(np.random.lognormal(mean=7.5, sigma=1.0)), 2) # ~$1,800 to $25,000
            amount = max(100.0, min(amount, 85000.0))
            tx_type = random.choice(["Wire", "Transfer", "Deposit", "Withdrawal"])
            channel = random.choice(["Online", "Branch"])
        else:
            amount = round(float(np.random.lognormal(mean=4.8, sigma=1.2)), 2) # ~$20 to $1,500
            amount = max(5.0, min(amount, 8500.0))
            tx_type = random.choice(["Deposit", "Transfer", "Withdrawal", "Wire"])
            channel = random.choice(["Online", "ATM", "Mobile", "Branch"])

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
            "country_code": c_code,
            "is_laundering": 0
        })
        tx_id_counter += 1

    # 3. Inject Typology 1: Structuring / Smurfing Clusters (Cash deposits between $9,100 and $9,950)
    structuring_cids = ["CUST-0015", "CUST-0042", "CUST-0150", "CUST-0420", "CUST-0899", "CUST-1089"]
    for s_cid in structuring_cids:
        if s_cid in df_customers["customer_id"].values:
            suspicious_customer_ids.add(s_cid)
            struct_start = base_date + timedelta(days=random.randint(5, 45))
            n_deposits = random.randint(8, 16)
            for k in range(n_deposits):
                tx_time = struct_start + timedelta(days=k // 4, hours=(k % 4) * 3 + random.randint(0, 2))
                amount = round(random.uniform(9120.0, 9940.0), 2)
                transactions.append({
                    "transaction_id": f"TX-{tx_id_counter}",
                    "customer_id": s_cid,
                    "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": amount,
                    "transaction_type": "Deposit",
                    "channel": "Branch",
                    "destination_account": f"ACC-{s_cid}-SELF",
                    "country_code": "US",
                    "is_laundering": 1
                })
                tx_id_counter += 1

    # 4. Inject Typology 2: Rapid Cash-Out Velocity Spikes (Wire In → Instant Branch Withdrawals)
    velocity_cids = ["CUST-0001", "CUST-0088", "CUST-0310", "CUST-0550", "CUST-0720"]
    for v_cid in velocity_cids:
        if v_cid in df_customers["customer_id"].values:
            suspicious_customer_ids.add(v_cid)
            vel_start = base_date + timedelta(days=random.randint(10, 50), hours=random.randint(9, 13))
            wire_amount = round(random.uniform(180000.0, 450000.0), 2)
            
            # Incoming Wire from offshore tax haven
            transactions.append({
                "transaction_id": f"TX-{tx_id_counter}",
                "customer_id": v_cid,
                "timestamp": vel_start.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": wire_amount,
                "transaction_type": "Wire",
                "channel": "Online",
                "destination_account": f"ACC-{v_cid}-VAULT",
                "country_code": random.choice(["KY", "PA", "AE"]),
                "is_laundering": 1
            })
            tx_id_counter += 1

            # Rapid sequential cash withdrawals within 90 minutes
            n_withdrawals = random.randint(4, 7)
            for m in range(n_withdrawals):
                tx_time = vel_start + timedelta(minutes=15 * (m + 1))
                transactions.append({
                    "transaction_id": f"TX-{tx_id_counter}",
                    "customer_id": v_cid,
                    "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": round((wire_amount / n_withdrawals) * 0.96, 2),
                    "transaction_type": "Withdrawal",
                    "channel": "Branch",
                    "destination_account": f"ACC-SHELL-{m+1}",
                    "country_code": "PA",
                    "is_laundering": 1
                })
                tx_id_counter += 1

    # 5. Inject Typology 3: FATF High-Risk Offshore Funneling (KY, PA, AE)
    offshore_cids = ["CUST-0005", "CUST-0120", "CUST-0240", "CUST-0600"]
    for o_cid in offshore_cids:
        if o_cid in df_customers["customer_id"].values:
            suspicious_customer_ids.add(o_cid)
            off_start = base_date + timedelta(days=random.randint(5, 55))
            n_transfers = random.randint(6, 12)
            for j in range(n_transfers):
                tx_time = off_start + timedelta(days=j * 2, hours=random.randint(1, 10))
                amount = round(random.uniform(75000.0, 220000.0), 2)
                transactions.append({
                    "transaction_id": f"TX-{tx_id_counter}",
                    "customer_id": o_cid,
                    "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": amount,
                    "transaction_type": "Wire",
                    "channel": "Online",
                    "destination_account": f"ACC-OFFSHORE-{j+100}",
                    "country_code": random.choice(["KY", "PA", "AE"]),
                    "is_laundering": 1
                })
                tx_id_counter += 1

    # Update customer-level is_laundering flag
    df_customers["is_laundering"] = df_customers["customer_id"].apply(lambda x: 1 if x in suspicious_customer_ids else 0)

    # Sort transactions chronologically
    df_transactions = pd.DataFrame(transactions)
    df_transactions["timestamp"] = pd.to_datetime(df_transactions["timestamp"])
    df_transactions = df_transactions.sort_values(by="timestamp").reset_index(drop=True)
    df_transactions["timestamp"] = df_transactions["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save CSV files
    cust_path = os.path.join(data_dir, "customers.csv")
    tx_path = os.path.join(data_dir, "transactions.csv")
    df_customers.to_csv(cust_path, index=False)
    df_transactions.to_csv(tx_path, index=False)

    laundering_tx_count = (df_transactions["is_laundering"] == 1).sum()
    laundering_cust_count = (df_customers["is_laundering"] == 1).sum()

    print(f"✅ Generated {len(df_customers):,} customers ({laundering_cust_count} flagged) at {cust_path}")
    print(f"✅ Generated {len(df_transactions):,} transactions ({laundering_tx_count} laundering flagged) at {tx_path}")

if __name__ == "__main__":
    generate_aml_dataset()
