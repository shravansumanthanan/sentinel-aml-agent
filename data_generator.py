import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_aml_dataset(data_dir=None, num_customers=1000, num_transactions=50000, start_date_str="2021-08-01", end_date_str="2026-08-01"):
    """
    Generates an institutional-grade synthetic AML dataset containing realistic
    customer profiles and transaction ledgers spanning 5 years (2021-2026) with embedded
    money laundering typologies:
      1. Structuring / Smurfing (Cash deposits under $10k CTR threshold across multiple temporal waves)
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

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    total_days = (end_date - start_date).days

    print(f"⚡ Generating institutional 5-Year AML dataset ({num_customers:,} customers, {num_transactions:,} transactions, span {start_date_str} to {end_date_str})...")

    # 1. Realistic Customer Name & Demographic Generators
    first_names = [
        "Marcus", "Elena", "Sophia", "David", "Aisha", "Chen", "Carlos", "Fatima", "Viktor", "Amara", 
        "Dmitri", "Sarah", "Oliver", "Maya", "Alexander", "Isabella", "Lucas", "Zara", "Julian", "Hannah",
        "Liam", "Noah", "Ethan", "Aria", "Mia", "Benjamin", "Charlotte", "Amelia", "Harper", "Evelyn",
        "Abigail", "Emily", "Ella", "Elizabeth", "Camila", "Luna", "Sofia", "Avery", "Mila", "Aria"
    ]
    last_names = [
        "Vance", "Rostova", "Chen", "Al-Mansoor", "Silva", "Kowalski", "O'Connor", "Patel", "Tanaka", "Dubois", 
        "Müller", "Siddiqui", "Zhang", "Novak", "Wright", "Kovacs", "Rossi", "Benali", "Volkov", "Nakamoto",
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
        "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
    ]
    corp_prefixes = [
        "Apex Global", "Vanguard", "Horizon", "Caspian", "Pinnacle", "Titanium", "Meridian", "Atlas", "Omega", "Biscayne",
        "Zephyr", "Nexus", "Aegis", "Solstice", "Helios", "Vortex", "Quantum", "Hyperion", "Sterling", "Valence"
    ]
    corp_suffixes = [
        "Trading LLC", "Logistics Inc", "Holdings", "Capital Partners", "Import Export Ltd", "Enterprises", 
        "Financial Co", "Investments Group", "International Corp", "Solutions S.A.", "Ventures Ltd", "Global Trading"
    ]

    occupations = [
        "Software Engineer", "Consultant", "Retail Business", "Import/Export",
        "Real Estate", "Student", "Retired", "Accountant", "Medical Doctor",
        "Financial Analyst", "Attorney", "Architect", "Logistics Manager",
        "E-commerce Operator", "Investment Banker", "Art Dealer", "Construction Contractor"
    ]
    countries = ["US", "CA", "GB", "DE", "FR", "SG", "AE", "KY", "PA", "JP", "CH", "AU", "BR", "MX", "IN"]
    kyc_statuses = ["Verified", "Verified", "Verified", "Verified", "Enhanced", "Pending"]
    
    customers = []
    suspicious_customer_ids = set()

    for i in range(1, num_customers + 1):
        cid = f"CUST-{i:04d}"
        
        # 18% corporate entities, 82% individual customers
        if random.random() < 0.18:
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

        # Account opened date between 2019-01-01 and 2025-06-01
        open_days_offset = random.randint(0, 2300)
        open_date_dt = datetime(2019, 1, 1) + timedelta(days=open_days_offset)
        open_date = open_date_dt.strftime("%Y-%m-%d")
        
        customers.append({
            "customer_id": cid,
            "customer_name": name,
            "risk_rating": init_risk,
            "account_opened_date": open_date,
            "account_open_dt": open_date_dt,
            "kyc_status": random.choice(kyc_statuses),
            "occupation": occ,
            "country": random.choice(countries),
            "is_laundering": 0 # Default, updated after typology injection
        })

    df_customers_temp = pd.DataFrame(customers)

    transactions = []
    tx_id_counter = 10000

    # 2. Baseline Legitimate Transactions (~93% of ledger)
    num_baseline = int(num_transactions * 0.93)
    for _ in range(num_baseline):
        cust = random.choice(customers)
        cust_open = cust["account_open_dt"]
        
        # Transaction must occur between max(start_date, cust_open) and end_date
        t_start = max(start_date, cust_open)
        if t_start >= end_date:
            t_start = start_date
            
        day_delta = (end_date - t_start).days
        if day_delta <= 0:
            day_delta = 1
            
        tx_time = t_start + timedelta(days=random.randint(0, day_delta), hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
        
        # Lognormal distribution for natural consumer & business payments
        if any(suffix in cust["customer_name"] for suffix in ["LLC", "Inc", "Ltd", "Corp", "Group"]):
            amount = round(float(np.random.lognormal(mean=7.8, sigma=1.1)), 2) # ~$2,000 to $45,000
            amount = max(150.0, min(amount, 120000.0))
            tx_type = random.choice(["Wire", "Transfer", "Deposit", "Withdrawal"])
            channel = random.choice(["Online", "Branch"])
        else:
            amount = round(float(np.random.lognormal(mean=4.9, sigma=1.2)), 2) # ~$25 to $2,500
            amount = max(10.0, min(amount, 9500.0))
            tx_type = random.choice(["Deposit", "Transfer", "Withdrawal", "Wire", "Payment"])
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

    # 3. Inject Typology 1: Structuring / Smurfing Clusters (Cash deposits $9,100 to $9,950 across multiple temporal waves)
    # Define multi-year structuring clusters
    structuring_wave_targets = [
        # Recent 3-30 Days (July 2026)
        ("CUST-0015", datetime(2026, 7, 20), 12),
        ("CUST-0042", datetime(2026, 7, 10), 15),
        ("CUST-0150", datetime(2026, 7, 1), 10),
        ("CUST-0420", datetime(2026, 7, 15), 14),
        # 30-90 Days (May-June 2026)
        ("CUST-0209", datetime(2026, 5, 15), 11),
        ("CUST-0330", datetime(2026, 6, 1), 12),
        ("CUST-0899", datetime(2026, 5, 20), 9),
        ("CUST-1089", datetime(2026, 6, 12), 16),
        # 90-180 Days (Jan-April 2026)
        ("CUST-0112", datetime(2026, 2, 10), 10),
        ("CUST-0560", datetime(2026, 3, 14), 13),
        # 1 Year (2025)
        ("CUST-0220", datetime(2025, 9, 5), 12),
        ("CUST-0640", datetime(2025, 11, 12), 14),
        ("CUST-0910", datetime(2025, 4, 18), 10),
        # 2-4 Years (2022-2024)
        ("CUST-0050", datetime(2024, 6, 10), 11),
        ("CUST-0380", datetime(2023, 10, 5), 13),
        ("CUST-0715", datetime(2022, 8, 20), 12),
    ]

    for s_cid, wave_start, n_deposits in structuring_wave_targets:
        if s_cid in df_customers_temp["customer_id"].values:
            suspicious_customer_ids.add(s_cid)
            for k in range(n_deposits):
                tx_time = wave_start + timedelta(days=k // 3, hours=(k % 3) * 4 + random.randint(0, 2), minutes=random.randint(0, 59))
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

    # 4. Inject Typology 2: Rapid Cash-Out Velocity Spikes across 5 Years
    velocity_wave_targets = [
        ("CUST-0001", datetime(2026, 7, 18, 10, 0)),
        ("CUST-0088", datetime(2026, 5, 22, 11, 30)),
        ("CUST-0310", datetime(2026, 1, 14, 9, 15)),
        ("CUST-0550", datetime(2025, 10, 8, 14, 0)),
        ("CUST-0720", datetime(2025, 3, 19, 10, 45)),
        ("CUST-0810", datetime(2024, 7, 11, 13, 20)),
        ("CUST-0940", datetime(2023, 11, 4, 11, 10)),
        ("CUST-0105", datetime(2022, 5, 15, 10, 30)),
    ]

    for v_cid, vel_start in velocity_wave_targets:
        if v_cid in df_customers_temp["customer_id"].values:
            suspicious_customer_ids.add(v_cid)
            wire_amount = round(random.uniform(180000.0, 480000.0), 2)
            
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

    # 5. Inject Typology 3: FATF High-Risk Offshore Funneling (KY, PA, AE) across 5 Years
    offshore_wave_targets = [
        ("CUST-0005", datetime(2026, 7, 5), 8),
        ("CUST-0120", datetime(2026, 4, 12), 10),
        ("CUST-0240", datetime(2025, 12, 1), 9),
        ("CUST-0600", datetime(2025, 6, 18), 11),
        ("CUST-0750", datetime(2024, 9, 22), 7),
        ("CUST-0880", datetime(2023, 4, 10), 12),
        ("CUST-0190", datetime(2022, 11, 15), 8),
    ]

    for o_cid, off_start, n_transfers in offshore_wave_targets:
        if o_cid in df_customers_temp["customer_id"].values:
            suspicious_customer_ids.add(o_cid)
            for j in range(n_transfers):
                tx_time = off_start + timedelta(days=j * 3, hours=random.randint(1, 10), minutes=random.randint(0, 59))
                amount = round(random.uniform(75000.0, 260000.0), 2)
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

    # Clean up customer frame (drop temporary account_open_dt column)
    df_customers = df_customers_temp.drop(columns=["account_open_dt"]).copy()
    
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
