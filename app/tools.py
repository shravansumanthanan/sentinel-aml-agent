import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import IsolationForest

class EDATool:
    """Performs automated exploratory data analysis and summary statistics."""
    def run(self, df_tx: pd.DataFrame, df_cust: pd.DataFrame) -> Dict[str, Any]:
        total_tx = len(df_tx)
        total_volume = float(df_tx["amount"].sum())
        avg_tx = float(df_tx["amount"].mean())
        max_tx = float(df_tx["amount"].max())
        unique_customers = int(df_tx["customer_id"].nunique())
        
        tx_type_dist = df_tx["transaction_type"].value_counts().to_dict()
        channel_dist = df_tx["channel"].value_counts().to_dict()
        risk_dist = df_cust["risk_rating"].value_counts().to_dict()
        
        top_cust = df_tx.groupby("customer_id")["amount"].agg(["sum", "count"]).reset_index()
        top_cust = top_cust.sort_values(by="sum", ascending=False).head(5).to_dict(orient="records")

        return {
            "summary": {
                "total_transactions": total_tx,
                "total_volume": round(total_volume, 2),
                "average_amount": round(avg_tx, 2),
                "max_amount": round(max_tx, 2),
                "unique_customers": unique_customers
            },
            "distributions": {
                "transaction_type": tx_type_dist,
                "channel": channel_dist,
                "customer_risk_ratings": risk_dist
            },
            "top_volume_customers": top_cust
        }


class AMLFeatureEngTool:
    """Engineers AML features: rolling velocity, structuring ratio, high-risk country volume, fan-in counts."""
    def run(self, df_tx: pd.DataFrame, time_window_days: int = 30) -> pd.DataFrame:
        df = df_tx.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # 1. Customer level aggregation
        cust_stats = df.groupby("customer_id").agg(
            total_tx_count=("transaction_id", "count"),
            total_tx_volume=("amount", "sum"),
            avg_amount=("amount", "mean"),
            std_amount=("amount", "std"),
            max_amount=("amount", "max")
        ).reset_index()
        cust_stats["std_amount"] = cust_stats["std_amount"].fillna(0)

        # 2. Structuring Band Count ($9,000 - $9,999)
        struct_band = df[(df["amount"] >= 9000.0) & (df["amount"] <= 9999.0)]
        struct_counts = struct_band.groupby("customer_id").size().reset_index(name="structuring_count")

        # 3. Merge features back
        features = pd.merge(cust_stats, struct_counts, on="customer_id", how="left")
        features["structuring_count"] = features["structuring_count"].fillna(0).astype(int)
        features["structuring_ratio"] = features["structuring_count"] / features["total_tx_count"]

        # 4. Rapid Cash-out Velocity feature
        df_sorted = df.sort_values(by=["customer_id", "timestamp"])
        df_sorted["prev_type"] = df_sorted.groupby("customer_id")["transaction_type"].shift(1)
        df_sorted["prev_time"] = df_sorted.groupby("customer_id")["timestamp"].shift(1)
        df_sorted["time_diff_hours"] = (df_sorted["timestamp"] - df_sorted["prev_time"]).dt.total_seconds() / 3600.0
        
        rapid_withdrawals = df_sorted[
            (df_sorted["transaction_type"] == "Withdrawal") & 
            (df_sorted["prev_type"].isin(["Wire", "Deposit"])) & 
            (df_sorted["time_diff_hours"] <= 2.0)
        ]
        rapid_counts = rapid_withdrawals.groupby("customer_id").size().reset_index(name="rapid_cashout_count")
        
        features = pd.merge(features, rapid_counts, on="customer_id", how="left")
        features["rapid_cashout_count"] = features["rapid_cashout_count"].fillna(0).astype(int)

        # 5. High-Risk Jurisdiction Feature (FATF Grey/Blacklist: KY, PA, AE)
        high_risk_jurisdictions = ["KY", "PA", "AE"]
        hr_txs = df[df["country_code"].isin(high_risk_jurisdictions)]
        hr_counts = hr_txs.groupby("customer_id").size().reset_index(name="high_risk_country_tx_count")
        hr_vol = hr_txs.groupby("customer_id")["amount"].sum().reset_index(name="high_risk_country_volume")

        features = pd.merge(features, hr_counts, on="customer_id", how="left")
        features = pd.merge(features, hr_vol, on="customer_id", how="left")
        features["high_risk_country_tx_count"] = features["high_risk_country_tx_count"].fillna(0).astype(int)
        features["high_risk_country_volume"] = features["high_risk_country_volume"].fillna(0.0)

        # 6. Smurfing Multi-Account Fan-In Count
        unique_dests = df.groupby("customer_id")["destination_account"].nunique().reset_index(name="distinct_destination_accounts")
        features = pd.merge(features, unique_dests, on="customer_id", how="left")
        features["distinct_destination_accounts"] = features["distinct_destination_accounts"].fillna(1).astype(int)

        return features


class HybridAnomalyTool:
    """
    Hybrid Anomaly Detector combining Isolation Forest (Unsupervised ML)
    with Domain Rule Engines (Structuring, Rapid Velocity, High-Risk Country, & Fan-In rules).
    """
    def run(self, df_features: pd.DataFrame, structuring_threshold: float = 9000.0) -> pd.DataFrame:
        df = df_features.copy()
        
        # 1. Machine Learning Outlier Detection (Isolation Forest)
        feature_cols = [
            "total_tx_count", "total_tx_volume", "avg_amount", 
            "structuring_count", "rapid_cashout_count", 
            "high_risk_country_tx_count", "distinct_destination_accounts"
        ]
        X = df[feature_cols].fillna(0)
        
        iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        df["iso_forest_anomaly_score"] = -iso_model.fit_predict(X)
        scores = iso_model.decision_function(X)
        df["ml_score"] = np.round((scores.max() - scores) / (scores.max() - scores.min() + 1e-6) * 100, 1)

        # 2. Rule Engine Evaluation
        # Rule 1: Structuring (5+ transactions in Structuring band)
        df["rule_structuring"] = df["structuring_count"] >= 5
        
        # Rule 2: Velocity Spike (Rapid cashouts >= 2)
        df["rule_velocity"] = df["rapid_cashout_count"] >= 2

        # Rule 3: High Volume Cash Trap
        df["rule_high_volume"] = (df["total_tx_volume"] > 100000.0) & (df["structuring_count"] >= 3)

        # Rule 4: High-Risk Jurisdiction Wire Rule (Transfers > $20k with KY/PA/AE)
        df["rule_high_risk_country"] = (df["high_risk_country_volume"] >= 20000.0)

        # Rule 5: Smurfing Fan-In Rule (4+ distinct accounts used in rapid pattern)
        df["rule_smurfing_fan_in"] = (df["distinct_destination_accounts"] >= 4) & (df["total_tx_count"] >= 5)

        # Combine Rule Hits
        df["rule_hits_count"] = (
            df["rule_structuring"].astype(int) + 
            df["rule_velocity"].astype(int) + 
            df["rule_high_volume"].astype(int) + 
            df["rule_high_risk_country"].astype(int) + 
            df["rule_smurfing_fan_in"].astype(int)
        )

        # Composite Risk Score Calculation (Hybrid)
        df["composite_risk_score"] = np.clip(
            df["ml_score"] * 0.35 + 
            df["rule_hits_count"] * 25.0 + 
            df["structuring_count"] * 4.0 + 
            df["high_risk_country_tx_count"] * 10.0, 
            0, 100
        ).round(1)

        return df


class RiskClassifierTool:
    """Categorizes score into Low, Medium, High Risk and assigns Escalation Action."""
    def run(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        df = df_scored.copy()
        
        def classify(row):
            score = row["composite_risk_score"]
            if score >= 75.0 or row["rule_hits_count"] >= 2:
                return "HIGH", "REPORT (File FinCEN SAR)"
            elif score >= 45.0 or row["structuring_count"] >= 3 or row["rule_high_risk_country"]:
                return "MEDIUM", "FLAG FOR REVIEW (Senior Analyst)"
            else:
                return "LOW", "MONITOR (Routine Check)"

        results = df.apply(classify, axis=1)
        df["risk_level"] = [r[0] for r in results]
        df["recommended_action"] = [r[1] for r in results]
        return df


class SingleEntityLookupTool:
    """Performs focused single customer inspection."""
    def run(self, customer_id: str, df_tx: pd.DataFrame, df_cust: pd.DataFrame, df_classified: pd.DataFrame) -> Dict[str, Any]:
        cust_info = df_cust[df_cust["customer_id"] == customer_id]
        if cust_info.empty:
            return {"found": False, "message": f"Customer {customer_id} not found in database."}
        
        cust_dict = cust_info.iloc[0].to_dict()
        cust_txs = df_tx[df_tx["customer_id"] == customer_id].sort_values(by="timestamp", ascending=False).to_dict(orient="records")
        
        risk_info = df_classified[df_classified["customer_id"] == customer_id]
        risk_dict = risk_info.iloc[0].to_dict() if not risk_info.empty else {}

        return {
            "found": True,
            "customer": cust_dict,
            "risk_profile": risk_dict,
            "transaction_history": cust_txs
        }


class ThresholdStressTestTool:
    """
    (Lens 3 Feature) Scenario Stress Tester: Recalculates false positives and threat deltas
    when an analyst adjusts the structuring threshold amount.
    """
    def run(self, df_tx: pd.DataFrame, df_features: pd.DataFrame, lower_bound: float = 8500.0, upper_bound: float = 9999.0) -> Dict[str, Any]:
        df = df_tx.copy()
        band_txs = df[(df["amount"] >= lower_bound) & (df["amount"] <= upper_bound)]
        cust_band_counts = band_txs.groupby("customer_id").size().reset_index(name="new_struct_count")
        
        merged = pd.merge(df_features, cust_band_counts, on="customer_id", how="left")
        merged["new_struct_count"] = merged["new_struct_count"].fillna(0).astype(int)
        
        baseline_flagged = (df_features["structuring_count"] >= 5).sum()
        new_flagged = (merged["new_struct_count"] >= 5).sum()
        
        delta = int(new_flagged - baseline_flagged)

        return {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "baseline_flagged_customers": int(baseline_flagged),
            "new_flagged_customers": int(new_flagged),
            "customer_count_delta": delta,
            "interpretation": f"Lowering the lower structuring bound to ${lower_bound:,.2f} identified {delta} additional high-risk subjects." if delta > 0 else "No change in flagged subjects under this threshold."
        }


class SARGeneratorTool:
    """
    (Lens 2 Feature) Regulatory FinCEN SAR Narrative Generator.
    Produces formal human-readable FinCEN Suspicious Activity Report narratives.
    """
    def generate_sar(self, customer_id: str, cust_data: Dict[str, Any], risk_profile: Dict[str, Any], tx_list: List[Dict[str, Any]]) -> str:
        name = cust_data.get("customer_name", customer_id)
        country = cust_data.get("country", "Unknown")
        occ = cust_data.get("occupation", "Unknown")
        
        risk_score = risk_profile.get("composite_risk_score", 0)
        struct_count = risk_profile.get("structuring_count", 0)
        total_vol = risk_profile.get("total_tx_volume", 0)
        rapid_cashout = risk_profile.get("rapid_cashout_count", 0)
        hr_vol = risk_profile.get("high_risk_country_volume", 0)

        narrative = f"""
================================================================================
FINCEN SUSPICIOUS ACTIVITY REPORT (SAR) NARRATIVE
Subject ID: {customer_id} | Name: {name}
Occupation: {occ} | Jurisdiction: {country}
Risk Rating: HIGH | Composite ML Risk Score: {risk_score}/100
================================================================================

EXECUTIVE SUMMARY:
Financial Institution Compliance Division is filing this Suspicious Activity Report (SAR) regarding suspicious transactional activity conducted by Subject {name} ({customer_id}) between June 2026 and July 2026. The Subject engaged in structured cash transactions designed to evade 31 C.F.R. § 1010.311 Currency Transaction Reporting (CTR) requirements, coupled with anomalous transaction velocity.

DETAILED PATTERN ANALYSIS:
1. STRUCTURING / SMURFING PATTERN:
   The Subject conducted {struct_count} discrete transactions falling within the $9,000.00 to $9,999.00 threshold band, accumulating a total volume of ${total_vol:,.2f}. The transactions occurred in rapid succession at branch teller windows. The pattern indicates deliberate structuring to remain below the mandatory $10,000.00 CTR reporting threshold.

2. RAPID VELOCITY & CASHOUT SPIKES:
   The subject's account exhibited {rapid_cashout} rapid cash-out events (incoming wire/deposit followed by immediate cash withdrawal within 120 minutes), presenting high risk of money laundering layering.

3. HIGH-RISK JURISDICTION EXPOSURE:
   Transferred ${hr_vol:,.2f} involving FATF high-risk jurisdictions (Cayman Islands, Panama, UAE), triggering mandatory compliance review.

RECOMMENDED REGULATORY ACTION:
- Status: MANDATORY REPORTING (SAR FILING REQUIRED)
- Immediate Action: Escalate to Anti-Money Laundering Officer (AMLO) for account freeze & 30-day monitoring.
================================================================================
""".strip()
        return narrative
