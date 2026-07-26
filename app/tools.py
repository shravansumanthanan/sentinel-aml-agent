import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest

# Centralized FATF High-Risk Jurisdictions (Grey / Blacklist)
HIGH_RISK_JURISDICTIONS: List[str] = ["KY", "PA", "AE"]

class EDATool:
    """Performs automated exploratory data analysis and summary statistics on loaded datasets."""
    def run(self, df_tx: pd.DataFrame, df_cust: pd.DataFrame) -> Dict[str, Any]:
        total_tx = len(df_tx)
        total_volume = float(df_tx["amount"].sum())
        avg_tx = float(df_tx["amount"].mean())
        max_tx = float(df_tx["amount"].max())
        min_tx = float(df_tx["amount"].min())
        unique_customers = int(df_tx["customer_id"].nunique())
        
        tx_type_dist = df_tx["transaction_type"].value_counts().to_dict()
        channel_dist = df_tx["channel"].value_counts().to_dict()
        risk_dist = df_cust["risk_rating"].value_counts().to_dict() if "risk_rating" in df_cust.columns else {}
        jurisdiction_vol = df_tx.groupby("country_code")["amount"].sum().round(2).sort_values(ascending=False).head(8).to_dict() if "country_code" in df_tx.columns else {}
        
        top_cust = df_tx.groupby("customer_id")["amount"].agg(["sum", "count"]).reset_index()
        top_cust = top_cust.sort_values(by="sum", ascending=False).head(5).to_dict(orient="records")

        return {
            "summary": {
                "total_transactions": total_tx,
                "total_volume": round(total_volume, 2),
                "average_amount": round(avg_tx, 2),
                "max_amount": round(max_tx, 2),
                "min_amount": round(min_tx, 2),
                "unique_customers": unique_customers
            },
            "distributions": {
                "transaction_type": tx_type_dist,
                "channel": channel_dist,
                "customer_risk_ratings": risk_dist,
                "jurisdiction_volumes": jurisdiction_vol
            },
            "top_volume_customers": top_cust
        }


class AMLFeatureEngTool:
    """Engineers AML features dynamically from transaction dataset distributions."""
    def run(self, df_tx: pd.DataFrame, time_window_days: int = 30) -> pd.DataFrame:
        df = df_tx.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Determine dynamic structuring threshold relative to dataset max transaction amount
        max_amt = df["amount"].max()
        ctr_limit = 10000.0 if max_amt >= 10000.0 else max_amt * 0.95
        lower_struct_limit = ctr_limit * 0.90
        upper_struct_limit = ctr_limit * 0.999

        # 1. Customer level aggregation
        cust_stats = df.groupby("customer_id").agg(
            total_tx_count=("transaction_id", "count"),
            total_tx_volume=("amount", "sum"),
            avg_amount=("amount", "mean"),
            std_amount=("amount", "std"),
            max_amount=("amount", "max")
        ).reset_index()
        cust_stats["std_amount"] = cust_stats["std_amount"].fillna(0)

        # 2. Dynamic Structuring Band Count (90% to 99.9% of CTR threshold)
        struct_band = df[(df["amount"] >= lower_struct_limit) & (df["amount"] <= upper_struct_limit)]
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

        # 5. High-Risk Jurisdiction Feature (FATF Grey/Blacklist)
        hr_txs = df[df["country_code"].isin(HIGH_RISK_JURISDICTIONS)]
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
    Data-Driven Hybrid Anomaly Detector combining an ML scorer (supervised XGBoost
    or unsupervised IsolationForest) with Dataset-Dynamic Rule Evaluations.

    When `use_precomputed_ml=True` the caller has already injected an `ml_score`
    column (values 0-100) produced by SupervisedAMLClassifier, so this tool skips
    refitting its own IsolationForest and uses that column directly.
    """
    def run(self, df_features: pd.DataFrame, use_precomputed_ml: bool = False) -> pd.DataFrame:
        df = df_features.copy()

        feature_cols = [
            "total_tx_count", "total_tx_volume", "avg_amount",
            "structuring_count", "rapid_cashout_count",
            "high_risk_country_tx_count", "distinct_destination_accounts"
        ]
        X = df[feature_cols].fillna(0)

        if use_precomputed_ml and "ml_score" in df.columns:
            # Use the supervised/unsupervised score already computed by SupervisedAMLClassifier
            pass  # ml_score column already present — nothing to do
        else:
            # Fallback: fit an inline IsolationForest (original behaviour)
            volume_95 = df["total_tx_volume"].quantile(0.95)
            outlier_ratio = (df["total_tx_volume"] > volume_95).mean()
            contamination = float(np.clip(outlier_ratio, 0.02, 0.15))

            iso_model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            iso_model.fit_predict(X)
            scores = iso_model.decision_function(X)

            score_range = scores.max() - scores.min()
            if score_range > 0:
                df["ml_score"] = np.round((scores.max() - scores) / score_range * 100, 1)
            else:
                df["ml_score"] = 50.0

        # Rule Engine (always runs regardless of ML source)
        df["rule_structuring"] = df["structuring_count"] >= 3
        df["rule_velocity"] = df["rapid_cashout_count"] >= 1
        vol_threshold = df["total_tx_volume"].quantile(0.85)
        df["rule_high_volume"] = (df["total_tx_volume"] >= vol_threshold) & (df["structuring_count"] >= 1)
        df["rule_high_risk_country"] = (df["high_risk_country_volume"] > 0.0)
        df["rule_smurfing_fan_in"] = (df["distinct_destination_accounts"] >= 3) & (df["total_tx_count"] >= 4)

        df["rule_hits_count"] = (
            df["rule_structuring"].astype(int) +
            df["rule_velocity"].astype(int) +
            df["rule_high_volume"].astype(int) +
            df["rule_high_risk_country"].astype(int) +
            df["rule_smurfing_fan_in"].astype(int)
        )

        # Composite score: ML 40% + Rule density 30% + Structuring 15% + Jurisdiction 15%
        rule_score = (df["rule_hits_count"] / 5.0) * 100.0
        struct_signal = np.clip((df["structuring_count"] / 5.0) * 100.0, 0, 100)
        hr_signal = np.clip((df["high_risk_country_tx_count"] / 3.0) * 100.0, 0, 100)

        composite = (df["ml_score"] * 0.40) + (rule_score * 0.30) + (struct_signal * 0.15) + (hr_signal * 0.15)
        df["composite_risk_score"] = np.clip(composite, 0, 100).round(1)

        return df


class RiskClassifierTool:
    """Categorizes composite scores dynamically into High, Medium, and Low Risk categories."""
    def run(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        df = df_scored.copy()
        
        # Calculate dynamic risk quantiles from dataset composite score distribution
        p75 = df["composite_risk_score"].quantile(0.75)
        p40 = df["composite_risk_score"].quantile(0.40)

        def classify(row):
            score = row["composite_risk_score"]
            if score >= p75 or row["rule_hits_count"] >= 2:
                return "HIGH", "REPORT (File FinCEN SAR)"
            elif score >= p40 or row["structuring_count"] >= 1 or row["rule_high_risk_country"]:
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
        df_cust_txs = df_tx[df_tx["customer_id"] == customer_id].sort_values(by="timestamp", ascending=False)
        df_clean_txs = df_cust_txs.replace([np.inf, -np.inf], np.nan)
        obj_cols = df_clean_txs.select_dtypes(include=["object", "string"]).columns
        num_cols = df_clean_txs.select_dtypes(exclude=["object", "string"]).columns
        df_clean_txs[obj_cols] = df_clean_txs[obj_cols].fillna("")
        df_clean_txs[num_cols] = df_clean_txs[num_cols].fillna(0)
        cust_txs = df_clean_txs.to_dict(orient="records")
        
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
    Scenario Stress Tester: Recalculates false positives and threat deltas dynamically
    when an analyst adjusts the structuring threshold amount.
    """
    def run(self, df_tx: pd.DataFrame, df_features: pd.DataFrame, lower_bound: float = 8500.0, upper_bound: float = 9999.0) -> Dict[str, Any]:
        df = df_tx.copy()
        band_txs = df[(df["amount"] >= lower_bound) & (df["amount"] <= upper_bound)]
        cust_band_counts = band_txs.groupby("customer_id").size().reset_index(name="new_struct_count")
        
        merged = pd.merge(df_features, cust_band_counts, on="customer_id", how="left")
        merged["new_struct_count"] = merged["new_struct_count"].fillna(0).astype(int)
        
        baseline_flagged = (df_features["structuring_count"] >= 3).sum()
        new_flagged = (merged["new_struct_count"] >= 3).sum()
        
        delta = int(new_flagged - baseline_flagged)

        return {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "baseline_flagged_customers": int(baseline_flagged),
            "new_flagged_customers": int(new_flagged),
            "customer_count_delta": delta,
            "interpretation": f"Lowering structuring bound to ${lower_bound:,.2f} flagged {delta} additional subjects from the dataset." if delta > 0 else "No change in flagged subjects under this threshold."
        }


class SARGeneratorTool:
    """
    Regulatory FinCEN SAR Narrative Generator.
    Produces formal human-readable FinCEN Suspicious Activity Report narratives directly from subject data.
    """
    def generate_sar(
        self,
        customer_id: str,
        cust_data: Dict[str, Any],
        risk_profile: Dict[str, Any],
        tx_list: List[Dict[str, Any]],
        model_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        name = cust_data.get("customer_name", customer_id)
        country = cust_data.get("country", "Unknown")
        occ = cust_data.get("occupation", "Unknown")
        
        risk_score = risk_profile.get("composite_risk_score", 0)
        ml_score = risk_profile.get("ml_score", 0)
        struct_count = risk_profile.get("structuring_count", 0)
        total_vol = risk_profile.get("total_tx_volume", 0)
        rapid_cashout = risk_profile.get("rapid_cashout_count", 0)
        hr_vol = risk_profile.get("high_risk_country_volume", 0)

        # ML Model Attribution line
        model_type = "IsolationForest (Unsupervised)"
        model_attr = "Unsupervised Anomaly Isolation"
        if model_info and model_info.get("is_supervised"):
            mtype = model_info.get("model_type", "Supervised ML")
            auc = model_info.get("auc_roc", "N/A")
            f1 = model_info.get("f1_score", "N/A")
            top_feats = model_info.get("feature_importances", [])[:3]
            feat_str = ", ".join([f"{f['feature']} ({f['importance']:.2f})" for f in top_feats]) if top_feats else "None"
            model_type = f"{mtype} (Supervised, AUC-ROC: {auc}, F1: {f1})"
            model_attr = f"Supervised ML Probabilities + IsolationForest Hybrid. Key Signals: {feat_str}"

        narrative = f"""
================================================================================
FINCEN SUSPICIOUS ACTIVITY REPORT (SAR) NARRATIVE
Subject ID: {customer_id} | Name: {name}
Occupation: {occ} | Jurisdiction: {country}
Risk Rating: HIGH | Composite Risk Score: {risk_score}/100 | Local ML Score: {ml_score}/100
Model Architecture: {model_type}
================================================================================

EXECUTIVE SUMMARY:
Compliance Division is filing this Suspicious Activity Report (SAR) regarding anomalous transactional activity conducted by Subject {name} ({customer_id}). The Subject engaged in structured cash transactions designed to evade Currency Transaction Reporting (CTR) requirements, coupled with anomalous transaction velocity.

DETAILED PATTERN ANALYSIS:
1. STRUCTURING PATTERN:
   The Subject conducted {struct_count} discrete transactions in the structuring threshold band, accumulating a total volume of ${total_vol:,.2f}. The pattern indicates deliberate structuring to remain below statutory reporting limits.

2. RAPID VELOCITY & CASHOUT SPIKES:
   The subject's account exhibited {rapid_cashout} rapid cash-out events (incoming deposit/wire followed by immediate withdrawal within 120 minutes), presenting high risk of money laundering layering.

3. HIGH-RISK JURISDICTION EXPOSURE:
   Transferred ${hr_vol:,.2f} involving FATF high-risk jurisdictions (KY, PA, AE), triggering mandatory compliance escalation.

4. MACHINE LEARNING ANOMALY ATTRIBUTION:
   {model_attr}

RECOMMENDED REGULATORY ACTION:
- Status: MANDATORY REPORTING (SAR FILING REQUIRED)
- Action: Escalate to Anti-Money Laundering Officer (AMLO) for account freeze & 30-day monitoring.
================================================================================
""".strip()
        return narrative
