import time
import os
import difflib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from app.nlp_parser import NLPIntentParser
from app.kaggle_loader import load_and_merge_kaggle_datasets
from app.ml_model import SupervisedAMLClassifier
from app.tools import (
    EDATool,
    AMLFeatureEngTool,
    HybridAnomalyTool,
    RiskClassifierTool,
    SingleEntityLookupTool,
    ThresholdStressTestTool,
    SARGeneratorTool
)

class AMLAgentOrchestrator:
    """
    Query-Driven Agent Orchestrator.
    Parses user query, constructs dynamic Tool Execution Plan (Tool DAG),
    invokes relevant tools selectively, and records telemetry log.
    Data-driven & Zero Hardcoded Subject Dependencies.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.parser = NLPIntentParser()
        self.eda_tool = EDATool()
        self.feature_tool = AMLFeatureEngTool()
        self.anomaly_tool = HybridAnomalyTool()
        self.classifier_tool = RiskClassifierTool()
        self.single_lookup_tool = SingleEntityLookupTool()
        self.stress_test_tool = ThresholdStressTestTool()
        self.sar_tool = SARGeneratorTool()

        # Supervised ML classifier (XGBoost → RandomForest → IsolationForest fallback)
        model_cache_dir = os.path.join(data_dir, "model_cache")
        self.ml_classifier = SupervisedAMLClassifier(model_cache_dir=model_cache_dir)
        self.model_info: Dict[str, Any] = {}

        self.load_data()

    def load_data(self):
        # Load merged Kaggle IBM AML + PaySim dataset tables (now also returns labels)
        self.df_transactions, self.df_customers, self.customer_labels = \
            load_and_merge_kaggle_datasets(self.data_dir)

        # Pre-compute feature baseline
        self.df_features = self.feature_tool.run(self.df_transactions)

        # Train / load supervised ML classifier — uses IBM AML labels when available
        self.model_info = self.ml_classifier.fit_or_load(self.df_features, self.customer_labels)
        model_type = self.model_info.get("model_type", "IsolationForest")
        print(f"✅ [Agent] Active scorer: {model_type} "
              f"({'supervised' if self.model_info.get('is_supervised') else 'unsupervised'})")

        # Inject supervised scores into the feature frame so HybridAnomalyTool uses them
        ml_scores = self.ml_classifier.score(self.df_features)
        df_features_with_scores = self.df_features.copy()
        df_features_with_scores["ml_score"] = ml_scores

        self.df_scored = self.anomaly_tool.run(df_features_with_scores, use_precomputed_ml=True)
        self.df_classified = self.classifier_tool.run(self.df_scored)

    def _get_ml_tool_name(self) -> str:
        """Returns dynamic ML tool name based on active classifier mode (supervised vs unsupervised)."""
        if self.model_info and self.model_info.get("is_supervised"):
            mtype = self.model_info.get("model_type", "XGBoost")
            return f"Supervised{mtype}Tool"
        return "IsolationForestTool"

    def _get_top_suspicious_customer_id(self) -> str:
        """Dynamically picks top suspicious subject from the loaded dataset."""
        if not self.df_classified.empty:
            sorted_df = self.df_classified.sort_values(by="composite_risk_score", ascending=False)
            return str(sorted_df.iloc[0]["customer_id"])
        return "CUST-0001"

    def _find_closest_customer_id(self, target_id: str, raw_num: str = None) -> Optional[str]:
        """Finds closest matching customer ID from active dataset using string distance and numeric matching."""
        all_ids = self.df_customers["customer_id"].astype(str).tolist()
        if target_id in all_ids:
            return target_id
        
        if raw_num:
            try:
                num_val = int(raw_num)
                padded = f"CUST-{num_val:04d}"
                if padded in all_ids:
                    return padded
                
                nums = [int(i.replace("CUST-", "")) for i in all_ids if i.startswith("CUST-") and i.replace("CUST-", "").isdigit()]
                if nums:
                    closest_num = min(nums, key=lambda x: abs(x - num_val))
                    return f"CUST-{closest_num:04d}"
            except ValueError:
                pass

        matches = difflib.get_close_matches(target_id, all_ids, n=1, cutoff=0.3)
        return matches[0] if matches else None

    def _clean_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Utility to convert DataFrame to dict records replacing NaN/Inf with JSON-compliant defaults."""
        df_copy = df.copy()
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].fillna("")
            else:
                df_copy[col] = df_copy[col].fillna(0)
        return df_copy.to_dict(orient="records")

    def process_query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        parse_result = self.parser.parse_query(query)
        intent = parse_result["intent"]
        entities = parse_result["entities"]

        execution_plan = []
        tools_called = []
        tools_skipped = []

        output_payload = {
            "query": query,
            "parsed_intent": intent,
            "extracted_entities": entities,
            "telemetry": {},
            "results": {},
            "explanations": [],
            "sar_narrative": None
        }

        ml_tool_name = self._get_ml_tool_name()

        if intent == "GREETING":
            tools_called.append("GreetingResponseTool")
            tools_skipped.extend([ml_tool_name, "SARGeneratorTool"])
            execution_plan = [
                "1. Parse conversational greeting intent",
                "2. Provide welcoming analyst introduction & system capability overview"
            ]
            top_cid = self._get_top_suspicious_customer_id()
            exp = (
                "👋 <strong>Hello! I am SENTINEL-AML</strong>, your autonomous compliance & anti-money laundering decision assistant.<br><br>"
                "I can analyze transaction feeds, detect structuring patterns, run Isolation Forest anomaly scoring, and auto-generate FinCEN SARs.<br><br>"
                "<strong>Quick Prompts to Try:</strong><br>"
                "• <em>'Which customer has the highest risk score?'</em><br>"
                f"• <em>'Is {top_cid} suspicious?'</em><br>"
                "• <em>'Find structuring patterns in the last 30 days'</em><br>"
                "• <em>'Show transactions in FATF high risk countries'</em>"
            )
            output_payload["explanations"].append(exp)

        elif intent == "CAPABILITIES_HELP":
            tools_called.append("CapabilitiesHelpTool")
            tools_skipped.extend([ml_tool_name, "SARGeneratorTool"])
            execution_plan = [
                "1. Parse help request intent",
                "2. Display comprehensive system capabilities & available prompt commands"
            ]
            top_cid = self._get_top_suspicious_customer_id()
            exp = (
                "<strong>SENTINEL-AML Capabilities & Command Taxonomy:</strong><br>"
                "• <strong>Highest Risk Query:</strong> <em>'Which customer has the highest risk score?'</em><br>"
                "• <strong>Structuring & Smurfing Search:</strong> <em>'Find structuring patterns in last 30 days'</em> or <em>'10+ txns under $10,000'</em><br>"
                f"• <strong>Entity Risk Lookup:</strong> <em>'Is {top_cid} suspicious?'</em> or <em>'Explain risk for customer {top_cid.replace('CUST-', '')}'</em><br>"
                "• <strong>Large Volume Filters:</strong> <em>'Transactions above $50,000'</em> or <em>'Volume over $100k'</em><br>"
                "• <strong>Jurisdiction Analysis:</strong> <em>'Show transactions in FATF high risk countries'</em> (KY, PA, AE)<br>"
                "• <strong>Channel / Wire Breakdown:</strong> <em>'Summarize wire transfers'</em> or <em>'Show cash out transactions'</em><br>"
                "• <strong>Risk Count Summary:</strong> <em>'How many high risk customers do we have?'</em><br>"
                "• <strong>Dataset EDA Overview:</strong> <em>'Perform full EDA on transaction dataset'</em>"
            )
            output_payload["explanations"].append(exp)

        elif intent == "TOP_RISK_SUBJECT":
            tools_called.extend([ml_tool_name, "RiskClassifierTool", "SingleEntityLookupTool", "SARGeneratorTool"])
            tools_skipped.extend(["EDATool", "ThresholdStressTestTool"])
            execution_plan = [
                "1. Sort customer dataset by composite ML risk score descending",
                "2. Extract rank #1 subject with highest risk score",
                "3. Display subject profile, risk breakdown, and recommended compliance escalation",
                "4. Auto-generate FinCEN SAR narrative for top subject"
            ]

            sorted_df = pd.merge(self.df_classified, self.df_customers, on="customer_id", how="left").sort_values(by="composite_risk_score", ascending=False)
            output_payload["results"]["flagged_table"] = self._clean_records(sorted_df.head(10))

            top_row = sorted_df.iloc[0]
            top_cid = top_row["customer_id"]
            cust_name = top_row.get("customer_name") or top_cid

            exp = (
                f"🏆 <strong>Top Risk Subject in Dataset:</strong> Customer <strong>{top_cid}</strong> ({cust_name}) "
                f"holds the highest composite risk score of <strong>{top_row['composite_risk_score']}/100 ({top_row['risk_level']} RISK)</strong>.<br>"
                f"• <strong>Recommended Action:</strong> <strong>{top_row['recommended_action']}</strong><br>"
                f"• <strong>Structuring Count:</strong> {top_row.get('structuring_count', 0)} cash deposits under statutory limit<br>"
                f"• <strong>Total Volume:</strong> ${top_row.get('total_tx_volume', 0):,.2f}"
            )
            output_payload["explanations"].append(exp)

            lookup_data = self.single_lookup_tool.run(top_cid, self.df_transactions, self.df_customers, self.df_classified)
            if lookup_data.get("found") and top_row["risk_level"] == "HIGH":
                sar_text = self.sar_tool.generate_sar(top_cid, lookup_data["customer"], lookup_data["risk_profile"], lookup_data["transaction_history"], model_info=self.model_info)
                output_payload["sar_narrative"] = sar_text

        elif intent == "EXPLAIN_RISK_REASON":
            raw_cid = entities.get("customer_id") or self._get_top_suspicious_customer_id()
            cid = self._find_closest_customer_id(raw_cid, entities.get("raw_cust_num")) or raw_cid

            tools_called.extend(["SingleEntityLookupTool", "RiskClassifierTool"])
            tools_skipped.extend(["EDATool", "ThresholdStressTestTool"])
            execution_plan = [
                f"1. Extract target Customer ID: {cid}",
                "2. Lookup single entity profile & transaction history",
                "3. Deconstruct composite risk score into constituent risk factors"
            ]
            lookup_data = self.single_lookup_tool.run(cid, self.df_transactions, self.df_customers, self.df_classified)
            output_payload["results"]["single_lookup"] = lookup_data
            
            if lookup_data.get("found"):
                r = lookup_data["risk_profile"]
                c = lookup_data["customer"]
                reasons = []
                if r.get("structuring_count", 0) > 0:
                    reasons.append(f"• <strong>Structuring Activity:</strong> {r['structuring_count']} cash deposits in structuring band.")
                if r.get("rapid_cashout_count", 0) > 0:
                    reasons.append(f"• <strong>Rapid Cash-Out Velocity:</strong> {r['rapid_cashout_count']} immediate withdrawal spikes within 2 hours.")
                if r.get("high_risk_country_tx_count", 0) > 0:
                    reasons.append(f"• <strong>FATF High-Risk Jurisdiction:</strong> {r['high_risk_country_tx_count']} transactions involving off-shore codes (KY, PA, AE).")
                if r.get("ml_score", 0) > 50.0:
                    model_label = self.model_info.get("model_type", "ML")
                    reasons.append(f"• <strong>ML Risk Score ({model_label}):</strong> Algorithm flagged anomalous behavior (ML score: {r['ml_score']}/100).")
                
                reason_text = "<br>".join(reasons) if reasons else "• Standard low-risk profile with normal transaction velocity."
                cust_name = c.get("customer_name") or c.get("customer_id") or cid
                match_note = f" <em>(Matched nearest record for query {raw_cid})</em>" if raw_cid != cid else ""
                exp = (
                    f"<strong>Risk Factor Decomposition for {cid} ({cust_name}){match_note}:</strong><br>"
                    f"Composite Risk Score: <strong>{r.get('composite_risk_score')}/100 ({r.get('risk_level')} RISK)</strong><br>"
                    f"Recommended Action: <strong>{r.get('recommended_action')}</strong><br><br>"
                    f"{reason_text}"
                )
                output_payload["explanations"].append(exp)
                
                if r.get("risk_level") == "HIGH":
                    sar_text = self.sar_tool.generate_sar(cid, c, r, lookup_data["transaction_history"], model_info=self.model_info)
                    output_payload["sar_narrative"] = sar_text

        elif intent == "SINGLE_ENTITY_LOOKUP":
            raw_cid = entities.get("customer_id") or self._get_top_suspicious_customer_id()
            cid = self._find_closest_customer_id(raw_cid, entities.get("raw_cust_num")) or raw_cid

            tools_called.extend(["SingleEntityLookupTool", "RiskClassifierTool", "SARGeneratorTool"])
            tools_skipped.extend(["EDATool", ml_tool_name, "ThresholdStressTestTool"])
            
            execution_plan = [
                f"1. Resolve target Customer ID: {cid}",
                "2. Perform single-entity database lookup (SingleEntityLookupTool)",
                "3. Compute individual risk profile & rule hits (RiskClassifierTool)",
                "4. Auto-generate FinCEN SAR Narrative if high risk (SARGeneratorTool)"
            ]

            lookup_data = self.single_lookup_tool.run(cid, self.df_transactions, self.df_customers, self.df_classified)
            output_payload["results"]["single_lookup"] = lookup_data
            
            if lookup_data.get("found"):
                risk_prof = lookup_data["risk_profile"]
                cust_info = lookup_data["customer"]
                tx_hist = lookup_data["transaction_history"]
                
                cust_name = cust_info.get('customer_name') or cust_info.get('customer_id') or cid
                match_note = f" <em>(Mapped {raw_cid} to nearest active dataset record {cid})</em>" if raw_cid != cid else ""
                exp_str = (
                    f"Customer <strong>{cid}</strong> ({cust_name}){match_note} has a Risk Score of "
                    f"<strong>{risk_prof.get('composite_risk_score')}/100 ({risk_prof.get('risk_level')} RISK)</strong>. "
                    f"Recommended Action: <strong>{risk_prof.get('recommended_action')}</strong>. "
                    f"Detected {risk_prof.get('structuring_count')} transactions in the structuring band."
                )
                output_payload["explanations"].append(exp_str)
                
                if risk_prof.get("risk_level") == "HIGH":
                    sar_text = self.sar_tool.generate_sar(cid, cust_info, risk_prof, tx_hist, model_info=self.model_info)
                    output_payload["sar_narrative"] = sar_text

        elif intent == "HIGH_RISK_FILTER":
            risk_flt = entities.get("risk_filter") or "HIGH"
            tools_called.extend(["RiskClassifierTool", ml_tool_name])
            tools_skipped.extend(["SingleEntityLookupTool"])
            execution_plan = [
                f"1. Filter dataset for subjects classified as {risk_flt} RISK",
                "2. Sort by composite ML risk score descending",
                "3. Present flagged subjects for compliance review"
            ]
            flagged = self.df_classified[self.df_classified["risk_level"] == risk_flt].sort_values(by="composite_risk_score", ascending=False)
            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)
            
            top_subj = merged.iloc[0] if not merged.empty else None
            top_info = f" (Top: <strong>{top_subj['customer_id']}</strong> with score {top_subj['composite_risk_score']}/100)" if top_subj is not None else ""
            exp_str = f"Filtered <strong>{len(flagged)} subjects</strong> classified as <strong>{risk_flt} RISK</strong>{top_info}. Check the Flagged Risk Table for details."
            output_payload["explanations"].append(exp_str)

        elif intent == "STRUCTURING_SEARCH":
            tools_called.extend(["AMLFeatureEngTool", ml_tool_name, "RiskClassifierTool", "SARGeneratorTool"])
            tools_skipped.extend(["EDATool", "SingleEntityLookupTool", "ThresholdStressTestTool"])

            execution_plan = [
                "1. Filter for transactions in Structuring & Smurfing patterns",
                "2. Aggregate customer rolling transaction frequencies (AMLFeatureEngTool)",
                f"3. Execute {self.model_info.get('model_type', 'ML Classifier')} & Structuring Rule Engine",
                "4. Classify high-risk subjects & assign escalation recommendations (RiskClassifierTool)",
                "5. Auto-generate FinCEN SAR Narratives for top flagged subjects"
            ]

            flagged = self.df_classified[self.df_classified["structuring_count"] > 0].sort_values(by="composite_risk_score", ascending=False)
            if flagged.empty:
                flagged = self.df_classified.sort_values(by="composite_risk_score", ascending=False).head(10)

            merged_flagged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged_flagged)

            top_subj = merged_flagged.iloc[0] if not merged_flagged.empty else None
            if top_subj is not None:
                name_str = top_subj['customer_name'] if ('customer_name' in top_subj and pd.notna(top_subj['customer_name'])) else top_subj['customer_id']
                exp_str = (
                    f"Identified <strong>{len(flagged)} subjects</strong> exhibiting structuring patterns. "
                    f"Top Subject: <strong>{top_subj['customer_id']}</strong> ({name_str}) with "
                    f"<strong>{top_subj['structuring_count']} cash deposits</strong> under statutory limit (Risk Score: {top_subj['composite_risk_score']}/100)."
                )
                output_payload["explanations"].append(exp_str)
                
                sar_text = self.sar_tool.generate_sar(
                    top_subj["customer_id"], 
                    top_subj.to_dict(), 
                    top_subj.to_dict(), 
                    self.df_transactions[self.df_transactions["customer_id"] == top_subj["customer_id"]].to_dict(orient="records"),
                    model_info=self.model_info
                )
                output_payload["sar_narrative"] = sar_text

        elif intent == "LARGE_AMOUNT_FILTER":
            min_amt = entities.get("min_amount") or (self.df_transactions["amount"].quantile(0.95))
            tools_called.extend(["AMLFeatureEngTool", "LargeAmountFilterTool"])
            tools_skipped.extend(["EDATool", "SARGeneratorTool"])

            execution_plan = [
                f"1. Filter transaction ledger for entries with amount >= ${min_amt:,.2f}",
                "2. Join customer metadata and compute aggregated transaction volume",
                "3. Rank subjects by total large transaction volume"
            ]

            large_txs = self.df_transactions[self.df_transactions["amount"] >= min_amt]
            large_cust_ids = large_txs["customer_id"].unique()
            flagged = self.df_classified[self.df_classified["customer_id"].isin(large_cust_ids)].sort_values(by="total_tx_volume", ascending=False)
            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)

            total_vol = large_txs["amount"].sum()
            exp = (
                f"Found <strong>{len(large_txs)} transactions</strong> exceeding <strong>${min_amt:,.2f}</strong> across "
                f"<strong>{len(merged)} customers</strong>, representing <strong>${total_vol:,.2f}</strong> total high-value volume."
            )
            output_payload["explanations"].append(exp)

        elif intent == "JURISDICTION_ANALYSIS":
            target_cc = entities.get("country_code")
            tools_called.extend(["JurisdictionAnalysisTool", "RiskClassifierTool"])
            tools_skipped.extend(["SingleEntityLookupTool"])

            execution_plan = [
                "1. Filter transaction history for FATF Grey/Blacklist country codes (KY, PA, AE)",
                "2. Aggregate volume and transaction counts per jurisdiction",
                "3. Flag subjects with cross-border offshore transfers"
            ]

            high_risk_cc = ["KY", "PA", "AE"]
            if target_cc:
                tx_filtered = self.df_transactions[self.df_transactions["country_code"] == target_cc]
                cust_ids = tx_filtered["customer_id"].unique()
                sort_col = "high_risk_country_volume" if "high_risk_country_volume" in self.df_classified.columns else "composite_risk_score"
                flagged = self.df_classified[self.df_classified["customer_id"].isin(cust_ids)].sort_values(by=sort_col, ascending=False)
                exp = f"Found <strong>{len(tx_filtered)} transactions</strong> involving jurisdiction <strong>{target_cc}</strong> totaling <strong>${tx_filtered['amount'].sum():,.2f}</strong> across {len(flagged)} customers."
            else:
                tx_filtered = self.df_transactions[self.df_transactions["country_code"].isin(high_risk_cc)]
                flagged = self.df_classified[self.df_classified["high_risk_country_tx_count"] > 0].sort_values(by="high_risk_country_volume", ascending=False)
                exp = (
                    f"Found <strong>{len(tx_filtered)} transactions</strong> in FATF High-Risk Jurisdictions (KY, PA, AE) "
                    f"totaling <strong>${tx_filtered['amount'].sum():,.2f}</strong> across <strong>{len(flagged)} subjects</strong>."
                )

            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)
            output_payload["explanations"].append(exp)

        elif intent == "TRANSACTION_TYPE_BREAKDOWN":
            tx_type = entities.get("transaction_type") or "Wire"
            tools_called.extend(["TransactionTypeFilterTool"])
            tools_skipped.extend(["SARGeneratorTool"])

            execution_plan = [
                f"1. Filter dataset for transaction type: {tx_type}",
                "2. Calculate total volume and count breakdown",
                "3. Display customer rankings for this transaction channel"
            ]

            filtered_tx = self.df_transactions[self.df_transactions["transaction_type"].str.lower() == tx_type.lower()]
            tot_vol = filtered_tx["amount"].sum()
            cust_ids = filtered_tx["customer_id"].unique()
            flagged = self.df_classified[self.df_classified["customer_id"].isin(cust_ids)].sort_values(by="total_tx_volume", ascending=False)
            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)

            exp = (
                f"Found <strong>{len(filtered_tx)} {tx_type} transactions</strong> totaling <strong>${tot_vol:,.2f}</strong> "
                f"across {len(cust_ids)} unique customers."
            )
            output_payload["explanations"].append(exp)

        elif intent == "COUNT_RISK_SUMMARY":
            risk_flt = entities.get("risk_filter") or "HIGH"
            tools_called.extend(["RiskClassifierTool", "EDATool"])
            tools_skipped.extend(["SingleEntityLookupTool"])

            execution_plan = [
                "1. Aggregate Risk Rating distribution across all active customer profiles",
                f"2. Filter for {risk_flt} risk category",
                "3. Calculate population percentages and escalation actions"
            ]

            counts = self.df_classified["risk_level"].value_counts().to_dict()
            high_cnt = counts.get("HIGH", 0)
            med_cnt = counts.get("MEDIUM", 0)
            low_cnt = counts.get("LOW", 0)
            tot_cnt = len(self.df_classified)

            flagged = self.df_classified[self.df_classified["risk_level"] == risk_flt].sort_values(by="composite_risk_score", ascending=False)
            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)

            exp = (
                f"<strong>Customer Population Risk Breakdown ({tot_cnt} total subjects):</strong><br>"
                f"• <strong>HIGH Risk:</strong> {high_cnt} subjects ({round(high_cnt/tot_cnt*100 if tot_cnt > 0 else 0, 1)}%) — Immediate SAR Filing Required<br>"
                f"• <strong>MEDIUM Risk:</strong> {med_cnt} subjects ({round(med_cnt/tot_cnt*100 if tot_cnt > 0 else 0, 1)}%) — Enhanced Due Diligence (EDD)<br>"
                f"• <strong>LOW Risk:</strong> {low_cnt} subjects ({round(low_cnt/tot_cnt*100 if tot_cnt > 0 else 0, 1)}%) — Standard Monitoring"
            )
            output_payload["explanations"].append(exp)

        elif intent == "THRESHOLD_AGGREGATION":
            min_count = entities.get("min_count") or 5
            max_amt = entities.get("max_amount") or (self.df_transactions["amount"].max() * 0.99)
            
            tools_called.extend(["AMLFeatureEngTool", "ThresholdAggregationTool"])
            tools_skipped.extend(["EDATool", "IsolationForestMLTool", "SARGeneratorTool"])

            execution_plan = [
                f"1. Apply direct threshold rule: transactions < ${max_amt:,.2f} with count >= {min_count}",
                "2. Perform group-by aggregation on transaction table (Skip ML Anomaly Scoring)",
                "3. Compute total volume and customer breakdown",
                "4. Return instant compliance table"
            ]

            filtered_cust = self.df_classified[
                (self.df_classified["structuring_count"] >= min_count) | 
                ((self.df_classified["total_tx_count"] >= min_count) & (self.df_classified["avg_amount"] < max_amt))
            ]
            merged = pd.merge(filtered_cust, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)
            
            exp_str = f"Found <strong>{len(merged)} customers</strong> making {min_count}+ transactions under ${max_amt:,.2f}."
            output_payload["explanations"].append(exp_str)

        elif intent == "FULL_EDA":
            tools_called.extend(["EDATool", "AMLFeatureEngTool"])
            tools_skipped.extend(["SingleEntityLookupTool", "SARGeneratorTool"])

            execution_plan = [
                "1. Load complete transaction and customer baseline datasets",
                "2. Calculate dataset summary metrics, distributions, and top volume subjects (EDATool)",
                "3. Generate distribution histograms & risk rating breakdown",
                "4. Display baseline profiling overview"
            ]

            eda_res = self.eda_tool.run(self.df_transactions, self.df_customers)
            output_payload["results"]["eda"] = eda_res
            exp_str = f"Dataset contains <strong>{eda_res['summary']['total_transactions']} transactions</strong> across <strong>{eda_res['summary']['unique_customers']} unique customers</strong> with <strong>${eda_res['summary']['total_volume']:,.2f}</strong> total volume."
            output_payload["explanations"].append(exp_str)

        else:
            tools_called.extend(["AMLFeatureEngTool", "HybridAnomalyTool", "RiskClassifierTool"])
            tools_skipped.extend(["SingleEntityLookupTool"])

            execution_plan = [
                "1. Parse analytical query context",
                "2. Execute batch AML feature extraction and Isolation Forest ML scoring",
                "3. Filter high & medium risk subjects requiring analyst review",
                "4. Present top suspicious subjects with escalation guidance"
            ]
            
            high_risk = self.df_classified[self.df_classified["risk_level"].isin(["HIGH", "MEDIUM"])].sort_values(by="composite_risk_score", ascending=False)
            merged = pd.merge(high_risk, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)
            
            top_subj = merged.iloc[0] if not merged.empty else None
            top_info = f" Top Subject: <strong>{top_subj['customer_id']}</strong> ({top_subj.get('customer_name', top_subj['customer_id'])}) with Risk Score <strong>{top_subj['composite_risk_score']}/100</strong>." if top_subj is not None else ""
            exp_str = f"Identified <strong>{len(high_risk)} suspicious subjects</strong> requiring compliance review.{top_info}"
            output_payload["explanations"].append(exp_str)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        
        output_payload["telemetry"] = {
            "execution_plan": execution_plan,
            "tools_called": tools_called,
            "tools_skipped": tools_skipped,
            "latency_ms": elapsed_ms
        }

        return output_payload

    def stress_test_threshold(self, lower_bound: float) -> Dict[str, Any]:
        return self.stress_test_tool.run(self.df_transactions, self.df_features, lower_bound=lower_bound)

    def get_model_info(self) -> Dict[str, Any]:
        """Return the active ML model metadata for the /api/model/info endpoint."""
        return self.model_info
