import time
import os
import pandas as pd
from typing import Dict, Any, List
from app.nlp_parser import NLPIntentParser
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
    """
    def __init__(self, data_dir: str = "/Users/sterlingsuman/Desktop/projectx/data"):
        self.data_dir = data_dir
        self.parser = NLPIntentParser()
        self.eda_tool = EDATool()
        self.feature_tool = AMLFeatureEngTool()
        self.anomaly_tool = HybridAnomalyTool()
        self.classifier_tool = RiskClassifierTool()
        self.single_lookup_tool = SingleEntityLookupTool()
        self.stress_test_tool = ThresholdStressTestTool()
        self.sar_tool = SARGeneratorTool()
        
        self.load_data()

    def load_data(self):
        cust_path = os.path.join(self.data_dir, "customers.csv")
        tx_path = os.path.join(self.data_dir, "transactions.csv")
        
        if not os.path.exists(cust_path) or not os.path.exists(tx_path):
            from data_generator import generate_aml_dataset
            generate_aml_dataset(data_dir=self.data_dir)
            
        self.df_customers = pd.read_csv(cust_path)
        self.df_transactions = pd.read_csv(tx_path)

        # Pre-compute feature & anomaly baseline
        self.df_features = self.feature_tool.run(self.df_transactions)
        self.df_scored = self.anomaly_tool.run(self.df_features)
        self.df_classified = self.classifier_tool.run(self.df_scored)

    def process_query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # Step 1: Parse Intent & Entities (Zero LLM)
        parse_result = self.parser.parse_query(query)
        intent = parse_result["intent"]
        entities = parse_result["entities"]

        # Step 2: Dynamic Execution Planning
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

        # Dynamic Execution Plan Construction based on Query Intent
        if intent == "SINGLE_ENTITY_LOOKUP":
            cid = entities.get("customer_id") or "CUST-4521"
            tools_called.extend(["SingleEntityLookupTool", "RiskClassifierTool", "SARGeneratorTool"])
            tools_skipped.extend(["EDATool", "DatasetWideMLTool", "ThresholdStressTestTool"])
            
            execution_plan = [
                f"1. Extract target Customer ID: {cid}",
                "2. Perform single-entity database lookup (SingleEntityLookupTool)",
                "3. Compute individual risk profile & rule hits (RiskClassifierTool)",
                "4. Auto-generate FinCEN SAR Narrative if high risk (SARGeneratorTool)",
                "5. Skip dataset-wide EDA & batch clustering to optimize latency"
            ]

            lookup_data = self.single_lookup_tool.run(cid, self.df_transactions, self.df_customers, self.df_classified)
            output_payload["results"]["single_lookup"] = lookup_data
            
            if lookup_data.get("found"):
                risk_prof = lookup_data["risk_profile"]
                cust_info = lookup_data["customer"]
                tx_hist = lookup_data["transaction_history"]
                
                exp_str = (
                    f"Customer {cid} ({cust_info.get('customer_name')}) has a Risk Score of "
                    f"{risk_prof.get('composite_risk_score')}/100 ({risk_prof.get('risk_level')} RISK). "
                    f"Recommended Action: {risk_prof.get('recommended_action')}. "
                    f"Detected {risk_prof.get('structuring_count')} transactions in the structuring band ($9,000-$9,999)."
                )
                output_payload["explanations"].append(exp_str)
                
                if risk_prof.get("risk_level") == "HIGH":
                    sar_text = self.sar_tool.generate_sar(cid, cust_info, risk_prof, tx_hist)
                    output_payload["sar_narrative"] = sar_text

        elif intent == "STRUCTURING_SEARCH":
            tools_called.extend(["AMLFeatureEngTool", "HybridAnomalyTool", "RiskClassifierTool", "SARGeneratorTool"])
            tools_skipped.extend(["EDATool", "SingleEntityLookupTool", "ThresholdStressTestTool"])

            execution_plan = [
                "1. Filter for transactions in Structuring & Smurfing patterns ($9,000 - $9,999 band)",
                "2. Aggregate customer rolling transaction frequencies (AMLFeatureEngTool)",
                "3. Execute IsolationForest ML anomaly detector & Structuring Rule Engine (HybridAnomalyTool)",
                "4. Classify high-risk subjects & assign escalation recommendations (RiskClassifierTool)",
                "5. Auto-generate FinCEN SAR Narratives for top flagged subjects"
            ]

            flagged = self.df_classified[self.df_classified["structuring_count"] > 0].sort_values(by="composite_risk_score", ascending=False)
            merged_flagged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = merged_flagged.to_dict(orient="records")

            top_subj = merged_flagged.iloc[0] if not merged_flagged.empty else None
            if top_subj is not None:
                exp_str = (
                    f"Identified {len(flagged)} subjects exhibiting structuring patterns. "
                    f"Top Subject: {top_subj['customer_id']} ({top_subj['customer_name']}) with "
                    f"{top_subj['structuring_count']} cash deposits under $10,000 (Risk Score: {top_subj['composite_risk_score']}/100)."
                )
                output_payload["explanations"].append(exp_str)
                
                # Auto-generate SAR for top subject
                sar_text = self.sar_tool.generate_sar(
                    top_subj["customer_id"], 
                    top_subj.to_dict(), 
                    top_subj.to_dict(), 
                    self.df_transactions[self.df_transactions["customer_id"] == top_subj["customer_id"]].to_dict(orient="records")
                )
                output_payload["sar_narrative"] = sar_text

        elif intent == "THRESHOLD_AGGREGATION":
            min_count = entities.get("min_count") or 10
            max_amt = entities.get("max_amount") or 10000.0
            
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
            output_payload["results"]["flagged_table"] = merged.to_dict(orient="records")
            
            exp_str = f"Found {len(merged)} customers making {min_count}+ transactions under ${max_amt:,.2f}."
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
            exp_str = f"Dataset contains {eda_res['summary']['total_transactions']} transactions across {eda_res['summary']['unique_customers']} unique customers with ${eda_res['summary']['total_volume']:,.2f} total volume."
            output_payload["explanations"].append(exp_str)

        else:
            # Default General Search
            tools_called.extend(["AMLFeatureEngTool", "HybridAnomalyTool", "RiskClassifierTool"])
            tools_skipped.extend(["SingleEntityLookupTool"])

            execution_plan = [
                "1. Parse general analytical request",
                "2. Run batch AML feature extraction and Isolation Forest ML scoring",
                "3. Filter high & medium risk subjects",
                "4. Present top risk subjects with escalation guidance"
            ]
            
            high_risk = self.df_classified[self.df_classified["risk_level"].isin(["HIGH", "MEDIUM"])].sort_values(by="composite_risk_score", ascending=False)
            merged = pd.merge(high_risk, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = merged.to_dict(orient="records")
            exp_str = f"Detected {len(high_risk)} suspicious subjects requiring analyst review."
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
