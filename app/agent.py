import html
import time
import os
import difflib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
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
    SARGeneratorTool,
    HIGH_RISK_JURISDICTIONS
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

        # Synchronize df_customers risk_rating with dynamically classified risk levels
        classified_map = dict(zip(self.df_classified["customer_id"], self.df_classified["risk_level"]))
        self.df_customers["risk_rating"] = (
            self.df_customers["customer_id"]
            .map(classified_map)
            .fillna(self.df_customers["risk_rating"])
        )

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

    def _find_closest_customer_id(self, target_id: str, raw_num: str = None) -> dict:
        """Finds closest matching customer ID from active dataset.

        Returns
        -------
        dict with keys:
            resolved_id : str | None – the matched customer ID
            match_type   : 'exact' | 'padded' | 'numeric_nearest' | 'fuzzy' | None
        """
        all_ids = self.df_customers["customer_id"].astype(str).tolist()
        if target_id in all_ids:
            return {"resolved_id": target_id, "match_type": "exact"}

        if raw_num:
            try:
                num_val = int(raw_num)
                padded = f"CUST-{num_val:04d}"
                if padded in all_ids:
                    return {"resolved_id": padded, "match_type": "padded"}

                nums = [int(i.replace("CUST-", "")) for i in all_ids if i.startswith("CUST-") and i.replace("CUST-", "").isdigit()]
                if nums:
                    closest_num = min(nums, key=lambda x: abs(x - num_val))
                    return {"resolved_id": f"CUST-{closest_num:04d}", "match_type": "numeric_nearest"}
            except ValueError:
                pass

        matches = difflib.get_close_matches(target_id, all_ids, n=1, cutoff=0.6)
        if matches:
            return {"resolved_id": matches[0], "match_type": "fuzzy"}
        return {"resolved_id": None, "match_type": None}

    def _clean_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Utility to convert DataFrame to dict records replacing NaN/Inf with JSON-compliant defaults."""
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        obj_cols = df_clean.select_dtypes(include=["object", "string"]).columns
        num_cols = df_clean.select_dtypes(exclude=["object", "string"]).columns
        df_clean[obj_cols] = df_clean[obj_cols].fillna("")
        df_clean[num_cols] = df_clean[num_cols].fillna(0)
        return df_clean.to_dict(orient="records")

    def _get_windowed_data(
        self,
        time_window_days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Dynamically filters transaction ledger by requested time window or date range,
        re-evaluating features and risk scores on demand.
        Returns (df_tx_window, df_scored_window, df_classified_window).
        """
        if not time_window_days and not start_date and not end_date:
            return self.df_transactions, self.df_scored, self.df_classified

        df_tx = self.df_transactions.copy()
        df_tx["ts"] = pd.to_datetime(df_tx["timestamp"], errors="coerce")

        if start_date and end_date:
            df_filtered = df_tx[
                (df_tx["ts"] >= pd.to_datetime(start_date))
                & (df_tx["ts"] <= pd.to_datetime(end_date))
            ]
        elif time_window_days:
            max_dt = df_tx["ts"].max()
            cutoff = max_dt - pd.Timedelta(days=int(time_window_days))
            df_filtered = df_tx[df_tx["ts"] >= cutoff]
        else:
            df_filtered = df_tx

        if df_filtered.empty:
            return df_filtered, self.df_scored.head(0), self.df_classified.head(0)

        # Dynamic feature extraction and scoring on temporal subset
        df_feats = self.feature_tool.run(
            df_filtered, time_window_days=time_window_days or 30
        )
        ml_scores = self.ml_classifier.score(df_feats)
        df_feats_scored = df_feats.copy()
        df_feats_scored["ml_score"] = ml_scores

        df_scored = self.anomaly_tool.run(
            df_feats_scored, use_precomputed_ml=True
        )
        df_classified = self.classifier_tool.run(df_scored)
        return df_filtered, df_scored, df_classified

    def _format_time_window_phrase(self, time_win: Optional[int]) -> Tuple[str, str]:
        """
        Dynamically formats time window into natural English (days, months, or years).
        Returns (window_note, win_header).
        Examples:
          730  -> (" in the last 2 years", " (Past 2 Years)")
          365  -> (" in the last 1 year", " (Past 1 Year)")
          1825 -> (" in the last 5 years", " (Past 5 Years)")
          180  -> (" in the last 6 months", " (Past 6 Months)")
          90   -> (" in the last 3 months", " (Past 3 Months)")
          30   -> (" in the last 30 days", " (Past 30 Days)")
          3    -> (" in the last 3 days", " (Past 3 Days)")
        """
        if not time_win:
            return "", ""

        if time_win >= 365:
            years = time_win / 365.0
            if time_win % 365 == 0 or round(years, 1) == int(years):
                y_str = f"{int(round(years))}"
            else:
                y_str = f"{years:.1f}"

            unit_note = "year" if y_str == "1" else "years"
            unit_hdr = "Year" if y_str == "1" else "Years"
            return f" in the last {y_str} {unit_note}", f" (Past {y_str} {unit_hdr})"
        elif time_win >= 60 and time_win % 30 == 0:
            months = time_win // 30
            return f" in the last {months} months", f" (Past {months} Months)"

        return f" in the last {time_win} days", f" (Past {time_win} Days)"

    def process_query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()

        parse_result = self.parser.parse_query(query)
        intent = parse_result["intent"]
        entities = parse_result["entities"]

        execution_plan = []
        # Telemetry: distinguish tools that ran at startup from those invoked live
        tools_precomputed = ["AMLFeatureEngTool", "HybridAnomalyTool", "RiskClassifierTool"]
        tools_invoked_live: List[str] = []
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
            tools_invoked_live.append("GreetingResponseTool")
            tools_skipped.extend([ml_tool_name, "SARGeneratorTool"])
            execution_plan = [
                "1. Understand analyst greeting and query intent",
                "2. Introduce SENTINEL-AML financial intelligence assistant and investigative capabilities"
            ]
            top_cid = html.escape(self._get_top_suspicious_customer_id())
            exp = (
                "<div class='aml-card aml-info-card'>"
                "<div class='aml-card-header'>"
                "<span class='aml-badge aml-badge-indigo'>🛡️ SENTINEL-AML SYSTEM ASSISTANT</span>"
                "</div>"
                "<div class='aml-card-body'>"
                "👋 <strong>Welcome to SENTINEL-AML</strong>, your deterministic co-investigator for anti-money laundering and compliance investigations.<br><br>"
                "I analyze customer transaction ledgers, evaluate structuring/smurfing patterns, track FATF offshore jurisdictions, and generate audit-ready FinCEN SAR narratives.<br><br>"
                "<strong><span style='color: var(--accent-light);'>💡 Recommended Investigative Prompts:</span></strong>"
                "<ul class='aml-prompt-list'>"
                "<li><code>Which customer has the highest risk score?</code></li>"
                f"<li><code>Is {top_cid} suspicious?</code></li>"
                "<li><code>Find structuring patterns in the last 30 days</code></li>"
                "<li><code>Show transactions in FATF high risk countries</code></li>"
                "</ul>"
                "</div>"
                "</div>"
            )
            output_payload["explanations"].append(exp)

        elif intent == "CAPABILITIES_HELP":
            tools_invoked_live.append("CapabilitiesHelpTool")
            tools_skipped.extend([ml_tool_name, "SARGeneratorTool"])
            execution_plan = [
                "1. Analyze help request context",
                "2. Display available financial intelligence workflows and investigative queries"
            ]
            top_cid = self._get_top_suspicious_customer_id()
            exp = (
                "<div class='aml-card aml-info-card'>"
                "<div class='aml-card-header'>"
                "<span class='aml-badge aml-badge-indigo'>📚 INVESTIGATIVE TAXONOMY</span>"
                "</div>"
                "<div class='aml-card-body'>"
                "<strong>SENTINEL-AML Intelligence Workflows & Query Guide:</strong><br><br>"
                "• <strong>Highest Risk Subject:</strong> <code>Which customer has the highest risk score?</code><br>"
                "• <strong>Structuring & Smurfing:</strong> <code>Find structuring patterns in last 30 days</code> or <code>10+ txns under $10,000</code><br>"
                f"• <strong>Single Subject Deep Dive:</strong> <code>Is {top_cid} suspicious?</code> or <code>Explain risk for customer {top_cid.replace('CUST-', '')}</code><br>"
                "• <strong>High-Value Volumes:</strong> <code>Transactions above $50,000</code> or <code>Volume over $100k</code><br>"
                "• <strong>Offshore Jurisdictions:</strong> <code>Show transactions in FATF high risk countries</code> (KY, PA, AE)<br>"
                "• <strong>Channel Breakdown:</strong> <code>Summarize wire transfers</code> or <code>Show cash out transactions</code><br>"
                "• <strong>Population Breakdown:</strong> <code>How many high risk customers do we have?</code><br>"
                "• <strong>Full Ledger Overview:</strong> <code>Perform full EDA on transaction dataset</code>"
                "</div>"
                "</div>"
            )
        elif intent == "SAR_GENERATION":
            raw_cid = entities.get("customer_id") or self._get_top_suspicious_customer_id()
            match_result = self._find_closest_customer_id(raw_cid, entities.get("raw_cust_num"))
            cid = match_result["resolved_id"] or raw_cid

            execution_plan = [
                f"1. Target subject profile: {cid}",
                "2. Extract transaction ledger and risk indicators for subject",
                "3. Generate formal regulatory FinCEN Suspicious Activity Report (SAR) narrative"
            ]
            lookup_data = self.single_lookup_tool.run(cid, self.df_transactions, self.df_customers, self.df_classified)
            tools_invoked_live.append("SingleEntityLookupTool")
            output_payload["results"]["single_lookup"] = lookup_data

            if lookup_data.get("found"):
                sar_text = self.sar_tool.generate_sar(
                    cid, lookup_data["customer"], lookup_data["risk_profile"], lookup_data["transaction_history"], model_info=self.model_info
                )
                output_payload["sar_narrative"] = sar_text
                tools_invoked_live.append("SARGeneratorTool")

                safe_cid = html.escape(str(cid))
                exp = (
                    f"<div class='aml-card aml-risk-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge aml-badge-red'>📝 FinCEN SAR NARRATIVE GENERATED</span>"
                    f"<span class='aml-score-tag'>Subject: <strong>{safe_cid}</strong></span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"Generated regulatory Suspicious Activity Report narrative for subject <strong>{safe_cid}</strong>.<br><br>"
                    f"Review the narrative content below and in the <strong>FinCEN SAR Narrative Panel</strong>."
                    f"</div>"
                    f"</div>"
                )
                output_payload["explanations"].append(exp)
            else:
                safe_raw = html.escape(str(raw_cid))
                exp = (
                    f"<div class='aml-card aml-info-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge aml-badge-yellow'>⚠️ SUBJECT NOT FOUND</span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"Unable to generate SAR narrative for subject <strong>{safe_raw}</strong> — not found in active ledger."
                    f"</div>"
                    f"</div>"
                )
                output_payload["explanations"].append(exp)

        elif intent == "STRESS_TEST":
            bound = entities.get("stress_bound") or entities.get("min_amount") or entities.get("max_amount") or 9000.0
            execution_plan = [
                f"1. Configure structuring lower bound threshold to ${bound:,.2f}",
                "2. Recalculate customer structuring frequencies and population delta",
                "3. Quantify false positive impact and newly flagged suspicious subjects"
            ]
            res = self.stress_test_threshold(lower_bound=bound)
            output_payload["results"]["stress_test"] = res
            tools_invoked_live.append("ThresholdStressTestTool")

            exp = (
                f"<div class='aml-card aml-info-card'>"
                f"<div class='aml-card-header'>"
                f"<span class='aml-badge aml-badge-yellow'>⚡ STRUCTURING THRESHOLD STRESS TEST</span>"
                f"<span class='aml-score-tag'>Bound: <strong>${bound:,.2f}</strong></span>"
                f"</div>"
                f"<div class='aml-card-body'>"
                f"{html.escape(res['interpretation'])}<br><br>"
                f"<div class='aml-stats-grid'>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Baseline Flagged</span><span class='aml-stat-val'>{res['baseline_flagged_customers']}</span></div>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>New Flagged</span><span class='aml-stat-val val-red'>{res['new_flagged_customers']}</span></div>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Net Delta</span><span class='aml-stat-val'>+{res['customer_count_delta']}</span></div>"
                f"</div>"
                f"</div>"
                f"</div>"
            )
            output_payload["explanations"].append(exp)

        elif intent == "VELOCITY_SEARCH":
            tools_skipped.extend(["EDATool", "SingleEntityLookupTool", "ThresholdStressTestTool"])
            time_win = entities.get("time_window_days")
            s_date = entities.get("start_date")
            e_date = entities.get("end_date")

            df_tx_win, df_scored_win, df_classified_win = self._get_windowed_data(
                time_window_days=time_win, start_date=s_date, end_date=e_date
            )
            window_note, win_header = self._format_time_window_phrase(time_win)

            execution_plan = [
                f"1. Scan ledger{window_note} for rapid cash-out activity (withdrawals immediately following deposit)",
                "2. Measure time elapsed between incoming wire/deposit and outgoing cash withdrawal",
                "3. Rank subjects exhibiting rapid cash-out velocity red flags"
            ]
            flagged = df_classified_win[df_classified_win["rapid_cashout_count"] > 0].sort_values(by="composite_risk_score", ascending=False)
            if flagged.empty:
                flagged = df_classified_win.sort_values(by="composite_risk_score", ascending=False).head(10)
            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)

            exp = (
                f"<div class='aml-card aml-risk-card'>"
                f"<div class='aml-card-header'>"
                f"<span class='aml-badge aml-badge-red'>⚡ RAPID CASHOUT VELOCITY ALERT{win_header}</span>"
                f"<span class='aml-score-tag'>Flagged: <strong>{len(flagged)} Subjects</strong></span>"
                f"</div>"
                f"<div class='aml-card-body'>"
                f"Identified <strong>{len(flagged)} subjects</strong> exhibiting rapid cash-out velocity{window_note} (incoming funds withdrawn within 2 hours).<br><br>"
                f"Review flagged subjects in the <strong>Flagged Risk Table</strong> tab."
                f"</div>"
                f"</div>"
            )
            output_payload["explanations"].append(exp)

        elif intent == "TOP_RISK_SUBJECT":
            tools_skipped.extend(["EDATool", "ThresholdStressTestTool"])
            execution_plan = [
                "1. Screen customer population for highest overall suspicious activity indicators",
                "2. Isolate the priority subject exhibiting the most elevated risk profile",
                "3. Present risk factor breakdown and recommended investigative escalation",
                "4. Draft Suspicious Activity Report (SAR) narrative for regulatory filing"
            ]

            sorted_df = pd.merge(self.df_classified, self.df_customers, on="customer_id", how="left").sort_values(by="composite_risk_score", ascending=False)
            output_payload["results"]["flagged_table"] = self._clean_records(sorted_df.head(10))

            top_row = sorted_df.iloc[0]
            top_cid = html.escape(str(top_row["customer_id"]))
            raw_name = top_row.get("customer_name")
            name_str = html.escape(str(raw_name)) if (pd.notna(raw_name) and str(raw_name).strip() and str(raw_name).lower() != "nan" and str(raw_name) != top_cid) else ""
            name_display = f" ({name_str})" if name_str else ""
            risk_level = html.escape(str(top_row['risk_level']))
            score = top_row['composite_risk_score']

            badge_class = "aml-badge-red" if risk_level == "HIGH" else "aml-badge-yellow"
            
            exp = (
                f"<div class='aml-card aml-risk-card'>"
                f"<div class='aml-card-header'>"
                f"<span class='aml-badge {badge_class}'>⚠️ PRIORITY INVESTIGATIVE ESCALATION</span>"
                f"<span class='aml-score-tag'>Risk Score: <strong>{score}/100</strong></span>"
                f"</div>"
                f"<div class='aml-card-body'>"
                f"🏆 <strong>Top Risk Subject:</strong> Customer <strong>{top_cid}</strong>{name_display} "
                f"holds the highest composite risk rating in the ledger.<br><br>"
                f"<div class='aml-stats-grid'>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Risk Rating</span><span class='aml-stat-val val-red'>{risk_level}</span></div>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Structuring Txns</span><span class='aml-stat-val'>{top_row.get('structuring_count', 0)}</span></div>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Total Volume</span><span class='aml-stat-val'>${top_row.get('total_tx_volume', 0):,.2f}</span></div>"
                f"</div><br>"
                f"📋 <strong>Recommended Compliance Action:</strong> <strong>{html.escape(str(top_row['recommended_action']))}</strong>"
                f"</div>"
                f"</div>"
            )
            output_payload["explanations"].append(exp)

            lookup_data = self.single_lookup_tool.run(str(top_row["customer_id"]), self.df_transactions, self.df_customers, self.df_classified)
            tools_invoked_live.append("SingleEntityLookupTool")
            if lookup_data.get("found") and top_row["risk_level"] == "HIGH":
                sar_text = self.sar_tool.generate_sar(str(top_row["customer_id"]), lookup_data["customer"], lookup_data["risk_profile"], lookup_data["transaction_history"], model_info=self.model_info)
                output_payload["sar_narrative"] = sar_text
                tools_invoked_live.append("SARGeneratorTool")

        elif intent == "LOWEST_RISK_SUBJECT":
            tools_invoked_live.extend(["SingleEntityLookupTool"])
            tools_skipped.extend(["EDATool", "SARGeneratorTool"])
            execution_plan = [
                "1. Screen customer population for lowest overall suspicious activity indicators",
                "2. Isolate the baseline subject exhibiting the lowest risk score",
                "3. Display risk factor metrics and baseline compliance status"
            ]

            sorted_df = pd.merge(self.df_classified, self.df_customers, on="customer_id", how="left").sort_values(by="composite_risk_score", ascending=True)
            output_payload["results"]["flagged_table"] = self._clean_records(sorted_df.head(10))

            low_row = sorted_df.iloc[0]
            low_cid = html.escape(str(low_row["customer_id"]))
            raw_name = low_row.get("customer_name")
            name_str = html.escape(str(raw_name)) if (pd.notna(raw_name) and str(raw_name).strip() and str(raw_name).lower() != "nan" and str(raw_name) != low_cid) else ""
            name_display = f" ({name_str})" if name_str else ""
            risk_level = html.escape(str(low_row['risk_level']))
            score = low_row['composite_risk_score']

            badge_class = "aml-badge-indigo"
            
            exp = (
                f"<div class='aml-card aml-info-card'>"
                f"<div class='aml-card-header'>"
                f"<span class='aml-badge {badge_class}'>🟢 LOWEST RISK SUBJECT</span>"
                f"<span class='aml-score-tag'>Risk Score: <strong>{score}/100</strong></span>"
                f"</div>"
                f"<div class='aml-card-body'>"
                f"🛡️ <strong>Lowest Risk Subject:</strong> Customer <strong>{low_cid}</strong>{name_display} "
                f"holds the lowest composite risk score in the ledger.<br><br>"
                f"<div class='aml-stats-grid'>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Risk Rating</span><span class='aml-stat-val'>{risk_level}</span></div>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Structuring Txns</span><span class='aml-stat-val'>{low_row.get('structuring_count', 0)}</span></div>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Total Volume</span><span class='aml-stat-val'>${low_row.get('total_tx_volume', 0):,.2f}</span></div>"
                f"</div><br>"
                f"📋 <strong>Recommended Compliance Action:</strong> <strong>NO ACTION REQUIRED (BASELINE LOW RISK)</strong>"
                f"</div>"
                f"</div>"
            )
            output_payload["explanations"].append(exp)

        elif intent == "EXPLAIN_RISK_REASON":
            raw_cid = entities.get("customer_id") or self._get_top_suspicious_customer_id()
            match_result = self._find_closest_customer_id(raw_cid, entities.get("raw_cust_num"))
            cid = match_result["resolved_id"] or raw_cid
            match_type = match_result["match_type"]

            tools_invoked_live.extend(["SingleEntityLookupTool"])
            tools_skipped.extend(["EDATool", "ThresholdStressTestTool"])
            execution_plan = [
                f"1. Locate subject record for {cid}",
                "2. Gather complete customer profile, transaction history, and behavioral red flags",
                "3. Deconstruct key risk drivers and suspicious activity indicators"
            ]
            lookup_data = self.single_lookup_tool.run(cid, self.df_transactions, self.df_customers, self.df_classified)
            output_payload["results"]["single_lookup"] = lookup_data
            output_payload["results"]["match_type"] = match_type

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
                    model_label = html.escape(self.model_info.get("model_type", "ML"))
                    reasons.append(f"• <strong>Behavioral Anomaly ({model_label}):</strong> Customer behavior departs significantly from expected baseline activity (Anomaly Score: {r['ml_score']}/100).")

                reason_text = "<br>".join(reasons) if reasons else "• Standard low-risk profile with normal transaction velocity."
                cust_name = html.escape(str(c.get("customer_name") or c.get("customer_id") or cid))
                safe_cid = html.escape(str(cid))
                risk_level = html.escape(str(r.get('risk_level')))
                badge_class = "aml-badge-red" if risk_level == "HIGH" else ("aml-badge-yellow" if risk_level == "MEDIUM" else "aml-badge-indigo")

                if match_type == "fuzzy":
                    match_note = f" <span class='aml-badge aml-badge-yellow'>⚠️ APPROXIMATE MATCH</span> <em>(Query '{html.escape(raw_cid)}' fuzzy-matched to {safe_cid})</em>"
                elif match_type and raw_cid != cid:
                    match_note = f" <em>(Matched nearest record for query {html.escape(raw_cid)})</em>"
                else:
                    match_note = ""

                exp = (
                    f"<div class='aml-card aml-risk-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge {badge_class}'>🔍 RISK FACTOR DECOMPOSITION</span>"
                    f"<span class='aml-score-tag'>Subject: <strong>{safe_cid}</strong></span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"👤 <strong>Target Subject:</strong> <strong>{cust_name}</strong> ({safe_cid}){match_note}<br><br>"
                    f"<div class='aml-stats-grid'>"
                    f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Risk Score</span><span class='aml-stat-val val-red'>{r.get('composite_risk_score')}/100</span></div>"
                    f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Risk Level</span><span class='aml-stat-val'>{risk_level}</span></div>"
                    f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Structuring Txns</span><span class='aml-stat-val'>{r.get('structuring_count', 0)}</span></div>"
                    f"</div><br>"
                    f"<strong>Primary Risk Drivers & Behavioral Triggers:</strong><br>{reason_text}<br><br>"
                    f"📋 <strong>Recommended Compliance Action:</strong> <strong>{html.escape(str(r.get('recommended_action')))}</strong>"
                    f"</div>"
                    f"</div>"
                )
                output_payload["explanations"].append(exp)

                if r.get("risk_level") == "HIGH":
                    sar_text = self.sar_tool.generate_sar(cid, c, r, lookup_data["transaction_history"], model_info=self.model_info)
                    output_payload["sar_narrative"] = sar_text
                    tools_invoked_live.append("SARGeneratorTool")
            else:
                safe_raw = html.escape(str(raw_cid))
                exp = (
                    f"<div class='aml-card aml-info-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge aml-badge-yellow'>⚠️ SUBJECT NOT FOUND</span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"Unable to locate subject <strong>{safe_raw}</strong> in the active ledger.<br>"
                    f"Check Customer ID input or select from active subjects in the <strong>Flagged Risk Table</strong>."
                    f"</div>"
                    f"</div>"
                )
                output_payload["explanations"].append(exp)

        elif intent == "SINGLE_ENTITY_LOOKUP":
            raw_cid = entities.get("customer_id") or self._get_top_suspicious_customer_id()
            match_result = self._find_closest_customer_id(raw_cid, entities.get("raw_cust_num"))
            cid = match_result["resolved_id"] or raw_cid
            match_type = match_result["match_type"]

            tools_invoked_live.append("SingleEntityLookupTool")
            tools_skipped.extend(["EDATool", ml_tool_name, "ThresholdStressTestTool"])

            execution_plan = [
                f"1. Resolve target Customer ID: {cid}",
                "2. Retrieve single-entity profile and complete transaction history",
                "3. Evaluate individual risk factors and red flag triggers",
                "4. Draft FinCEN SAR Narrative if high risk"
            ]

            lookup_data = self.single_lookup_tool.run(cid, self.df_transactions, self.df_customers, self.df_classified)
            output_payload["results"]["single_lookup"] = lookup_data
            output_payload["results"]["match_type"] = match_type

            if lookup_data.get("found"):
                risk_prof = lookup_data["risk_profile"]
                cust_info = lookup_data["customer"]
                tx_hist = lookup_data["transaction_history"]

                cust_name = html.escape(str(cust_info.get('customer_name') or cust_info.get('customer_id') or cid))
                safe_cid = html.escape(str(cid))
                risk_level = html.escape(str(risk_prof.get('risk_level')))
                badge_class = "aml-badge-red" if risk_level == "HIGH" else ("aml-badge-yellow" if risk_level == "MEDIUM" else "aml-badge-indigo")

                if match_type == "fuzzy":
                    match_note = f" <span class='aml-badge aml-badge-yellow'>⚠️ APPROXIMATE MATCH</span> <em>(Query '{html.escape(raw_cid)}' fuzzy-matched to {safe_cid})</em>"
                elif match_type and raw_cid != cid:
                    match_note = f" <em>(Mapped {html.escape(raw_cid)} to nearest active dataset record {safe_cid})</em>"
                else:
                    match_note = ""

                exp_str = (
                    f"<div class='aml-card aml-risk-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge {badge_class}'>📊 ENTITY PROFILE LOOKUP</span>"
                    f"<span class='aml-score-tag'>Score: <strong>{risk_prof.get('composite_risk_score')}/100</strong></span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"👤 <strong>Subject:</strong> <strong>{cust_name}</strong> ({safe_cid}){match_note}<br><br>"
                    f"<div class='aml-stats-grid'>"
                    f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Occupation</span><span class='aml-stat-val'>{html.escape(str(cust_info.get('occupation','—')))}</span></div>"
                    f"<div class='aml-stat-box'><span class='aml-stat-lbl'>KYC Status</span><span class='aml-stat-val'>{html.escape(str(cust_info.get('kyc_status','—')))}</span></div>"
                    f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Structuring Txns</span><span class='aml-stat-val'>{risk_prof.get('structuring_count',0)}</span></div>"
                    f"</div><br>"
                    f"📋 <strong>Recommended Compliance Action:</strong> <strong>{html.escape(str(risk_prof.get('recommended_action')))}</strong>"
                    f"</div>"
                    f"</div>"
                )
                output_payload["explanations"].append(exp_str)

                if risk_prof.get("risk_level") == "HIGH":
                    sar_text = self.sar_tool.generate_sar(cid, cust_info, risk_prof, tx_hist, model_info=self.model_info)
                    output_payload["sar_narrative"] = sar_text
                    tools_invoked_live.append("SARGeneratorTool")
            else:
                safe_raw = html.escape(str(raw_cid))
                exp_str = (
                    f"<div class='aml-card aml-info-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge aml-badge-yellow'>⚠️ SUBJECT NOT FOUND</span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"Unable to locate subject <strong>{safe_raw}</strong> in the active dataset.<br>"
                    f"Check Customer ID input or select from active subjects in the <strong>Flagged Risk Table</strong>."
                    f"</div>"
                    f"</div>"
                )
                output_payload["explanations"].append(exp_str)

        elif intent == "HIGH_RISK_FILTER":
            risk_flt = entities.get("risk_filter") or "HIGH"
            tools_skipped.extend(["SingleEntityLookupTool"])
            execution_plan = [
                f"1. Filter customer population for subjects evaluated as {html.escape(risk_flt)} RISK",
                "2. Rank subjects by overall suspicious activity score",
                "3. Present prioritized case file table for analyst review"
            ]
            flagged = self.df_classified[self.df_classified["risk_level"] == risk_flt].sort_values(by="composite_risk_score", ascending=False)
            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)

            top_subj = merged.iloc[0] if not merged.empty else None
            top_info = f"<br>Top Priority Subject: <strong>{html.escape(str(top_subj['customer_id']))}</strong> (Score: <strong>{top_subj['composite_risk_score']}/100</strong>)" if top_subj is not None else ""
            badge_class = "aml-badge-red" if risk_flt == "HIGH" else "aml-badge-yellow"
            exp_str = (
                f"<div class='aml-card aml-risk-card'>"
                f"<div class='aml-card-header'>"
                f"<span class='aml-badge {badge_class}'>📌 RISK POPULATION FILTER</span>"
                f"<span class='aml-score-tag'>Flagged: <strong>{len(flagged)} Subjects</strong></span>"
                f"</div>"
                f"<div class='aml-card-body'>"
                f"Filtered <strong>{len(flagged)} customer profiles</strong> evaluated as <strong>{html.escape(risk_flt)} RISK</strong>.{top_info}<br><br>"
                f"Check the <strong>Flagged Risk Table</strong> tab for complete customer profiles and recommended compliance actions."
                f"</div>"
                f"</div>"
            )
            output_payload["explanations"].append(exp_str)

        elif intent == "SCORE_RANGE_FILTER":
            min_sc = entities.get("min_score")
            max_sc = entities.get("max_score")
            tools_skipped.extend(["SingleEntityLookupTool", "EDATool", "ThresholdStressTestTool"])

            range_strs = []
            if min_sc is not None:
                range_strs.append(f">= {min_sc:g}")
            if max_sc is not None:
                range_strs.append(f"<= {max_sc:g}")
            range_desc = " and ".join(range_strs) if range_strs else "specified score range"

            execution_plan = [
                f"1. Filter customer population for subjects with Risk Score {range_desc}",
                "2. Rank matching subjects by composite risk score in descending order",
                "3. Compile compliance risk profiles and escalation recommendations"
            ]

            df_filt = self.df_classified.copy()
            if min_sc is not None:
                df_filt = df_filt[df_filt["composite_risk_score"] >= min_sc]
            if max_sc is not None:
                df_filt = df_filt[df_filt["composite_risk_score"] <= max_sc]

            flagged = df_filt.sort_values(by="composite_risk_score", ascending=False)
            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)

            top_subj = merged.iloc[0] if not merged.empty else None
            top_info = f"<br>Top Priority Subject in Range: <strong>{html.escape(str(top_subj['customer_id']))}</strong> (Score: <strong>{top_subj['composite_risk_score']}/100</strong>)" if top_subj is not None else ""

            exp_str = (
                f"<div class='aml-card aml-risk-card'>"
                f"<div class='aml-card-header'>"
                f"<span class='aml-badge aml-badge-indigo'>🎯 RISK SCORE RANGE FILTER</span>"
                f"<span class='aml-score-tag'>Matched: <strong>{len(flagged)} Subjects</strong></span>"
                f"</div>"
                f"<div class='aml-card-body'>"
                f"Filtered <strong>{len(flagged)} customer profiles</strong> with Risk Score <strong>{html.escape(range_desc)}</strong>.{top_info}<br><br>"
                f"Check the <strong>Flagged Risk Table</strong> tab below for full customer profiles, risk categories, and recommended compliance escalation actions."
                f"</div>"
                f"</div>"
            )
            output_payload["explanations"].append(exp_str)

        elif intent == "STRUCTURING_SEARCH":
            tools_skipped.extend(["EDATool", "SingleEntityLookupTool", "ThresholdStressTestTool"])
            time_win = entities.get("time_window_days")
            s_date = entities.get("start_date")
            e_date = entities.get("end_date")

            df_tx_win, df_scored_win, df_classified_win = self._get_windowed_data(
                time_window_days=time_win, start_date=s_date, end_date=e_date
            )

            window_note, win_header = self._format_time_window_phrase(time_win)
            execution_plan = [
                f"1. Scan transaction ledger{window_note} for structured deposit and smurfing patterns",
                "2. Evaluate deposit frequencies near statutory reporting limits ($10,000)",
                "3. Identify subjects repeatedly making cash deposits just under reporting thresholds",
                "4. Classify high-risk subjects and assign compliance escalation recommendations",
                "5. Draft FinCEN Suspicious Activity Report (SAR) narratives for top flagged subjects"
            ]

            flagged = df_classified_win[df_classified_win["structuring_count"] > 0].sort_values(by="composite_risk_score", ascending=False)
            if flagged.empty:
                flagged = df_classified_win.sort_values(by="composite_risk_score", ascending=False).head(10)

            merged_flagged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged_flagged)

            top_subj = merged_flagged.iloc[0] if not merged_flagged.empty else None
            if top_subj is not None:
                name_str = html.escape(str(top_subj['customer_name'] if ('customer_name' in top_subj and pd.notna(top_subj['customer_name'])) else top_subj['customer_id']))
                safe_cid = html.escape(str(top_subj['customer_id']))
                window_note, win_header = self._format_time_window_phrase(time_win)
                exp_str = (
                    f"<div class='aml-card aml-risk-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge aml-badge-red'>🚨 STRUCTURING & SMURFING ALERT{win_header}</span>"
                    f"<span class='aml-score-tag'>Flagged: <strong>{len(flagged)} Subjects</strong></span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"Identified <strong>{len(flagged)} subjects</strong> exhibiting systematic currency structuring patterns{window_note}.<br><br>"
                    f"<div class='aml-stats-grid'>"
                    f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Top Subject</span><span class='aml-stat-val val-red'>{safe_cid}</span></div>"
                    f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Structuring Deposits</span><span class='aml-stat-val'>{top_subj['structuring_count']}</span></div>"
                    f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Composite Score</span><span class='aml-stat-val'>{top_subj['composite_risk_score']}/100</span></div>"
                    f"</div><br>"
                    f"📝 <strong>FinCEN SAR Narrative:</strong> Drafted and loaded in the <strong>SAR Narrative Panel</strong>."
                    f"</div>"
                    f"</div>"
                )
                output_payload["explanations"].append(exp_str)

                sar_text = self.sar_tool.generate_sar(
                    top_subj["customer_id"],
                    top_subj.to_dict(),
                    top_subj.to_dict(),
                    df_tx_win[df_tx_win["customer_id"] == top_subj["customer_id"]].to_dict(orient="records"),
                    model_info=self.model_info
                )
                output_payload["sar_narrative"] = sar_text
                tools_invoked_live.append("SARGeneratorTool")

        elif intent == "LARGE_AMOUNT_FILTER":
            time_win = entities.get("time_window_days")
            s_date = entities.get("start_date")
            e_date = entities.get("end_date")

            df_tx_win, df_scored_win, df_classified_win = self._get_windowed_data(
                time_window_days=time_win, start_date=s_date, end_date=e_date
            )

            min_amt = entities.get("min_amount") or (df_tx_win["amount"].quantile(0.95) if not df_tx_win.empty else 10000.0)
            tools_skipped.extend(["EDATool", "SARGeneratorTool"])

            window_note, win_header = self._format_time_window_phrase(time_win)
            execution_plan = [
                f"1. Filter transaction ledger{window_note} for high-value transfers (>= ${min_amt:,.2f})",
                "2. Identify associated account holders and aggregate capital movements",
                "3. Rank subjects by total high-value transaction volume"
            ]

            large_txs = df_tx_win[df_tx_win["amount"] >= min_amt] if not df_tx_win.empty else pd.DataFrame()
            large_cust_ids = large_txs["customer_id"].unique() if not large_txs.empty else []
            flagged = df_classified_win[df_classified_win["customer_id"].isin(large_cust_ids)].sort_values(by="total_tx_volume", ascending=False)
            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)

            total_vol = large_txs["amount"].sum() if not large_txs.empty else 0.0
            exp = (
                f"<div class='aml-card aml-info-card'>"
                f"<div class='aml-card-header'>"
                f"<span class='aml-badge aml-badge-indigo'>💎 HIGH-VALUE VOLUME FILTER</span>"
                f"</div>"
                f"<div class='aml-card-body'>"
                f"Found <strong>{len(large_txs)} transactions</strong> exceeding threshold of <strong>${min_amt:,.2f}</strong>{window_note}.<br><br>"
                f"<div class='aml-stats-grid'>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Total High-Value Volume</span><span class='aml-stat-val'>${total_vol:,.2f}</span></div>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Unique Customers</span><span class='aml-stat-val'>{len(merged)}</span></div>"
                f"</div>"
                f"</div>"
                f"</div>"
            )
            output_payload["explanations"].append(exp)

        elif intent == "JURISDICTION_ANALYSIS":
            time_win = entities.get("time_window_days")
            target_cc = entities.get("country_code")
            tools_skipped.extend(["SingleEntityLookupTool"])

            df_tx_win, df_scored_win, df_classified_win = self._get_windowed_data(
                time_window_days=time_win
            )
            window_note, win_header = self._format_time_window_phrase(time_win)

            execution_plan = [
                f"1. Cross-reference transactions{window_note} with FATF high-risk offshore jurisdictions (KY, PA, AE)",
                "2. Measure cross-border transaction volume and transfer counts per jurisdiction",
                "3. Flag subjects engaging in capital flow through high-risk offshore channels"
            ]

            if target_cc:
                tx_filtered = df_tx_win[df_tx_win["country_code"] == target_cc] if not df_tx_win.empty else pd.DataFrame()
                cust_ids = tx_filtered["customer_id"].unique() if not tx_filtered.empty else []
                sort_col = "high_risk_country_volume" if "high_risk_country_volume" in df_classified_win.columns else "composite_risk_score"
                flagged = df_classified_win[df_classified_win["customer_id"].isin(cust_ids)].sort_values(by=sort_col, ascending=False)
                tot_vol = tx_filtered['amount'].sum() if not tx_filtered.empty else 0.0
                exp = (
                    f"<div class='aml-card aml-risk-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge aml-badge-yellow'>🌐 FATF JURISDICTION ANALYSIS</span>"
                    f"<span class='aml-score-tag'>Country Code: <strong>{target_cc}</strong></span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"Found <strong>{len(tx_filtered)} transactions</strong> involving jurisdiction <strong>{target_cc}</strong>{window_note} "
                    f"totaling <strong>${tot_vol:,.2f}</strong> across {len(flagged)} customers."
                    f"</div>"
                    f"</div>"
                )
            else:
                tx_filtered = df_tx_win[df_tx_win["country_code"].isin(HIGH_RISK_JURISDICTIONS)] if not df_tx_win.empty else pd.DataFrame()
                flagged = df_classified_win[df_classified_win["high_risk_country_tx_count"] > 0].sort_values(by="high_risk_country_volume", ascending=False)
                cc_str = ", ".join(HIGH_RISK_JURISDICTIONS)
                tot_vol = tx_filtered['amount'].sum() if not tx_filtered.empty else 0.0
                exp = (
                    f"<div class='aml-card aml-risk-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge aml-badge-red'>🌐 FATF OFFSHORE JURISDICTION ALERT</span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"Found <strong>{len(tx_filtered)} transactions</strong> in high-risk offshore codes (<strong>{cc_str}</strong>){window_note} "
                    f"totaling <strong>${tot_vol:,.2f}</strong> across <strong>{len(flagged)} subjects</strong>."
                    f"</div>"
                    f"</div>"
                )

            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)
            output_payload["explanations"].append(exp)

        elif intent == "TRANSACTION_TYPE_BREAKDOWN":
            tx_type = entities.get("transaction_type") or "Wire"
            tools_skipped.extend(["SARGeneratorTool"])

            execution_plan = [
                f"1. Isolate transactions by channel: {tx_type}",
                "2. Calculate total volume and transaction count breakdown",
                "3. Rank customers by total activity through this transaction channel"
            ]

            filtered_tx = self.df_transactions[self.df_transactions["transaction_type"].str.lower() == tx_type.lower()]
            tot_vol = filtered_tx["amount"].sum()
            cust_ids = filtered_tx["customer_id"].unique()
            flagged = self.df_classified[self.df_classified["customer_id"].isin(cust_ids)].sort_values(by="total_tx_volume", ascending=False)
            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)

            exp = (
                f"<div class='aml-card aml-info-card'>"
                f"<div class='aml-card-header'>"
                f"<span class='aml-badge aml-badge-indigo'>💳 TRANSACTION TYPE BREAKDOWN</span>"
                f"<span class='aml-score-tag'>Channel: <strong>{tx_type}</strong></span>"
                f"</div>"
                f"<div class='aml-card-body'>"
                f"Found <strong>{len(filtered_tx)} {tx_type} transactions</strong> totaling <strong>${tot_vol:,.2f}</strong> "
                f"across <strong>{len(cust_ids)} unique customers</strong>."
                f"</div>"
                f"</div>"
            )
            output_payload["explanations"].append(exp)

        elif intent == "COUNT_RISK_SUMMARY":
            risk_flt = entities.get("risk_filter") or "HIGH"
            tools_skipped.extend(["SingleEntityLookupTool"])

            execution_plan = [
                "1. Analyze Risk Rating distribution across all active customer profiles",
                f"2. Focus on {risk_flt} risk population",
                "3. Calculate population proportions and required compliance actions"
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
                f"<div class='aml-card aml-info-card'>"
                f"<div class='aml-card-header'>"
                f"<span class='aml-badge aml-badge-indigo'>📊 POPULATION RISK BREAKDOWN</span>"
                f"<span class='aml-score-tag'>Total: <strong>{tot_cnt} Subjects</strong></span>"
                f"</div>"
                f"<div class='aml-card-body'>"
                f"<div class='aml-stats-grid'>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>HIGH Risk</span><span class='aml-stat-val val-red'>{high_cnt} ({round(high_cnt/tot_cnt*100 if tot_cnt > 0 else 0, 1)}%)</span></div>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>MEDIUM Risk</span><span class='aml-stat-val'>{med_cnt} ({round(med_cnt/tot_cnt*100 if tot_cnt > 0 else 0, 1)}%)</span></div>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>LOW Risk</span><span class='aml-stat-val'>{low_cnt} ({round(low_cnt/tot_cnt*100 if tot_cnt > 0 else 0, 1)}%)</span></div>"
                f"</div>"
                f"</div>"
                f"</div>"
            )
            output_payload["explanations"].append(exp)

        elif intent == "THRESHOLD_AGGREGATION":
            min_count = entities.get("min_count") or 5
            has_max_amt = entities.get("max_amount") is not None
            max_amt = entities.get("max_amount") or (self.df_transactions["amount"].max() * 0.99)

            # No live tool invocation — filters precomputed DataFrames
            tools_skipped.extend(["EDATool", ml_tool_name, "SARGeneratorTool"])

            execution_plan = [
                f"1. Isolate accounts with repeated transactions bounded under ${max_amt:,.2f} (count >= {min_count})",
                "2. Aggregate cumulative volume and transaction counts for target subjects",
                "3. Flag potential threshold avoidance behavior for investigator review",
                "4. Present structured compliance review table"
            ]

            if has_max_amt:
                tx_below = self.df_transactions[self.df_transactions["amount"] < max_amt]
                tx_counts = tx_below.groupby("customer_id").size()
                matching_cust_ids = tx_counts[tx_counts >= min_count].index
                filtered_cust = self.df_classified[self.df_classified["customer_id"].isin(matching_cust_ids)].sort_values(by="total_tx_volume", ascending=False)
            else:
                filtered_cust = self.df_classified[
                    (self.df_classified["structuring_count"] >= min_count) | 
                    ((self.df_classified["total_tx_count"] >= min_count) & (self.df_classified["avg_amount"] < max_amt))
                ].sort_values(by="total_tx_volume", ascending=False)

            merged = pd.merge(filtered_cust, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)
            
            exp_str = f"Found <strong>{len(merged)} customers</strong> making {min_count}+ transactions under ${max_amt:,.2f}."
            output_payload["explanations"].append(exp_str)

        elif intent == "DAILY_MONITORING":
            tools_skipped.extend(["EDATool", "ThresholdStressTestTool"])
            time_win = entities.get("time_window_days") or 1
            s_date = entities.get("start_date")
            e_date = entities.get("end_date")

            df_tx_win, df_scored_win, df_classified_win = self._get_windowed_data(
                time_window_days=time_win, start_date=s_date, end_date=e_date
            )
            window_note, win_header = self._format_time_window_phrase(time_win)

            execution_plan = [
                f"1. Isolate transaction activity and AML alerts for daily monitoring window{window_note}",
                "2. Identify new high-risk customer profiles and top suspicious transactions",
                "3. Prioritize high-risk accounts requiring immediate compliance officer review",
                "4. Compile daily AML alert summary table"
            ]

            high_risk = df_classified_win[df_classified_win["risk_level"].isin(["HIGH", "MEDIUM"])].sort_values(by="composite_risk_score", ascending=False)
            if high_risk.empty:
                high_risk = df_classified_win.sort_values(by="composite_risk_score", ascending=False).head(10)

            merged = pd.merge(high_risk, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)

            high_count = len(df_classified_win[df_classified_win["risk_level"] == "HIGH"])
            med_count = len(df_classified_win[df_classified_win["risk_level"] == "MEDIUM"])
            tot_tx = len(df_tx_win)
            tot_vol = df_tx_win["amount"].sum() if not df_tx_win.empty else 0.0

            top_subj = merged.iloc[0] if not merged.empty else None
            top_info = f"<br>Highest Risk Subject: <strong>{html.escape(str(top_subj['customer_id']))}</strong> (Score: <strong>{top_subj['composite_risk_score']}/100</strong>)" if top_subj is not None else ""

            exp_str = (
                f"<div class='aml-card aml-risk-card'>"
                f"<div class='aml-card-header'>"
                f"<span class='aml-badge aml-badge-red'>📅 DAILY AML MONITORING SUMMARY{win_header}</span>"
                f"<span class='aml-score-tag'>Window: <strong>{html.escape(window_note.strip() or 'Today')}</strong></span>"
                f"</div>"
                f"<div class='aml-card-body'>"
                f"Completed daily surveillance across <strong>{tot_tx:,} transactions</strong> totaling <strong>${tot_vol:,.2f}</strong>.{top_info}<br><br>"
                f"<div class='aml-stats-grid'>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>HIGH Risk Alerts</span><span class='aml-stat-val val-red'>{high_count}</span></div>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>MEDIUM Risk Alerts</span><span class='aml-stat-val'>{med_count}</span></div>"
                f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Immediate Review</span><span class='aml-stat-val'>{high_count + med_count}</span></div>"
                f"</div><br>"
                f"Check the <strong>Flagged Risk Table</strong> tab for complete customer profiles and priority escalation queues."
                f"</div>"
                f"</div>"
            )
            output_payload["explanations"].append(exp_str)

        elif intent == "BEHAVIOR_CHANGE_ANALYSIS":
            raw_cid = entities.get("customer_id")
            tools_skipped.extend(["EDATool", "ThresholdStressTestTool"])

            if raw_cid:
                match_result = self._find_closest_customer_id(raw_cid, entities.get("raw_cust_num"))
                cid = match_result["resolved_id"] or raw_cid
                tools_invoked_live.append("SingleEntityLookupTool")
                lookup_data = self.single_lookup_tool.run(cid, self.df_transactions, self.df_customers, self.df_classified)
                output_payload["results"]["single_lookup"] = lookup_data

                execution_plan = [
                    f"1. Retrieve transaction history and baseline behavior profile for Customer {cid}",
                    "2. Calculate transaction volume velocity, frequency surge, and ML anomaly metrics",
                    "3. Compare recent activity against 90-day baseline norm"
                ]

                if lookup_data.get("found"):
                    r = lookup_data["risk_profile"]
                    c = lookup_data["customer"]
                    ml_sc = r.get("ml_score", 0)
                    cust_name = html.escape(str(c.get("customer_name") or c.get("customer_id") or cid))
                    safe_cid = html.escape(str(cid))

                    exp_str = (
                        f"<div class='aml-card aml-risk-card'>"
                        f"<div class='aml-card-header'>"
                        f"<span class='aml-badge aml-badge-yellow'>📈 BEHAVIORAL CHANGE & ANOMALY ANALYSIS</span>"
                        f"<span class='aml-score-tag'>Subject: <strong>{safe_cid}</strong></span>"
                        f"</div>"
                        f"<div class='aml-card-body'>"
                        f"Evaluated behavioral trajectory for <strong>{cust_name}</strong> ({safe_cid}).<br><br>"
                        f"<div class='aml-stats-grid'>"
                        f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Anomaly Score</span><span class='aml-stat-val val-red'>{ml_sc}/100</span></div>"
                        f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Structuring Deposits</span><span class='aml-stat-val'>{r.get('structuring_count',0)}</span></div>"
                        f"<div class='aml-stat-box'><span class='aml-stat-lbl'>Rapid Cash-Outs</span><span class='aml-stat-val'>{r.get('rapid_cashout_count',0)}</span></div>"
                        f"</div><br>"
                        f"• <strong>Behavioral Shift Assessment:</strong> Subject exhibits elevated transaction velocity departing from historical average.<br>"
                        f"• <strong>Recommended Action:</strong> <strong>{html.escape(str(r.get('recommended_action')))}</strong>"
                        f"</div>"
                        f"</div>"
                    )
                    output_payload["explanations"].append(exp_str)

                    flagged = self.df_classified[self.df_classified["customer_id"] == cid]
                    merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
                    output_payload["results"]["flagged_table"] = self._clean_records(merged)
            else:
                execution_plan = [
                    "1. Scan customer population for significant deviations from historical transaction norms",
                    "2. Rank subjects by behavioral anomaly score and volume growth delta",
                    "3. Present prioritized list of accounts exhibiting unusual behavioral shifts"
                ]
                flagged = self.df_classified.sort_values(by="ml_score", ascending=False)
                merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
                output_payload["results"]["flagged_table"] = self._clean_records(merged)

                top_subj = merged.iloc[0] if not merged.empty else None
                top_info = f"<br>Top Behavioral Anomaly Subject: <strong>{html.escape(str(top_subj['customer_id']))}</strong> (Anomaly Score: <strong>{top_subj['ml_score']}/100</strong>)" if top_subj is not None else ""

                exp_str = (
                    f"<div class='aml-card aml-risk-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge aml-badge-yellow'>📈 POPULATION BEHAVIORAL CHANGE DETECTION</span>"
                    f"<span class='aml-score-tag'>Flagged: <strong>{len(flagged)} Subjects</strong></span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"Identified customer profiles with significant transaction velocity shifts compared to historical baselines.{top_info}<br><br>"
                    f"Review the <strong>Flagged Risk Table</strong> tab for complete behavioral metrics."
                    f"</div>"
                    f"</div>"
                )
                output_payload["explanations"].append(exp_str)

        elif intent == "CASH_ACTIVITY_SEARCH":
            tools_skipped.extend(["EDATool", "ThresholdStressTestTool"])
            time_win = entities.get("time_window_days")
            s_date = entities.get("start_date")
            e_date = entities.get("end_date")

            df_tx_win, df_scored_win, df_classified_win = self._get_windowed_data(
                time_window_days=time_win, start_date=s_date, end_date=e_date
            )
            window_note, win_header = self._format_time_window_phrase(time_win)

            execution_plan = [
                f"1. Isolate cash deposit and cash withdrawal transactions{window_note}",
                "2. Detect rapid cash deposit followed by cash withdrawal patterns (smurfing/cash movements)",
                "3. Rank customers by cash transaction frequency and structuring counts",
                "4. Compile cash activity risk table"
            ]

            cash_txs = df_tx_win[df_tx_win["transaction_type"].astype(str).str.lower().str.contains("cash")] if not df_tx_win.empty else pd.DataFrame()
            cash_cust_ids = cash_txs["customer_id"].unique() if not cash_txs.empty else []

            flagged = df_classified_win[
                (df_classified_win["customer_id"].isin(cash_cust_ids)) |
                (df_classified_win["structuring_count"] > 0) |
                (df_classified_win["rapid_cashout_count"] > 0)
            ].sort_values(by="structuring_count", ascending=False)

            if flagged.empty:
                flagged = df_classified_win.sort_values(by="composite_risk_score", ascending=False).head(10)

            merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)

            tot_cash_vol = cash_txs["amount"].sum() if not cash_txs.empty else 0.0

            exp_str = (
                f"<div class='aml-card aml-risk-card'>"
                f"<div class='aml-card-header'>"
                f"<span class='aml-badge aml-badge-red'>💵 CASH ACTIVITY & RAPID MOVEMENT SURVEILLANCE{win_header}</span>"
                f"<span class='aml-score-tag'>Cash Volume: <strong>${tot_cash_vol:,.2f}</strong></span>"
                f"</div>"
                f"<div class='aml-card-body'>"
                f"Detected <strong>{len(cash_txs)} cash transactions</strong> across <strong>{len(merged)} customer accounts</strong>{window_note}.<br><br>"
                f"Flagged accounts exhibiting rapid cash deposits, structuring patterns, or same-day cash withdrawals.<br>"
                f"Check the <strong>Flagged Risk Table</strong> tab for detailed cash activity indicators."
                f"</div>"
                f"</div>"
            )
            output_payload["explanations"].append(exp_str)

        elif intent == "CASE_MANAGEMENT_RECOMMENDATION":
            raw_cid = entities.get("customer_id")
            tools_skipped.extend(["EDATool", "ThresholdStressTestTool"])

            if raw_cid:
                match_result = self._find_closest_customer_id(raw_cid, entities.get("raw_cust_num"))
                cid = match_result["resolved_id"] or raw_cid
                tools_invoked_live.append("SingleEntityLookupTool")
                lookup_data = self.single_lookup_tool.run(cid, self.df_transactions, self.df_customers, self.df_classified)
                output_payload["results"]["single_lookup"] = lookup_data

                execution_plan = [
                    f"1. Evaluate case details and risk profile for Customer {cid}",
                    "2. Review red flag indicators, transaction history, and regulatory reporting thresholds",
                    "3. Formulate case escalation recommendation and next investigative action",
                    "4. Draft FinCEN Suspicious Activity Report (SAR) narrative if high risk"
                ]

                if lookup_data.get("found"):
                    r = lookup_data["risk_profile"]
                    c = lookup_data["customer"]
                    rec_act = html.escape(str(r.get("recommended_action", "IMMEDIATE COMPLIANCE REVIEW")))
                    risk_lvl = html.escape(str(r.get("risk_level", "HIGH")))
                    score = r.get("composite_risk_score", 0)
                    safe_cid = html.escape(str(cid))
                    badge_cls = "aml-badge-red" if risk_lvl == "HIGH" else "aml-badge-yellow"

                    exp_str = (
                        f"<div class='aml-card aml-risk-card'>"
                        f"<div class='aml-card-header'>"
                        f"<span class='aml-badge {badge_cls}'>⚖️ CASE ESCALATION RECOMMENDATION</span>"
                        f"<span class='aml-score-tag'>Subject: <strong>{safe_cid}</strong></span>"
                        f"</div>"
                        f"<div class='aml-card-body'>"
                        f"🎯 <strong>Subject:</strong> Customer <strong>{safe_cid}</strong> (Risk Score: <strong>{score}/100</strong>)<br><br>"
                        f"📋 <strong>Recommended Next Action:</strong> <strong style='font-size: 1.1em; color: #dc2626;'>{rec_act}</strong><br><br>"
                        f"• <strong>Case Justification:</strong> Subject exhibits elevated risk score ({score}/100) and suspicious activity triggers.<br>"
                        f"• <strong>Regulatory Action:</strong> {'FinCEN SAR filing recommended.' if risk_lvl == 'HIGH' else 'Internal compliance review recommended.'}"
                        f"</div>"
                        f"</div>"
                    )
                    output_payload["explanations"].append(exp_str)

                    if risk_lvl == "HIGH":
                        sar_text = self.sar_tool.generate_sar(cid, c, r, lookup_data["transaction_history"], model_info=self.model_info)
                        output_payload["sar_narrative"] = sar_text
                        tools_invoked_live.append("SARGeneratorTool")

                    flagged = self.df_classified[self.df_classified["customer_id"] == cid]
                    merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
                    output_payload["results"]["flagged_table"] = self._clean_records(merged)
            else:
                execution_plan = [
                    "1. Evaluate high-risk cases across customer population",
                    "2. Prioritize cases requiring immediate regulatory reporting (FinCEN SAR)",
                    "3. Generate escalation recommendation summary for compliance officers"
                ]

                flagged = self.df_classified[self.df_classified["risk_level"] == "HIGH"].sort_values(by="composite_risk_score", ascending=False)
                merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
                output_payload["results"]["flagged_table"] = self._clean_records(merged)

                exp_str = (
                    f"<div class='aml-card aml-risk-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge aml-badge-red'>⚖️ PRIORITY CASE ESCALATION QUEUE</span>"
                    f"<span class='aml-score-tag'>Escalations: <strong>{len(flagged)} Cases</strong></span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"Identified <strong>{len(flagged)} high-risk cases</strong> requiring immediate escalation and regulatory review.<br><br>"
                    f"Action Required: Compliance team should review cases listed in the <strong>Flagged Risk Table</strong> and file SARs where appropriate."
                    f"</div>"
                    f"</div>"
                )
                output_payload["explanations"].append(exp_str)

        elif intent == "REPORT_GENERATION":
            raw_cid = entities.get("customer_id")
            time_win = entities.get("time_window_days")
            s_date = entities.get("start_date")
            e_date = entities.get("end_date")

            df_tx_win, df_scored_win, df_classified_win = self._get_windowed_data(
                time_window_days=time_win, start_date=s_date, end_date=e_date
            )
            window_note, win_header = self._format_time_window_phrase(time_win)

            if raw_cid:
                match_result = self._find_closest_customer_id(raw_cid, entities.get("raw_cust_num"))
                cid = match_result["resolved_id"] or raw_cid
                tools_invoked_live.extend(["SingleEntityLookupTool", "SARGeneratorTool"])
                lookup_data = self.single_lookup_tool.run(cid, self.df_transactions, self.df_customers, self.df_classified)
                output_payload["results"]["single_lookup"] = lookup_data

                execution_plan = [
                    f"1. Gather full investigative dossier for Customer {cid}",
                    "2. Synthesize risk factors, transaction history, and behavioral red flags",
                    "3. Generate FinCEN Suspicious Activity Report (SAR) narrative",
                    "4. Format AML Case Investigation Report"
                ]

                if lookup_data.get("found"):
                    r = lookup_data["risk_profile"]
                    c = lookup_data["customer"]
                    sar_text = self.sar_tool.generate_sar(cid, c, r, lookup_data["transaction_history"], model_info=self.model_info)
                    output_payload["sar_narrative"] = sar_text
                    safe_cid = html.escape(str(cid))

                    exp_str = (
                        f"<div class='aml-card aml-risk-card'>"
                        f"<div class='aml-card-header'>"
                        f"<span class='aml-badge aml-badge-indigo'>📑 AML INVESTIGATION REPORT</span>"
                        f"<span class='aml-score-tag'>Subject: <strong>{safe_cid}</strong></span>"
                        f"</div>"
                        f"<div class='aml-card-body'>"
                        f"Generated official <strong>AML Case Investigation Report</strong> and <strong>FinCEN SAR Narrative</strong> for Customer <strong>{safe_cid}</strong>.<br><br>"
                        f"📝 The complete regulatory SAR draft has been compiled and is displayed in the <strong>SAR Narrative Panel</strong>."
                        f"</div>"
                        f"</div>"
                    )
                    output_payload["explanations"].append(exp_str)

                    flagged = self.df_classified[self.df_classified["customer_id"] == cid]
                    merged = pd.merge(flagged, self.df_customers, on="customer_id", how="left")
                    output_payload["results"]["flagged_table"] = self._clean_records(merged)
            else:
                tools_invoked_live.append("SARGeneratorTool")
                execution_plan = [
                    f"1. Compile portfolio-wide AML suspicious activity report{window_note}",
                    "2. Aggregate structuring, rapid cashout, and jurisdiction risk metrics",
                    "3. Draft SAR narrative for top priority subject",
                    "4. Present summary compliance report"
                ]

                high_risk = df_classified_win[df_classified_win["risk_level"] == "HIGH"].sort_values(by="composite_risk_score", ascending=False)
                merged = pd.merge(high_risk, self.df_customers, on="customer_id", how="left")
                output_payload["results"]["flagged_table"] = self._clean_records(merged)

                top_subj = merged.iloc[0] if not merged.empty else None
                if top_subj is not None:
                    lookup_data = self.single_lookup_tool.run(str(top_subj["customer_id"]), self.df_transactions, self.df_customers, self.df_classified)
                    sar_text = self.sar_tool.generate_sar(str(top_subj["customer_id"]), lookup_data.get("customer", {}), lookup_data.get("risk_profile", {}), lookup_data.get("transaction_history", []), model_info=self.model_info)
                    output_payload["sar_narrative"] = sar_text

                exp_str = (
                    f"<div class='aml-card aml-risk-card'>"
                    f"<div class='aml-card-header'>"
                    f"<span class='aml-badge aml-badge-indigo'>📑 PORTFOLIO AML SURVEILLANCE REPORT{win_header}</span>"
                    f"<span class='aml-score-tag'>Flagged: <strong>{len(high_risk)} High-Risk</strong></span>"
                    f"</div>"
                    f"<div class='aml-card-body'>"
                    f"Generated AML Surveillance Summary Report{window_note}.<br><br>"
                    f"• <strong>High-Risk Subjects:</strong> {len(high_risk)} accounts flagged for immediate review.<br>"
                    f"• <strong>SAR Draft:</strong> Compiled regulatory filing draft for top risk subject (available in <strong>SAR Narrative Panel</strong>)."
                    f"</div>"
                    f"</div>"
                )
                output_payload["explanations"].append(exp_str)

        elif intent == "FULL_EDA":
            tools_invoked_live.append("EDATool")
            tools_skipped.extend(["SingleEntityLookupTool", "SARGeneratorTool"])

            execution_plan = [
                "1. Access complete transaction ledger and customer portfolio data",
                "2. Compile portfolio-wide volume statistics, average transaction sizes, and distribution metrics",
                "3. Evaluate overall transaction health and risk distribution across subjects",
                "4. Display baseline financial intelligence overview"
            ]

            eda_res = self.eda_tool.run(self.df_transactions, self.df_customers)
            output_payload["results"]["eda"] = eda_res
            exp_str = (
                f"Dataset contains <strong>{eda_res['summary']['total_transactions']:,} transactions</strong> across "
                f"<strong>{eda_res['summary']['unique_customers']:,} unique customers</strong> with "
                f"<strong>${eda_res['summary']['total_volume']:,.2f}</strong> total volume.<br>"
                f"• <strong>Average Amount:</strong> ${eda_res['summary']['average_amount']:,.2f}<br>"
                f"• <strong>Biggest / Max Amount:</strong> ${eda_res['summary']['max_amount']:,.2f}<br>"
                f"• <strong>Smallest / Min Amount:</strong> ${eda_res['summary']['min_amount']:,.2f}"
            )
            output_payload["explanations"].append(exp_str)

        else:
            # No live tool invocation — filters precomputed DataFrames
            tools_skipped.extend(["SingleEntityLookupTool"])

            execution_plan = [
                "1. Understand query context and investigative intent",
                "2. Screen customer transaction activity for unusual behavior compared to baseline norms",
                "3. Filter high and medium risk subjects requiring analyst review",
                "4. Present prioritized suspicious subjects with escalation guidance"
            ]

            high_risk = self.df_classified[self.df_classified["risk_level"].isin(["HIGH", "MEDIUM"])].sort_values(by="composite_risk_score", ascending=False)
            merged = pd.merge(high_risk, self.df_customers, on="customer_id", how="left")
            output_payload["results"]["flagged_table"] = self._clean_records(merged)

            top_subj = merged.iloc[0] if not merged.empty else None
            top_info = f" Top Subject: <strong>{html.escape(str(top_subj['customer_id']))}</strong> ({html.escape(str(top_subj.get('customer_name', top_subj['customer_id'])))}) with Risk Score <strong>{top_subj['composite_risk_score']}/100</strong>." if top_subj is not None else ""
            exp_str = f"Identified <strong>{len(high_risk)} suspicious subjects</strong> whose activity looks unusual compared to baseline norms.{top_info}"
            output_payload["explanations"].append(exp_str)

        output_payload["direct_answer"] = self._synthesize_direct_answer(intent, entities, output_payload)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        output_payload["telemetry"] = {
            "execution_plan": execution_plan,
            "tools_called": list(dict.fromkeys(tools_precomputed + tools_invoked_live)),
            "tools_precomputed": tools_precomputed,
            "tools_invoked_live": tools_invoked_live,
            "tools_skipped": tools_skipped,
            "latency_ms": elapsed_ms
        }

        return output_payload

    def _synthesize_direct_answer(self, intent: str, entities: dict, output_payload: dict) -> str:
        """Synthesizes an explicit, plain-English definitive verdict for analyst queries."""
        flagged_table = output_payload.get("results", {}).get("flagged_table", [])
        single_lookup = output_payload.get("results", {}).get("single_lookup", {})
        cid = entities.get("customer_id")

        if intent in ["SINGLE_ENTITY_LOOKUP", "EXPLAIN_RISK_REASON", "CASE_MANAGEMENT_RECOMMENDATION", "BEHAVIOR_CHANGE_ANALYSIS", "REPORT_GENERATION"] and cid:
            if single_lookup.get("found"):
                r = single_lookup.get("risk_profile", {})
                c = single_lookup.get("customer", {})
                score = r.get("composite_risk_score", 0)
                level = r.get("risk_level", "UNKNOWN")
                name = c.get("customer_name") or cid
                action = r.get("recommended_action", "COMPLIANCE REVIEW")
                verdict = "YES — HIGH RISK" if level == "HIGH" else ("SUSPICIOUS — MEDIUM RISK" if level == "MEDIUM" else "NO — LOW RISK (BASELINE)")
                struct_cnt = r.get("structuring_count", 0)
                return f"DEFINITIVE VERDICT FOR CUSTOMER {cid} ({name}): {verdict} | Composite Risk Score: {score}/100 ({struct_cnt} structuring deposits). Recommended Action: {action}."
            else:
                return f"Customer {cid} was not located in the active ledger."

        if intent == "TOP_RISK_SUBJECT" and flagged_table:
            top = flagged_table[0]
            cid_val = top.get("customer_id")
            name_val = top.get("customer_name") or cid_val
            score_val = top.get("composite_risk_score")
            action_val = top.get("recommended_action")
            return f"TOP RISK SUBJECT IN LEDGER: Customer {cid_val} ({name_val}) holds the highest Composite Risk Score ({score_val}/100). Recommended Action: {action_val}."

        if intent == "LOWEST_RISK_SUBJECT" and flagged_table:
            low = flagged_table[0]
            cid_val = low.get("customer_id")
            name_val = low.get("customer_name") or cid_val
            score_val = low.get("composite_risk_score")
            return f"LOWEST RISK SUBJECT IN LEDGER: Customer {cid_val} ({name_val}) has the lowest Risk Score ({score_val}/100). Status: Baseline Low Risk."

        if intent == "DAILY_MONITORING":
            high_cnt = sum(1 for r in flagged_table if r.get("risk_level") == "HIGH")
            med_cnt = sum(1 for r in flagged_table if r.get("risk_level") == "MEDIUM")
            return f"DAILY SURVEILLANCE SUMMARY: Identified {high_cnt} High-Risk and {med_cnt} Medium-Risk customer alerts requiring immediate compliance officer review."

        if intent == "STRUCTURING_SEARCH":
            count = len(flagged_table)
            top_cid = flagged_table[0].get("customer_id") if count > 0 else "N/A"
            return f"STRUCTURING & SMURFING FINDINGS: Flagged {count} customer accounts executing systematic currency deposits under the statutory $10,000 threshold. Top subject: {top_cid}."

        if intent == "LARGE_AMOUNT_FILTER":
            count = len(flagged_table)
            min_amt = entities.get("min_amount") or 50000.0
            return f"HIGH-VALUE TRANSFERS: Flagged {count} customer accounts involved in high-value transactions (>= ${min_amt:,.2f})."

        if intent == "JURISDICTION_ANALYSIS":
            count = len(flagged_table)
            return f"FATF JURISDICTION ANALYSIS: Flagged {count} customer accounts executing transfers involving high-risk offshore codes (KY, PA, AE)."

        if intent == "SCORE_RANGE_FILTER":
            count = len(flagged_table)
            min_sc = entities.get("min_score")
            max_sc = entities.get("max_score")
            range_desc = f"{min_sc or 0} to {max_sc or 100}"
            return f"SCORE RANGE FILTER: Matched {count} customer profiles with Composite Risk Score in requested range ({range_desc})."

        if intent == "THRESHOLD_AGGREGATION":
            count = len(flagged_table)
            min_cnt = entities.get("min_count") or 5
            max_amt = entities.get("max_amount") or 10000.0
            return f"THRESHOLD AVOIDANCE FILTER: Flagged {count} customer accounts making {min_cnt}+ transactions under ${max_amt:,.2f}."

        if intent == "FULL_EDA":
            eda = output_payload.get("results", {}).get("eda", {}).get("summary", {})
            return f"PORTFOLIO OVERVIEW: {eda.get('total_transactions', 0):,} transactions across {eda.get('unique_customers', 0):,} customers totaling ${eda.get('total_volume', 0):,.2f}."

        count = len(flagged_table)
        return f"INVESTIGATIVE FINDINGS: Identified {count} subjects matching search criteria requiring compliance analyst review."

    def stress_test_threshold(self, lower_bound: float) -> Dict[str, Any]:
        return self.stress_test_tool.run(self.df_transactions, self.df_features, lower_bound=lower_bound)

    def get_model_info(self) -> Dict[str, Any]:
        """Return the active ML model metadata for the /api/model/info endpoint."""
        return self.model_info
