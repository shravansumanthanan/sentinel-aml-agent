import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.nlp_parser import NLPIntentParser
from app.ml_model import SupervisedAMLClassifier, FEATURE_COLS
from app.kaggle_loader import load_and_merge_kaggle_datasets
from app.agent import AMLAgentOrchestrator
from app.main import (
    app,
    summary_endpoint,
    model_info_endpoint,
    chat_endpoint,
    stress_test_endpoint,
    upload_dataset_endpoint,
    ChatRequest,
    StressTestRequest,
)
from fastapi import HTTPException, UploadFile
import io
import asyncio


class TestSentinelAMLNLPParsing(unittest.TestCase):
    """Unit tests for NLP Intent and Entity Extraction."""

    def setUp(self):
        self.parser = NLPIntentParser()

    def test_top_risk_subject_intent(self):
        res = self.parser.parse_query("Which customer has the highest risk score?")
        self.assertEqual(res["intent"], "TOP_RISK_SUBJECT")

    def test_structuring_search_intent(self):
        res = self.parser.parse_query("Find structuring patterns in the last 30 days")
        self.assertEqual(res["intent"], "STRUCTURING_SEARCH")

    def test_single_entity_lookup_intent(self):
        res = self.parser.parse_query("Is customer CUST-0150 suspicious?")
        self.assertEqual(res["intent"], "SINGLE_ENTITY_LOOKUP")
        self.assertEqual(res["entities"]["customer_id"], "CUST-0150")

    def test_jurisdiction_analysis_intent(self):
        res = self.parser.parse_query("Show transactions in FATF high risk countries")
        self.assertEqual(res["intent"], "JURISDICTION_ANALYSIS")

    def test_nlp_intent_edge_cases(self):
        """Parametrized subtests for query variations, uppercase terms, and entity edge cases."""
        cases = [
            ("Find STRUCTURING patterns", "STRUCTURING_SEARCH", {"pattern_type": "STRUCTURING"}),
            ("check cust 0150", "SINGLE_ENTITY_LOOKUP", {"customer_id": "CUST-0150"}),
            ("Show transactions in FATF grey list jurisdictions", "JURISDICTION_ANALYSIS", {}),
            ("Filter by GRAY LIST countries", "JURISDICTION_ANALYSIS", {}),
            ("Analyze transactions over 50k", "LARGE_AMOUNT_FILTER", {"min_amount": 50000.0}),
            ("Is cust 420 suspicious?", "SINGLE_ENTITY_LOOKUP", {"customer_id": "CUST-0420"}),
            ("Which customers made more than 10 transactions under £10,000?", "THRESHOLD_AGGREGATION", {"min_count": 10, "max_amount": 10000.0, "min_amount": None}),
            ("Which customers made more than 5 transactions under €5,000?", "THRESHOLD_AGGREGATION", {"min_count": 5, "max_amount": 5000.0, "min_amount": None}),
            ("Which customers made more than 12 transactions under ₹50,000?", "THRESHOLD_AGGREGATION", {"min_count": 12, "max_amount": 50000.0, "min_amount": None}),
            ("Average transaction amount?", "FULL_EDA", {}),
            ("what is the mean transaction size?", "FULL_EDA", {}),
            ("overall avg amount", "FULL_EDA", {}),
            ("Biggest transaction?", "FULL_EDA", {}),
            ("Largest transaction?", "FULL_EDA", {}),
            ("Smallest transaction?", "FULL_EDA", {}),
            ("Maximum transaction size", "FULL_EDA", {}),
            ("Which countries?", "JURISDICTION_ANALYSIS", {}),
            ("Which jurisdictions?", "JURISDICTION_ANALYSIS", {}),
            ("Which customer has the lowest risk score?", "LOWEST_RISK_SUBJECT", {}),
            ("generate SAR for user 12", "SAR_GENERATION", {"customer_id": "CUST-0012"}),
            ("run stress test at $9500", "STRESS_TEST", {"stress_bound": 9500.0}),
            ("rapid cash out velocity alerts in past month", "VELOCITY_SEARCH", {"pattern_type": "RAPID_CASHOUT", "time_window_days": 30}),
        ]
        for query, expected_intent, expected_entities in cases:
            with self.subTest(query=query):
                res = self.parser.parse_query(query)
                self.assertEqual(res["intent"], expected_intent, f"Failed intent for query: {query}")
                for key, val in expected_entities.items():
                    self.assertEqual(res["entities"][key], val, f"Failed entity '{key}' for query: {query}")


class TestSupervisedAMLClassifier(unittest.TestCase):
    """Unit tests for ML Classifier (Hybrid Supervised & Unsupervised)."""

    def setUp(self):
        self.test_cache = os.path.join(PROJECT_ROOT, "data", "ci_test_cache")
        self.clf = SupervisedAMLClassifier(model_cache_dir=self.test_cache)

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_cache):
            shutil.rmtree(self.test_cache)

    def test_unsupervised_isolation_forest_fallback(self):
        np.random.seed(42)
        n = 100
        df_feat = pd.DataFrame(np.random.randn(n, len(FEATURE_COLS)), columns=FEATURE_COLS)
        df_feat["customer_id"] = [f"CUST-{i:04d}" for i in range(n)]

        info = self.clf.fit(df_feat, labels=None)
        self.assertFalse(info["is_supervised"])
        self.assertEqual(info["model_type"], "IsolationForest")

        scores = self.clf.score(df_feat)
        self.assertEqual(len(scores), n)
        self.assertTrue(all(0.0 <= s <= 100.0 for s in scores))

    def test_supervised_hybrid_composite_scoring(self):
        np.random.seed(42)
        n = 200
        df_feat = pd.DataFrame(np.random.randn(n, len(FEATURE_COLS)), columns=FEATURE_COLS)
        df_feat["customer_id"] = [f"CUST-{i:04d}" for i in range(n)]
        labels = pd.Series(np.random.choice([0, 1], size=n, p=[0.85, 0.15]), index=df_feat["customer_id"])

        info = self.clf.fit(df_feat, labels=labels)
        self.assertTrue(info["is_supervised"])
        self.assertIn("auc_roc", info)

        scores = self.clf.score(df_feat)
        self.assertEqual(len(scores), n)
        self.assertTrue(all(0.0 <= s <= 100.0 for s in scores))


class TestFastAPIEndpoints(unittest.TestCase):
    """Integration tests for FastAPI endpoints."""

    def test_dataset_summary_endpoint(self):
        res = summary_endpoint()
        self.assertIn("summary", res)
        self.assertIn("total_transactions", res["summary"])

    def test_model_info_endpoint(self):
        info = model_info_endpoint()
        self.assertIn("model_type", info)
        self.assertIn("is_supervised", info)

    def test_chat_endpoint_execution(self):
        req = ChatRequest(query="Which customer has the highest risk score?")
        res = chat_endpoint(req)
        self.assertEqual(res["parsed_intent"], "TOP_RISK_SUBJECT")
        self.assertIn("telemetry", res)
        self.assertIn("tools_called", res["telemetry"])

    def test_stress_test_endpoint(self):
        req = StressTestRequest(lower_bound=8500.0)
        res = stress_test_endpoint(req)
        self.assertEqual(res["lower_bound"], 8500.0)
        self.assertIn("interpretation", res)

    def test_upload_dataset_endpoint_validation(self):
        """Validates that non-CSV uploads raise 400 Bad Request."""
        invalid_file = UploadFile(filename="invalid.txt", file=io.BytesIO(b"dummy data"))
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(upload_dataset_endpoint(transactions_file=invalid_file))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn(".csv file", ctx.exception.detail)


class TestAIRegressionGuard(unittest.TestCase):
    """
    AI Regression Guard Suite.
    Catches AI-specific blind spots: payload contract drift, missing telemetry keys,
    upload contract parity, and cross-platform data generator path bugs.
    """

    def test_ai_regression_response_payload_contract(self):
        """BUG-REGRESSION: Ensures AI edits never silently drop mandatory top-level API keys."""
        req = ChatRequest(query="Find structuring patterns in the last 30 days")
        res = chat_endpoint(req)

        required_keys = ["query", "parsed_intent", "extracted_entities", "telemetry", "results", "explanations", "sar_narrative"]
        for key in required_keys:
            self.assertIn(key, res, f"AI Regression Guard: Payload missing mandatory key '{key}'")

    def test_ai_regression_telemetry_keys(self):
        """BUG-REGRESSION: Telemetry must contain explicit execution plan, tools called/skipped, and numeric latency."""
        req = ChatRequest(query="Is CUST-0001 suspicious?")
        res = chat_endpoint(req)
        telemetry = res["telemetry"]

        self.assertIn("execution_plan", telemetry)
        self.assertIn("tools_called", telemetry)
        self.assertIn("tools_skipped", telemetry)
        self.assertIn("latency_ms", telemetry)
        self.assertIsInstance(telemetry["latency_ms"], (int, float))

    def test_ai_regression_upload_payload_contract(self):
        """BUG-REGRESSION: Ensures CSV upload endpoint always returns required top-level contract keys."""
        import shutil
        tx_path = os.path.join(PROJECT_ROOT, "data", "transactions.csv")
        cust_path = os.path.join(PROJECT_ROOT, "data", "customers.csv")
        tx_bak = tx_path + ".bak"
        cust_bak = cust_path + ".bak"
        if os.path.exists(tx_path): shutil.copy(tx_path, tx_bak)
        if os.path.exists(cust_path): shutil.copy(cust_path, cust_bak)

        try:
            sample_csv = b"transaction_id,customer_id,timestamp,amount,transaction_type,channel,destination_account,country_code,is_laundering\nTX-001,CUST-001,2026-01-01 10:00:00,5000.0,Deposit,Branch,ACC-1,US,0\n"
            tx_file = UploadFile(filename="transactions.csv", file=io.BytesIO(sample_csv))
            res = asyncio.run(upload_dataset_endpoint(transactions_file=tx_file))

            required_keys = ["status", "filename", "total_transactions", "unique_customers", "active_model", "is_supervised", "message"]
            for key in required_keys:
                self.assertIn(key, res, f"AI Regression Guard: Upload response missing mandatory contract key '{key}'")
            self.assertEqual(res["status"], "success")
        finally:
            if os.path.exists(tx_bak):
                shutil.move(tx_bak, tx_path)
            if os.path.exists(cust_bak):
                shutil.move(cust_bak, cust_path)
            from app.main import _get_orchestrator
            _get_orchestrator().load_data()

    def test_ai_regression_synthetic_data_generator_parity(self):
        """BUG-REGRESSION: data_generator must run cleanly without throwing path or permission errors."""
        from data_generator import generate_aml_dataset
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            generate_aml_dataset(data_dir=temp_dir, num_customers=50, num_transactions=200)
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "customers.csv")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "transactions.csv")))
        finally:
            shutil.rmtree(temp_dir)

    def test_ai_regression_entity_lookup_fallback(self):
        """BUG-REGRESSION: Single entity lookup for missing subject returns non-empty explanations."""
        req = ChatRequest(query="Is customer CUST-999999 suspicious?")
        res = chat_endpoint(req)
        self.assertEqual(res["parsed_intent"], "SINGLE_ENTITY_LOOKUP")
        self.assertTrue(len(res["explanations"]) > 0, "Explanations should not be empty for unmapped customer ID")

    def test_ai_regression_telemetry_tools_accuracy(self):
        """BUG-REGRESSION: Telemetry tools_called should not contain duplicate entries."""
        req = ChatRequest(query="Which customer has the highest risk score?")
        res = chat_endpoint(req)
        called = res["telemetry"]["tools_called"]
        self.assertEqual(len(called), len(set(called)), "tools_called list must contain unique tool entries")


if __name__ == "__main__":
    unittest.main()


