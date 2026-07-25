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
    ChatRequest,
    StressTestRequest,
)


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
        labels = pd.Series(np.random.choice([0, 1], size=n, p=[0.85, 0.15]), index=df_feat.index)

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


if __name__ == "__main__":
    unittest.main()
