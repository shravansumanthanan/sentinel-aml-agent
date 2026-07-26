"""
SENTINEL-AML — Supervised ML Classifier
========================================
Dual-mode AML risk scorer:
  • SUPERVISED: XGBoost trained on IBM AML `is_laundering` labels
  • UNSUPERVISED: Isolation Forest fallback when no labels are available

Model and scaler are cached to <data_dir>/model_cache/ via joblib.
Cache is invalidated automatically when the dataset size changes >10%.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, Any, List

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Feature columns shared with HybridAnomalyTool ──────────────────────────
FEATURE_COLS: List[str] = [
    "total_tx_count",
    "total_tx_volume",
    "avg_amount",
    "std_amount",
    "structuring_count",
    "structuring_ratio",
    "rapid_cashout_count",
    "high_risk_country_tx_count",
    "high_risk_country_volume",
    "distinct_destination_accounts",
]

# Minimum number of labeled positive samples required to enable supervised mode
MIN_POSITIVE_SAMPLES = 10

# Cap training rows to avoid OOM on very large Kaggle datasets
MAX_TRAIN_ROWS = 500_000


class SupervisedAMLClassifier:
    """
    Dual-mode AML risk scorer with on-disk caching.

    Usage
    -----
    clf = SupervisedAMLClassifier(model_cache_dir="data/model_cache")
    model_info = clf.fit_or_load(df_features, customer_labels)  # train or load cache
    scores_0_100 = clf.score(df_features)                        # [0, 100]
    """

    def __init__(self, model_cache_dir: str = "data/model_cache"):
        self.model_cache_dir = model_cache_dir
        os.makedirs(model_cache_dir, exist_ok=True)

        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.is_supervised = False
        self.model_info: Dict[str, Any] = {}

        self._model_path = os.path.join(model_cache_dir, "aml_supervised_model.pkl")
        self._scaler_path = os.path.join(model_cache_dir, "aml_scaler.pkl")
        self._meta_path = os.path.join(model_cache_dir, "aml_model_meta.json")

    # ── Cache helpers (Audit Steps 2 & 3) ──────────────────────────────────

    @staticmethod
    def _compute_data_hash(df: pd.DataFrame, cols: list) -> str:
        """SHA-256 content hash of the feature matrix for cache invalidation.

        This replaces the old row-count-only check — two datasets with identical
        sizes but different distributions now correctly invalidate the cache.
        """
        raw = df[cols].fillna(0).values.tobytes()
        return hashlib.sha256(raw).hexdigest()

    def _has_valid_cache(self, data_hash: str) -> bool:
        """Return True only if ALL cache artefacts exist and the data hash matches."""
        # Atomic bundle check (Step 3): every piece must be present
        required = [self._model_path, self._meta_path]
        if not all(os.path.exists(p) for p in required):
            return False
        try:
            with open(self._meta_path) as f:
                meta = json.load(f)
            # Content-hash check (Step 2)
            if meta.get("data_hash") != data_hash:
                print(f"⚠️  [ML] Cache data hash mismatch — will retrain.")
                return False
            # If supervised, scaler must also exist (Step 3 atomicity)
            if meta.get("is_supervised") and not os.path.exists(self._scaler_path):
                print(f"⚠️  [ML] Scaler missing for supervised model — invalidating cache.")
                return False
            return True
        except Exception:
            return False

    def _load_cached(self) -> bool:
        """Load model artefacts from disk. Returns True on success.

        Atomic bundle invariant (Step 3): if ANY artefact fails to load,
        the entire cache is treated as invalid.
        """
        try:
            self.model = joblib.load(self._model_path)
            with open(self._meta_path) as f:
                self.model_info = json.load(f)
            self.is_supervised = self.model_info.get("is_supervised", False)
            # Scaler is required for supervised models (atomic consistency)
            if self.is_supervised:
                if not os.path.exists(self._scaler_path):
                    raise FileNotFoundError("Scaler missing for supervised model")
                self.scaler = joblib.load(self._scaler_path)
            else:
                self.scaler = None
            mtype = self.model_info.get("model_type", "unknown")
            print(f"✅ [ML] Loaded cached {mtype} from {self._model_path}")
            return True
        except Exception as e:
            print(f"⚠️  [ML] Cache load failed ({e}) — will retrain.")
            self._clear_cache()
            return False

    def _save_cache(self):
        """Persist model artefacts to disk as an atomic bundle."""
        try:
            joblib.dump(self.model, self._model_path)
            if self.scaler is not None:
                joblib.dump(self.scaler, self._scaler_path)
            elif os.path.exists(self._scaler_path):
                # Remove stale scaler from a previous supervised run
                os.remove(self._scaler_path)
            with open(self._meta_path, "w") as f:
                json.dump(self.model_info, f, indent=2)
            print(f"✅ [ML] Model cached → {self._model_path}")
        except Exception as e:
            print(f"⚠️  [ML] Could not cache model: {e}")

    def _clear_cache(self):
        """Remove all cache artefacts to force a clean retrain."""
        for p in [self._model_path, self._scaler_path, self._meta_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # ── Training ────────────────────────────────────────────────────────────

    def fit(self, df_features: pd.DataFrame, labels: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the model.

        Parameters
        ----------
        df_features : DataFrame with FEATURE_COLS columns (customer-level).
        labels      : Optional Series aligned with df_features index where
                      1 = confirmed money laundering, 0 = clean/unknown.

        Returns
        -------
        model_info dict with metrics.
        """
        X = df_features[FEATURE_COLS].fillna(0)
        n_samples = len(X)
        data_hash = self._compute_data_hash(df_features, FEATURE_COLS)
        n_pos = int(labels.sum()) if labels is not None else 0
        report: Dict[str, Any] = {"n_samples": n_samples, "data_hash": data_hash}

        use_supervised = (labels is not None) and (n_pos >= MIN_POSITIVE_SAMPLES)

        self.iso_model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )
        self.iso_model.fit(X)

        if use_supervised:
            if "customer_id" in df_features.columns:
                y = labels.reindex(df_features["customer_id"]).fillna(0).astype(int).values
            else:
                y = labels.reindex(df_features.index).fillna(0).astype(int).values
            n_neg = int((y == 0).sum())
            report["n_positive"] = int(y.sum())
            report["n_negative"] = n_neg

            # Sub-sample if dataset is enormous to stay within memory budget
            if n_samples > MAX_TRAIN_ROWS:
                print(f"⚠️  [ML] Dataset has {n_samples:,} rows — sampling {MAX_TRAIN_ROWS:,} for training.")
                idx = np.random.choice(n_samples, MAX_TRAIN_ROWS, replace=False)
                X_train_all = X.iloc[idx]
                y_train_all = y[idx]
            else:
                X_train_all, y_train_all = X, y

            # Feature scaling
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_train_all)

            # Stratified train / test split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled, y_train_all, test_size=0.2, random_state=42,
                stratify=y_train_all if len(np.unique(y_train_all)) > 1 else None
            )

            scale_pos_weight = n_neg / max(int(y_train_all.sum()), 1)

            if XGBOOST_AVAILABLE:
                self.model = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    scale_pos_weight=scale_pos_weight,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                )
                model_type = "XGBoostClassifier"
            else:
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
                model_type = "RandomForestClassifier"

            self.model.fit(X_tr, y_tr)

            # Evaluate on held-out test set
            y_pred = self.model.predict(X_te)
            y_prob = self.model.predict_proba(X_te)[:, 1]
            clf_rep = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
            auc = float(roc_auc_score(y_te, y_prob)) if len(np.unique(y_te)) > 1 else 0.0

            self.is_supervised = True
            report.update({
                "model_type": model_type,
                "is_supervised": True,
                "scoring_strategy": "Hybrid Composite (70% Supervised + 30% IsolationForest)",
                "auc_roc": round(auc, 4),
                "precision": round(clf_rep.get("1", {}).get("precision", 0.0), 4),
                "recall": round(clf_rep.get("1", {}).get("recall", 0.0), 4),
                "f1_score": round(clf_rep.get("1", {}).get("f1-score", 0.0), 4),
                "feature_importances": self.get_feature_importance(),
            })
            print(
                f"✅ [ML] {model_type} trained | "
                f"AUC-ROC={auc:.4f} | F1={report['f1_score']} | "
                f"Positives={report['n_positive']}/{n_samples}"
            )

        else:
            # ── IsolationForest fallback ─────────────────────────────────
            self.model = self.iso_model
            self.scaler = None
            self.is_supervised = False

            reason = (
                f"is_laundering column missing"
                if labels is None
                else f"Only {n_pos} labeled positives (need ≥ {MIN_POSITIVE_SAMPLES})"
            )
            report.update({
                "model_type": "IsolationForest",
                "is_supervised": False,
                "scoring_strategy": "Pure IsolationForest Anomaly Scoring (100%)",
                "contamination": 0.05,
                "reason": reason,
            })
            print(f"⚠️  [ML] IsolationForest (unsupervised) — {reason}")

        self.model_info = report
        self._save_cache()
        return report

    # ── Inference ───────────────────────────────────────────────────────────

    def score(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Score all rows in df_features.

        Returns
        -------
        np.ndarray of float, shape (n,), values in [0, 100].
        Higher score = higher money-laundering risk.
        """
        if self.model is None:
            return np.full(len(df_features), 50.0)

        X = df_features[FEATURE_COLS].fillna(0)

        # Unsupervised IsolationForest anomaly score component
        iso_model = getattr(self, "iso_model", self.model)
        if iso_model is not None and hasattr(iso_model, "decision_function"):
            raw_iso = iso_model.decision_function(X)
            iso_range = raw_iso.max() - raw_iso.min()
            iso_score = (raw_iso.max() - raw_iso) / iso_range if iso_range > 0 else np.full(len(X), 0.5)
        else:
            iso_score = np.full(len(X), 0.5)

        if self.is_supervised:
            X_in = self.scaler.transform(X) if self.scaler is not None else X.values
            proba = self.model.predict_proba(X_in)[:, 1]
            # Hybrid composite: 70% supervised probability + 30% IsolationForest anomaly score
            composite = 0.70 * proba + 0.30 * iso_score
            return np.round(composite * 100, 1)
        else:
            return np.round(iso_score * 100, 1)

    # ── Utility ─────────────────────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return top-N feature importances (supervised models only)."""
        if not self.is_supervised or not hasattr(self.model, "feature_importances_"):
            return []
        pairs = sorted(
            zip(FEATURE_COLS, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": f, "importance": round(float(imp), 4)} for f, imp in pairs[:top_n]]

    def fit_or_load(
        self,
        df_features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Load cached model if the content-hash matches AND all artefacts are present;
        otherwise train from scratch.
        """
        data_hash = self._compute_data_hash(df_features, FEATURE_COLS)
        if self._has_valid_cache(data_hash):
            if self._load_cached():
                return self.model_info
        return self.fit(df_features, labels)
