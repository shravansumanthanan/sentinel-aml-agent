import re
from typing import Dict, Any, Optional

class NLPIntentParser:
    """
    Deterministic NLP Intent and Entity Extractor (Zero LLM).
    Uses robust regular expressions, keyword tokenization, and numerical pattern matching.
    """
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        
        intent = "GENERAL_EXPLORATION"
        entities = {
            "customer_id": None,
            "time_window_days": None,
            "max_amount": None,
            "min_amount": None,
            "min_count": None,
            "risk_filter": None,
            "pattern_type": None
        }

        # 1. Extract Customer ID (e.g., CUST-4521, 4521, customer ID 1089)
        cust_match = re.search(r'(?:cust-?|customer\s*(?:id)?\s*#?)\s*([0-9]{1,5})', query_lower)
        if not cust_match:
            cust_match = re.search(r'\b([0-9]{4})\b', query_lower)
        
        if cust_match:
            cust_num = cust_match.group(1)
            entities["customer_id"] = f"CUST-{int(cust_num):04d}" if len(cust_num) < 4 else f"CUST-{cust_num}"

        # 2. Extract Time Window (e.g., "last 30 days", "7d", "past week")
        time_match = re.search(r'(?:last|past|in\s*the)\s*([0-9]+)\s*(?:days?|d)', query_lower)
        if time_match:
            entities["time_window_days"] = int(time_match.group(1))
        elif "week" in query_lower:
            entities["time_window_days"] = 7
        elif "month" in query_lower:
            entities["time_window_days"] = 30

        # 3. Extract Max Amount Threshold (e.g., "under $10,000", "< 10000", "below 9500")
        max_amt_match = re.search(r'(?:under|below|less\s*than|<|\$)\s*\$?([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)', query_lower)
        if max_amt_match:
            val_str = max_amt_match.group(1).replace(',', '')
            entities["max_amount"] = float(val_str)

        # 4. Extract Min Transaction Count (e.g., "10+ transactions", "5 or more", "more than 8")
        min_count_match = re.search(r'([0-9]+)\+\s*(?:transactions|txns|tx|deposits)?', query_lower)
        if not min_count_match:
            min_count_match = re.search(r'(?:more\s*than|>|at\s*least)\s*([0-9]+)', query_lower)
        if min_count_match:
            entities["min_count"] = int(min_count_match.group(1))

        # 5. Extract Risk Filter
        if "high risk" in query_lower or "high-risk" in query_lower:
            entities["risk_filter"] = "HIGH"
        elif "medium risk" in query_lower:
            entities["risk_filter"] = "MEDIUM"

        # 6. Intent Classification Logic
        if entities["customer_id"] and ("suspicious" in query_lower or "is" in query_lower or "check" in query_lower or "lookup" in query_lower):
            intent = "SINGLE_ENTITY_LOOKUP"
        elif "structuring" in query_lower or "smurfing" in query_lower:
            intent = "STRUCTURING_SEARCH"
            entities["pattern_type"] = "STRUCTURING"
        elif entities["min_count"] is not None or (entities["max_amount"] is not None and "which customers" in query_lower):
            intent = "THRESHOLD_AGGREGATION"
        elif "eda" in query_lower or "profile" in query_lower or "summary" in query_lower or "distribution" in query_lower or "baseline" in query_lower:
            intent = "FULL_EDA"
        elif "high risk" in query_lower or "flag" in query_lower or "anomal" in query_lower:
            intent = "HIGH_RISK_FILTER"
        elif entities["customer_id"]:
            intent = "SINGLE_ENTITY_LOOKUP"

        return {
            "query": query,
            "intent": intent,
            "entities": entities
        }
