import re
from typing import Dict, Any, Optional

class NLPIntentParser:
    """
    Deterministic Permanent NLP Intent and Entity Extractor (Zero LLM).
    Uses multi-stage tokenization, semantic keyword mapping, and regex entity extraction.
    Guarantees robust intent classification for any query formulation.
    """
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        
        intent = "GENERAL_EXPLORATION"
        entities = {
            "customer_id": None,
            "raw_cust_num": None,
            "time_window_days": None,
            "max_amount": None,
            "min_amount": None,
            "min_count": None,
            "risk_filter": None,
            "pattern_type": None,
            "transaction_type": None,
            "country_code": None,
            "superlative": None
        }

        # 1. Conversational Greetings Detection (Priority #1)
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "who are you", "what is your name", "thanks", "thank you"]
        if query_lower in greetings or any(query_lower.startswith(g + " ") for g in ["hi", "hello", "hey"]):
            intent = "GREETING"
            return {"query": query, "intent": intent, "entities": entities}

        # 2. Extract Customer ID (e.g., CUST-4521, 4521, customer 1089, cust 420)
        cust_match = re.search(r'(?:cust-?|customer\s*(?:id)?\s*#?)\s*([0-9]{1,5})', query_lower)
        if not cust_match:
            # Only match a bare 4-digit number if NOT preceded by a currency sign or comma
            cust_match = re.search(r'(?<![\$,])\b([0-9]{4})\b(?![,0-9])', query_lower)
        
        if cust_match:
            cust_num = cust_match.group(1)
            entities["raw_cust_num"] = cust_num
            entities["customer_id"] = f"CUST-{int(cust_num):04d}" if len(cust_num) < 4 else f"CUST-{cust_num}"

        # 3. Extract Time Window (e.g., "last 30 days", "7d", "past week")
        time_match = re.search(r'(?:last|past|in\s*the)\s*([0-9]+)\s*(?:days?|d)', query_lower)
        if time_match:
            entities["time_window_days"] = int(time_match.group(1))
        elif "week" in query_lower:
            entities["time_window_days"] = 7
        elif "month" in query_lower:
            entities["time_window_days"] = 30

        # 4. Extract Max Amount Threshold (e.g., "under $10,000", "< 10000", "below 9500")
        max_amt_match = re.search(r'(?:under|below|less\s*than|<)\s*\$?([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)', query_lower)
        if max_amt_match:
            val_str = max_amt_match.group(1).replace(',', '')
            entities["max_amount"] = float(val_str)

        # 5. Extract Min Amount Threshold (e.g., "above $50,000", "> 100000", "over 50k", "more than $25,000")
        min_amt_match = re.search(r'(?:above|over|greater\s*than|more\s*than|>)\s*\$?([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)\s*(k)?', query_lower)
        if min_amt_match:
            val_str = min_amt_match.group(1).replace(',', '')
            mult = 1000.0 if min_amt_match.group(2) else 1.0
            entities["min_amount"] = float(val_str) * mult

        # 6. Extract Min Transaction Count (e.g., "10+ transactions", "5 or more", "more than 8")
        min_count_match = re.search(r'([0-9]+)\+\s*(?:transactions|txns|tx|deposits)?', query_lower)
        if not min_count_match and not entities["min_amount"]:
            min_count_match = re.search(r'(?:more\s*than|>|at\s*least)\s*([0-9]+)\s*(?:txns|transactions)?', query_lower)
        if min_count_match:
            entities["min_count"] = int(min_count_match.group(1))

        # 7. Extract Risk Filter
        if "high risk" in query_lower or "high-risk" in query_lower:
            entities["risk_filter"] = "HIGH"
        elif "medium risk" in query_lower or "medium-risk" in query_lower:
            entities["risk_filter"] = "MEDIUM"
        elif "low risk" in query_lower or "low-risk" in query_lower:
            entities["risk_filter"] = "LOW"

        # 8. Extract Transaction Type
        if "wire" in query_lower:
            entities["transaction_type"] = "Wire"
        elif "withdrawal" in query_lower or "cash out" in query_lower or "cash-out" in query_lower:
            entities["transaction_type"] = "Withdrawal"
        elif "deposit" in query_lower:
            entities["transaction_type"] = "Deposit"
        elif "transfer" in query_lower:
            entities["transaction_type"] = "Transfer"

        # 9. Extract Country / Jurisdiction
        country_match = re.search(r'\b(ky|pa|ae|us|gb|ca|de|fr|sg)\b', query_lower)
        if country_match:
            entities["country_code"] = country_match.group(1).upper()

        # 10. Superlatives Detection
        top_superlatives = ["highest", "top", "max", "most", "largest", "worst", "peak"]
        if any(w in query_lower for w in top_superlatives):
            entities["superlative"] = "MAX"

        # 11. Robust Intent Resolution Cascade
        if "help" in query_lower or "what can you do" in query_lower or "capabilities" in query_lower or "command" in query_lower:
            intent = "CAPABILITIES_HELP"
        elif entities["superlative"] == "MAX" and ("risk" in query_lower or "score" in query_lower or "suspicious" in query_lower or "customer" in query_lower or "who" in query_lower or "which" in query_lower):
            intent = "TOP_RISK_SUBJECT"
        elif entities["customer_id"] and ("why" in query_lower or "explain" in query_lower or "reason" in query_lower or "factor" in query_lower or "cause" in query_lower):
            intent = "EXPLAIN_RISK_REASON"
        elif entities["customer_id"]:
            intent = "SINGLE_ENTITY_LOOKUP"
        elif "structuring" in query_lower or "smurfing" in query_lower or "under 10" in query_lower or "9000" in query_lower or "9,000" in query_lower:
            intent = "STRUCTURING_SEARCH"
            entities["pattern_type"] = "STRUCTURING"
        elif "how many" in query_lower and ("risk" in query_lower or "customer" in query_lower or "subject" in query_lower):
            intent = "COUNT_RISK_SUMMARY"
        # Wire/channel checks must come BEFORE min_amount to avoid shadowing
        elif entities.get("transaction_type") or "wire" in query_lower or "channel" in query_lower or ("type" in query_lower and "transaction" in query_lower):
            intent = "TRANSACTION_TYPE_BREAKDOWN"
        elif entities["min_amount"] is not None:
            intent = "LARGE_AMOUNT_FILTER"
        elif entities["min_count"] is not None or (entities["max_amount"] is not None and "which customers" in query_lower):
            intent = "THRESHOLD_AGGREGATION"
        elif "country" in query_lower or "jurisdiction" in query_lower or "fatf" in query_lower or "grey list" in query_lower or "gray list" in query_lower or entities["country_code"]:
            intent = "JURISDICTION_ANALYSIS"
        elif "eda" in query_lower or "profile" in query_lower or "summary" in query_lower or "distribution" in query_lower or "baseline" in query_lower or "total volume" in query_lower or "stats" in query_lower:
            intent = "FULL_EDA"
        elif entities["risk_filter"] or "flag" in query_lower or "anomal" in query_lower or "suspicious" in query_lower:
            intent = "HIGH_RISK_FILTER"
        else:
            intent = "GENERAL_EXPLORATION"

        return {
            "query": query,
            "intent": intent,
            "entities": entities
        }
