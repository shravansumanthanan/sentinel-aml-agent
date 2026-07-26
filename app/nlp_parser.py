import re
from typing import Dict, Any, Optional

class NLPIntentParser:
    """
    Intelligent Permanent NLP Intent and Entity Extractor (Zero LLM / High Performance).
    Uses multi-stage tokenization, semantic keyword mapping, and regex entity extraction.
    Guarantees 100% deterministic intent classification and structured parameter parsing.
    """
    
    def __init__(self):
        # Pre-compiled regular expressions for optimal parsing performance
        self.pat_cust_explicit = re.compile(
            r'\bCUST[-_]?([0-9]{1,8})\b|'
            r'\bC[-_]?([0-9]{4,8})\b|'
            r'\b(?:customer|cust|subject|user|account)\s*[-#_]?\s*([0-9]{1,8})\b|'
            r'\b(?:id|#)\s*([0-9]{1,8})\b',
            re.IGNORECASE
        )
        self.pat_cust_bare = re.compile(r'(?<![\$,\d])\b([0-9]{4})\b(?![,0-9])')
        
        self.pat_time_window = re.compile(
            r'(?:last|past|in\s*the)\s*([0-9]+)\s*(days?|d|weeks?|w|months?|m|years?|y)\b',
            re.IGNORECASE
        )
        self.pat_date_range = re.compile(
            r'\b([0-9]{4}-[0-9]{2}-[0-9]{2})\s*(?:to|and|-)\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\b'
        )
        
        self.pat_amt_range = re.compile(
            r'(?:between|from)\s*[\$£€₹]?\s*([0-9,]+)\s*(?:and|to|-)\s*[\$£€₹]?\s*([0-9,]+)',
            re.IGNORECASE
        )
        self.pat_max_amt = re.compile(
            r'(?:under|below|less\s*than|<|underneath)\s*[\$£€₹]?\s*([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)\b',
            re.IGNORECASE
        )
        self.pat_min_amt = re.compile(
            r'(?:above|over|greater\s*than|more\s*than|>|exceeding|at\s*least)\s*[\$£€₹]?\s*([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+)\s*(k)?\b(?!\s*(?:transactions|txns|tx|deposits|withdrawals|items|subjects|customers|users|records|days?|d\b))',
            re.IGNORECASE
        )
        self.pat_gt_amount = re.compile(r'(?:greater\s*than|above|over|exceeding|more\s*than|>|at\s*least)\s*[\$£€₹]?\s*([0-9,]+)', re.IGNORECASE)
        self.pat_lt_amount = re.compile(r'(?:less\s*than|below|under|at\s*most|<)\s*[\$£€₹]?\s*([0-9,]+)', re.IGNORECASE)
        self.pat_min_count_plus = re.compile(
            r'([0-9]+)\+\s*(?:transactions|txns|tx|deposits|withdrawals|items|subjects|customers)?',
            re.IGNORECASE
        )
        self.pat_min_count_comp = re.compile(
            r'(?:more\s*than|>|at\s*least|over|above)\s*([0-9]+)\s*(?:txns|transactions|deposits|withdrawals|items|subjects|customers)',
            re.IGNORECASE
        )
        self.pat_country = re.compile(r'\b(ky|pa|ae|us|gb|ca|de|fr|sg|cayman|panama|uae|cayman islands)\b', re.IGNORECASE)
        self.pat_stress_bound = re.compile(
            r'(?:stress\s*test|sensitivity|threshold)\s*(?:at|with|for)?\s*[\$£€₹]?\s*([0-9,]+)',
            re.IGNORECASE
        )

    def parse_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        
        intent = "GENERAL_EXPLORATION"
        entities = {
            "customer_id": None,
            "raw_cust_num": None,
            "time_window_days": None,
            "start_date": None,
            "end_date": None,
            "max_amount": None,
            "min_amount": None,
            "min_score": None,
            "max_score": None,
            "min_count": None,
            "risk_filter": None,
            "pattern_type": None,
            "transaction_type": None,
            "country_code": None,
            "segment": None,
            "superlative": None,
            "stress_bound": None
        }

        # 1. Conversational Greetings Detection (Priority #1)
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "who are you", "what is your name", "thanks", "thank you"]
        if query_lower in greetings or any(query_lower.startswith(g + " ") for g in ["hi", "hello", "hey"]):
            intent = "GREETING"
            return {"query": query, "intent": intent, "entities": entities}

        # 1b. Extract Risk Score Range Filters
        is_score_mention = any(w in query_lower for w in ["score", "risk score", "composite score", "rating"])
        is_cust_mention = any(w in query_lower for w in ["customer", "customers", "subject", "subjects", "user", "users"])

        if is_score_mention or (is_cust_mention and "$" not in query_lower and "£" not in query_lower and "€" not in query_lower and "₹" not in query_lower and any(w in query_lower for w in ["between", "greater than", "less than", ">", "<", "above", "below"])):
            score_between = re.search(r'(?:between|from)?\s*([0-9]{1,3})\s*(?:and|to|-)\s*([0-9]{1,3})', query_lower)
            if score_between and (is_score_mention or "and" in query_lower or "-" in query_lower):
                try:
                    entities["min_score"] = float(score_between.group(1))
                    entities["max_score"] = float(score_between.group(2))
                except ValueError:
                    pass
            else:
                score_gt = re.search(r'(?:greater\s*than|>|above|exceeds|over)\s*([0-9]{1,3})', query_lower)
                score_lt = re.search(r'(?:less\s*than|<|below|under)\s*([0-9]{1,3})', query_lower)
                if score_gt and is_score_mention:
                    entities["min_score"] = float(score_gt.group(1))
                if score_lt and is_score_mention:
                    entities["max_score"] = float(score_lt.group(1))

        # 2. Extract Transaction Amounts (Min / Max)
        if entities["min_score"] is None and entities["max_score"] is None:
            amt_between = re.search(r'between\s*[\$£€₹]?\s*([0-9,]+)\s*(?:and|to|-)\s*[\$£€₹]?\s*([0-9,]+)', query_lower)
            if amt_between:
                try:
                    entities["min_amount"] = float(amt_between.group(1).replace(',', ''))
                    entities["max_amount"] = float(amt_between.group(2).replace(',', ''))
                except ValueError:
                    pass
            else:
                max_amt_match = self.pat_max_amt.search(query_lower)
                if max_amt_match:
                    val_str = max_amt_match.group(1).replace(',', '')
                    entities["max_amount"] = float(val_str)

                min_amt_match = self.pat_min_amt.search(query_lower)
                if min_amt_match:
                    val_str = min_amt_match.group(1).replace(',', '')
                    mult = 1000.0 if min_amt_match.group(2) else 1.0
                    entities["min_amount"] = float(val_str) * mult

        # 3. Extract Customer ID (Exact Regex Matching)
        cust_match = self.pat_cust_explicit.search(query_lower)
        if not cust_match:
            if entities["min_amount"] is None and entities["max_amount"] is None and entities["min_score"] is None and entities["max_score"] is None:
                cust_match = self.pat_cust_bare.search(query_lower)
        
        if cust_match:
            groups = [g for g in cust_match.groups() if g is not None]
            if groups:
                cust_num = groups[0]
                entities["raw_cust_num"] = cust_num
                entities["customer_id"] = f"CUST-{int(cust_num):04d}" if len(cust_num) < 4 else f"CUST-{cust_num}"

        # 4. Extract Date Range (e.g., "2026-01-01 to 2026-06-01" or "1 January and 31 January")
        date_range_match = self.pat_date_range.search(query_lower)
        if date_range_match:
            entities["start_date"] = date_range_match.group(1)
            entities["end_date"] = date_range_match.group(2)
        else:
            months_map = {
                "jan": "01", "january": "01", "feb": "02", "february": "02",
                "mar": "03", "march": "03", "apr": "04", "april": "04",
                "may": "05", "june": "06", "jun": "06", "july": "07", "jul": "07",
                "aug": "08", "august": "08", "sep": "09", "september": "09",
                "oct": "10", "october": "10", "nov": "11", "november": "11",
                "dec": "12", "december": "12",
            }
            named_date_match = re.search(
                r'\b([0-9]{1,2})\s+([a-z]+)\s*(?:to|and|-)\s*([0-9]{1,2})\s+([a-z]+)\b',
                query_lower
            )
            if named_date_match:
                d1, m1_str, d2, m2_str = named_date_match.groups()
                if m1_str in months_map and m2_str in months_map:
                    entities["start_date"] = f"2026-{months_map[m1_str]}-{int(d1):02d}"
                    entities["end_date"] = f"2026-{months_map[m2_str]}-{int(d2):02d}"

        # 5. Extract Time Window (e.g., "today", "last 24 hours", "last 30 days", "7d", "past 2 weeks", "past 3 months")
        if any(w in query_lower for w in ["today", "24 hours", "24h", "last 24", "yesterday"]):
            entities["time_window_days"] = 1
        else:
            time_match = self.pat_time_window.search(query_lower)
            if time_match:
                val = int(time_match.group(1))
                unit = time_match.group(2).lower()
                if unit.startswith("w"):
                    val *= 7
                elif unit.startswith("m"):
                    val *= 30
                elif unit.startswith("y"):
                    val *= 365
                entities["time_window_days"] = val
            elif "weekend" in query_lower:
                entities["time_window_days"] = 7
            elif "week" in query_lower:
                entities["time_window_days"] = 7
            elif "month" in query_lower:
                entities["time_window_days"] = 30
            elif "quarter" in query_lower or "90 days" in query_lower:
                entities["time_window_days"] = 90
            elif "year" in query_lower:
                entities["time_window_days"] = 365

        # 6. Extract Min Transaction Count
        min_count_match = self.pat_min_count_plus.search(query_lower)
        if not min_count_match:
            min_count_match = self.pat_min_count_comp.search(query_lower)
        if min_count_match:
            entities["min_count"] = int(min_count_match.group(1))

        # 7. Extract Risk Filter Category
        if "high risk" in query_lower or "high-risk" in query_lower or ("suspicious" in query_lower and not entities["customer_id"]):
            entities["risk_filter"] = "HIGH"
        elif "medium risk" in query_lower or "medium-risk" in query_lower:
            entities["risk_filter"] = "MEDIUM"
        elif "low risk" in query_lower or "low-risk" in query_lower or "clean" in query_lower or "safe" in query_lower:
            entities["risk_filter"] = "LOW"

        # 8. Extract AML Pattern Type
        if "structuring" in query_lower or "smurfing" in query_lower or "under 10" in query_lower or "9000" in query_lower or "9,000" in query_lower or "split deposit" in query_lower or "avoid reporting" in query_lower or "below 10" in query_lower or "below £10" in query_lower or "below $10" in query_lower or "below €10" in query_lower or "below ₹10" in query_lower or "repeated deposit" in query_lower or "repeated deposits" in query_lower:
            entities["pattern_type"] = "STRUCTURING"
        elif "rapid cash" in query_lower or "cash out" in query_lower or "cash-out" in query_lower or "velocity" in query_lower or "quick withdrawal" in query_lower:
            entities["pattern_type"] = "RAPID_CASHOUT"
        elif "offshore" in query_lower or "fatf" in query_lower or "grey list" in query_lower or "gray list" in query_lower or "tax haven" in query_lower or "sanctioned" in query_lower or "overseas" in query_lower or "international" in query_lower or "country" in query_lower or "countries" in query_lower or "jurisdiction" in query_lower or "jurisdictions" in query_lower or "cayman" in query_lower or "panama" in query_lower or "uae" in query_lower:
            entities["pattern_type"] = "OFFSHORE_JURISDICTION"

        # 9. Extract Transaction Type
        if "wire" in query_lower:
            entities["transaction_type"] = "Wire"
        elif "withdrawal" in query_lower or "cash out" in query_lower or "cash-out" in query_lower:
            entities["transaction_type"] = "Withdrawal"
        elif "deposit" in query_lower and entities["pattern_type"] != "STRUCTURING":
            entities["transaction_type"] = "Deposit"
        elif "transfer" in query_lower:
            entities["transaction_type"] = "Transfer"

        # 10. Extract Country / Jurisdiction Code
        country_match = self.pat_country.search(query_lower)
        if country_match:
            c_val = country_match.group(1).upper()
            country_map = {"CAYMAN": "KY", "CAYMAN ISLANDS": "KY", "PANAMA": "PA", "UAE": "AE"}
            entities["country_code"] = country_map.get(c_val, c_val)

        # 11. Extract Customer Segment / Occupation
        segments = ["retail", "corporate", "institutional", "import/export", "real estate", "student", "accountant", "physician", "attorney", "consultant"]
        for seg in segments:
            if seg in query_lower:
                entities["segment"] = seg.title()
                break

        # 12. Extract Stress Test Bound
        stress_match = self.pat_stress_bound.search(query_lower)
        if stress_match:
            try:
                entities["stress_bound"] = float(stress_match.group(1).replace(',', ''))
            except ValueError:
                pass

        # 13. Superlatives Detection
        top_superlatives = ["highest", "top", "max", "most", "largest", "biggest", "worst", "peak", "maximum", "riskiest"]
        min_superlatives = ["smallest", "minimum", "lowest", "least", "min", "bottom", "safest"]
        if any(w in query_lower for w in top_superlatives):
            entities["superlative"] = "MAX"
        elif any(w in query_lower for w in min_superlatives):
            entities["superlative"] = "MIN"

        # 14. Intelligent Intent Resolution Cascade
        if "help" in query_lower or "what can you do" in query_lower or "capabilities" in query_lower or "command" in query_lower or "guide" in query_lower:
            intent = "CAPABILITIES_HELP"
        elif any(w in query_lower for w in ["should customer", "should case", "should transaction", "should this", "recommend the next action", "recommend action", "immediate investigation", "escalated", "escalation", "requiring immediate review", "require immediate escalation", "cases require"]):
            intent = "CASE_MANAGEMENT_RECOMMENDATION"
        elif any(w in query_lower for w in ["generate report", "investigation report", "case report", "summary of today", "export all high-risk", "export high risk", "summary report", "aml case report", "summary of today's"]):
            intent = "REPORT_GENERATION"
        elif "sar" in query_lower or "suspicious activity report" in query_lower or "file report" in query_lower:
            intent = "SAR_GENERATION"
        elif any(w in query_lower for w in ["behaviour", "behavior", "changed", "increased dramatically", "differently from historical", "unusual spending", "changed significantly", "historical average", "spending behavior"]):
            intent = "BEHAVIOR_CHANGE_ANALYSIS"
        elif any(w in query_lower for w in ["cash deposit", "cash withdrawal", "frequent cash", "cash movement", "cash activity", "depositing and withdrawing", "large cash", "cash spikes", "cash deposits"]):
            intent = "CASH_ACTIVITY_SEARCH"
        elif entities["pattern_type"] == "OFFSHORE_JURISDICTION" or "country" in query_lower or "countries" in query_lower or "jurisdiction" in query_lower or "jurisdictions" in query_lower or "fatf" in query_lower or "offshore" in query_lower or "overseas" in query_lower or "international" in query_lower or "sanctioned" in query_lower or "cayman" in query_lower or "panama" in query_lower or "uae" in query_lower or entities["country_code"]:
            intent = "JURISDICTION_ANALYSIS"
        elif entities["min_count"] is not None or (entities["max_amount"] is not None and entities["min_amount"] is not None):
            intent = "THRESHOLD_AGGREGATION"
        elif entities["pattern_type"] == "STRUCTURING" or "structuring" in query_lower or "smurfing" in query_lower or "avoid reporting" in query_lower or "split deposit" in query_lower or "similar amount" in query_lower or "repeated deposit" in query_lower or "repeated deposits" in query_lower:
            intent = "STRUCTURING_SEARCH"
        elif ("today" in query_lower or "24 hours" in query_lower or "24h" in query_lower or "immediate review" in query_lower or "alerts generated yesterday" in query_lower or "today's" in query_lower or "todays" in query_lower) and not entities["customer_id"]:
            intent = "DAILY_MONITORING"
        elif "stress test" in query_lower or "sensitivity" in query_lower or entities["stress_bound"] is not None:
            intent = "STRESS_TEST"
        elif entities["superlative"] == "MAX" and ("risk" in query_lower or "score" in query_lower or "suspicious" in query_lower or "who" in query_lower or "which" in query_lower or "subject" in query_lower) and not ("transaction" in query_lower and "size" in query_lower):
            intent = "TOP_RISK_SUBJECT"
        elif entities["superlative"] == "MIN" and ("risk" in query_lower or "score" in query_lower or "suspicious" in query_lower or "who" in query_lower or "which" in query_lower or "subject" in query_lower) and not ("transaction" in query_lower and "size" in query_lower):
            intent = "LOWEST_RISK_SUBJECT"
        elif any(w in query_lower for w in ["why", "explain", "reason", "factor", "evidence", "confidence", "what pattern"]) and ("risk" in query_lower or "flag" in query_lower or "score" in query_lower or "alert" in query_lower or entities["customer_id"]):
            intent = "EXPLAIN_RISK_REASON"
        elif entities["customer_id"]:
            intent = "SINGLE_ENTITY_LOOKUP"
        elif (
            "eda" in query_lower
            or "profile" in query_lower
            or "summary" in query_lower
            or "distribution" in query_lower
            or "baseline" in query_lower
            or "total volume" in query_lower
            or "stats" in query_lower
            or "average" in query_lower
            or "avg" in query_lower
            or "mean" in query_lower
            or "median" in query_lower
            or "overview" in query_lower
            or ("transaction" in query_lower and ("size" in query_lower or "amount" in query_lower or "biggest" in query_lower or "largest" in query_lower or "smallest" in query_lower or "maximum" in query_lower or "minimum" in query_lower))
            or (entities["superlative"] is not None and "transaction" in query_lower and not any(w in query_lower for w in ["risk", "score", "suspicious", "who", "which"]))
        ):
            intent = "FULL_EDA"
        elif entities["pattern_type"] == "RAPID_CASHOUT" or "cash out" in query_lower or "velocity" in query_lower or "rapid cash" in query_lower or "every few minutes" in query_lower:
            intent = "VELOCITY_SEARCH"
        elif "how many" in query_lower and ("risk" in query_lower or "customer" in query_lower or "subject" in query_lower or "count" in query_lower or "population" in query_lower):
            intent = "COUNT_RISK_SUMMARY"
        elif entities.get("transaction_type") or "wire" in query_lower or "channel" in query_lower or ("type" in query_lower and "transaction" in query_lower):
            intent = "TRANSACTION_TYPE_BREAKDOWN"
        elif entities["min_score"] is not None or entities["max_score"] is not None:
            intent = "SCORE_RANGE_FILTER"
        elif entities["min_amount"] is not None or entities["max_amount"] is not None:
            intent = "LARGE_AMOUNT_FILTER"
        elif entities["risk_filter"] or "flag" in query_lower or "anomal" in query_lower or "suspicious" in query_lower:
            intent = "HIGH_RISK_FILTER"
        else:
            intent = "GENERAL_EXPLORATION"

        return {
            "query": query,
            "intent": intent,
            "entities": entities
        }
