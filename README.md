# Sentinel AML | Autonomous Anti-Money Laundering Decision & Orchestration Engine

> **AI-Powered Suspicious Activity Detection: High-performance, deterministic Anti-Money Laundering (AML) decision engine, hybrid ML anomaly scorer, and automated FinCEN Suspicious Activity Report (SAR) narrative generator.**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Supervised%20ML-orange.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Isolation%20Forest-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Problem Statement](#problem-statement)
  - [Business Summary](#business-summary)
  - [Objective](#objective)
- [Minimum Functional Requirements & Adaptive Agent Behavior](#minimum-functional-requirements--adaptive-agent-behavior)
  - [Expected Adaptive Query Behavior](#expected-adaptive-query-behavior)
  - [Core Functional Capabilities](#core-functional-capabilities)
- [Expected Agent Architecture](#expected-agent-architecture)
  - [System Architecture Flowchart](#system-architecture-flowchart)
  - [Tool Registry & Component Breakdown](#tool-registry--component-breakdown)
  - [Recommended Agent Output Structure](#recommended-agent-output-structure)
- [Data Sources & Dataset Information](#data-sources--dataset-information)
  - [Data Sources Used](#data-sources-used)
  - [Relational Data Schema](#relational-data-schema)
  - [Embedded Money Laundering Typologies](#embedded-money-laundering-typologies)
- [Technology Stack](#technology-stack)
- [Installation & Local Setup](#installation--local-setup)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Installation](#step-by-step-installation)
  - [Data Generation & Ingestion](#data-generation--ingestion)
  - [Running the Application](#running-the-application)
- [Usage Guide & Analyst Workflows](#usage-guide--analyst-workflows)
  - [Interactive Web Workstation](#interactive-web-workstation)
  - [Example Natural Language Queries](#example-natural-language-queries)
  - [Structuring Threshold Sensitivity Analysis](#structuring-threshold-sensitivity-analysis)
  - [Exporting SAR Narratives & Audit Reports](#exporting-sar-narratives--audit-reports)
- [API Reference](#api-reference)
- [Project Directory Structure](#project-directory-structure)
- [Testing & Quality Assurance](#testing--quality-assurance)

---

## Problem Statement

### Business Summary

Financial institutions globally are mandated by regulatory bodies (FinCEN, FATF, local authorities) to implement robust Anti-Money Laundering (AML) compliance programs. However, traditional rule-based systems generate excessive false positives, overwhelming compliance teams and increasing operational costs. Meanwhile, sophisticated money laundering techniques—including structuring, smurfing, and layering—evade conventional detection methods.

The challenge is to build an intelligent, autonomous agent that can learn from transaction patterns, identify suspicious behaviours, and provide explainable risk assessments with actionable escalation recommendations. Such an agent reduces false positives, improves detection accuracy, and enables compliance teams to focus on genuine threats rather than manual rule tuning.

### Objective

Sentinel AML is an AI-powered agent designed and implemented to:

1. **Perform Automated Exploratory Data Analysis (EDA)** on transaction and customer data to understand baseline behavior.
2. **Detect Anomalous Transaction Patterns** indicative of money laundering (e.g., structuring/smurfing, rapid cash-out velocity spikes, offshore jurisdiction funneling).
3. **Apply Anomaly Detection** using a hybrid ML and rule-based approach (Supervised XGBoost / RandomForest + Unsupervised Isolation Forest fallback).
4. **Generate Risk Scores & Flags** per transaction/customer (`Low`, `Medium`, `High`, `Critical`).
5. **Provide Explainable Risk Assessments** explaining why a transaction or subject is flagged as suspicious.
6. **Recommend Basic Escalation Actions** (`Monitor`, `Flag for Review / Review`, `Report / SAR Filing`).

---

## Minimum Functional Requirements & Adaptive Agent Behavior

### Expected Adaptive Query Behavior

The agent does **not** follow a fixed, rigid sequential pipeline. Instead, it parses the user's natural language query, extracts intent, filters, entities, and pattern types, and dynamically constructs an execution plan — invoking only the tools necessary to answer that specific query:

| User Query Example | Expected Adaptive Agent Behavior | Tools Invoked | Tools Skipped |
| :--- | :--- | :--- | :--- |
| **"Find structuring patterns in the last 30 days"** | Applies time filter first ($30\text{d}$); invokes structuring-focused feature engineering, anomaly scoring, risk classification, and SAR narrative generator; skips full EDA. | `AMLFeatureEngTool`, `HybridAnomalyTool`, `RiskClassifierTool`, `SARGeneratorTool` | `EDATool`, `SingleEntityLookupTool`, `ThresholdStressTestTool` |
| **"Which customers made 10+ transactions under $10,000?"** | Runs aggregation and threshold rules directly; ML anomaly detection is not required. | `AMLFeatureEngTool`, `RiskClassifierTool` | `EDATool`, `HybridAnomalyTool`, `SingleEntityLookupTool` |
| **"Is customer ID 4521 suspicious?"** | Performs single-entity lookup; explains existing flags or computes risk on-demand for that specific customer only. | `SingleEntityLookupTool`, `RiskClassifierTool`, `SARGeneratorTool` | `EDATool`, `ThresholdStressTestTool` |

### Core Functional Capabilities

The agent is equipped with the following 10 capabilities, invoked selectively based on query intent:

1. **Intent & Entity Extraction**: Extracts intent, filters (date range, segment, country, transaction type), and target AML pattern from natural language queries.
2. **Dynamic Execution Planning**: Builds an adaptive execution plan deciding which tools to call, in what order, and on which subset of data. Not every query needs every tool.
3. **Query-Scoped Preprocessing**: Loads the dataset and applies only the preprocessing relevant to the specific query.
4. **Selective EDA**: Runs EDA selectively when broad exploration is requested; skips it for targeted or single-entity queries.
5. **On-Demand Feature Creation**: Creates AML features on demand, such as transaction frequency, rolling sums, amount deviation, velocity, structuring ratios, and rapid cash-out patterns.
6. **Hybrid Anomaly & Pattern Detection**: Runs suspicious-pattern detection using ML (XGBoost / Isolation Forest), statistical, or rule-based methods on filtered data.
7. **Context-Aware Risk Classification**: Classifies results as `Low`, `Medium`, `High`, or `Critical` risk using context-appropriate thresholds.
8. **Explainable Flag Generation**: Generates human-readable explanations for each flag, directly tied to the query intent and detected AML pattern.
9. **Actionable Escalation Recommendations**: Recommends the exact next action: `Monitor`, `Review` (Flag for Review), or `Report` (SAR Filing).
10. **Structured & Auditable Output**: Returns results in a transparent, structured format including execution telemetry, tools called, tools skipped, latency, telemetry logs, and rationale for reviewers.

---

## Expected Agent Architecture

### System Architecture Flowchart

```
                          +-----------------------------------+
                          |      Analyst Query Request        |
                          |  "Find structuring in last 30d"   |
                          +-----------------+-----------------+
                                            |
                                            v
                          +-----------------------------------+
                          |     NLP Intent & Entity Parser    |
                          |  (Multi-Stage Tokenizer & Regex)  |
                          +-----------------+-----------------+
                                            |
                                            v
                          +-----------------------------------+
                          |      Agent Tool Orchestrator      |
                          |   (Constructs Dynamic Tool DAG)   |
                          +-----------------+-----------------+
                                            |
        +-----------------------------------+-----------------------------------+
        |                                   |                                   |
        v                                   v                                   v
+---------------+                   +---------------+                   +---------------+
| 1. EDA Tool   |                   | 2. Feature Eng|                   | Single Entity |
| (Profiling)   |                   |    Tool       |                   | Lookup Tool   |
+---------------+                   +-------+-------+                   +---------------+
                                            |
                                            v
                                    +---------------+
                                    |  3. Anomaly   |
                                    | Detection Tool|
                                    +-------+-------+
                                            |
                                            v
                                    +---------------+
                                    |   4. Risk     |
                                    | Classification|
                                    +-------+-------+
                                            |
                                            v
                                    +---------------+
                                    | 5. Explanation|
                                    |  Component /  |
                                    | SAR Generator |
                                    +---------------+
```

### Tool Registry & Component Breakdown

1. **EDA Tool (`EDATool`)**: Performs exploratory data analysis, dataset profiling, transaction volume/count aggregation, risk distribution metrics, and channel breakdown visualizations.
2. **Feature Engineering Tool (`AMLFeatureEngTool`)**: Creates model-ready or rule-ready AML features including transaction frequency, amount deviation, structuring ratios, rapid cash-out velocity, counterparty entropy, and high-risk jurisdiction volume.
3. **Anomaly Detection Tool (`HybridAnomalyTool` & `SupervisedAMLClassifier`)**: Scores suspicious transactions or customers using a dual-mode ML approach (Supervised XGBoost/RandomForest trained on historical labels + Unsupervised Isolation Forest fallback).
4. **Risk Classification Tool (`RiskClassifierTool`)**: Converts scores/signals into actionable risk categories (`Low`, `Medium`, `High`, `Critical`) based on model outputs and compliance business logic.
5. **Explanation Component / Rule Layer (`SARGeneratorTool`)**: Generates concise, audit-ready natural language reasons for flags and constructs official FinCEN SAR narratives adhering to **31 C.F.R. § 1010.311**.

### Recommended Agent Output Structure

Every query response generated by Sentinel AML returns a structured, query-aware payload containing:

1. **Query-Aware Execution Summary**: Displays original user request, extracted filters/entities, tools invoked, tools skipped, and latency in milliseconds (`latency_ms`).
2. **Top Suspicious Entities / Transactions**: Subject IDs, customer profiles, and itemized transaction tables returned by the selected analysis path.
3. **Risk Level Assignment**: Clear risk rating (`Low`, `Medium`, `High`, `Critical`) per flagged item.
4. **Contextual Explanation**: Concise natural language justification tied to the query intent and detected AML typology.
5. **Suggested Escalation Action**: Explicit recommendation (`Monitor` / `Flag for Review` / `Report SAR`).
6. **Supporting Charts & Visual Metrics**: Interactive risk breakdown, threshold stress-test curves (Chart.js), and downloadable PDF SAR reports.

---

## Data Sources & Dataset Information

### Data Sources Used

Sentinel AML natively supports three data ingestion channels:

1. **Synthetic 5-Year Multi-Wave Ledger Generator (`data_generator.py`)**:
   - Built-in script generating realistic customer demographics and transaction ledgers spanning 5 years (**2021 to 2026**).
   - Simulates realistic normal banking behavior alongside embedded money laundering typologies and ground-truth `is_laundering` binary labels (`0 = Legitimate`, `1 = Suspicious`).
2. **IBM Transactions for Anti-Money Laundering (AML)**:
   - Benchmark dataset from Kaggle (`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`).
   - Automated ingestion and column normalization provided by `download_kaggle.py` and `app/kaggle_loader.py`.
3. **PaySim Mobile Money Synthetic Financial Dataset**:
   - Synthetic mobile money dataset derived from real financial logs (`ealams/paysim1`).
   - Mapped into standardized relational customer and transaction schema contracts.

### Relational Data Schema

The system processes two core relational data structures:

#### `customers.csv`
| Column Name | Type | Description |
|-------------|------|-------------|
| `customer_id` | `STRING` | Primary key identifier (e.g., `CUST-1042`) |
| `customer_name` | `STRING` | Account holder legal name or corporate entity name |
| `risk_rating` | `STRING` | Subject baseline KYC risk classification (`Low`, `Medium`, `High`) |
| `account_opened_date` | `DATE` | ISO 8601 account creation date (`YYYY-MM-DD`) |
| `kyc_status` | `STRING` | Verification state (`Verified`, `Pending`, `Enhanced`) |
| `occupation` | `STRING` | Customer industry, profession, or corporate classification |
| `country` | `STRING` | ISO 2-letter country code of residence/incorporation |

#### `transactions.csv`
| Column Name | Type | Description |
|-------------|------|-------------|
| `transaction_id` | `STRING` | Primary key identifier (e.g., `TX-58012`) |
| `customer_id` | `STRING` | Foreign key referencing `customers.csv` |
| `timestamp` | `DATETIME` | ISO 8601 transaction timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `amount` | `FLOAT` | Transaction amount in USD ($\$$) |
| `transaction_type` | `STRING` | Category (`Deposit`, `Transfer`, `Withdrawal`, `Wire`) |
| `channel` | `STRING` | Originating channel (`Online`, `Branch`, `ATM`, `Mobile`) |
| `destination_account` | `STRING` | Counterparty receiving account identifier |
| `country_code` | `STRING` | Originating or destination jurisdiction code |
| `is_laundering` | `INTEGER` | Optional ground-truth target label (`0` or `1`) |

### Embedded Money Laundering Typologies

Sentinel AML synthetic data generation embeds three primary institutional laundering patterns:

1. **Structuring / Smurfing**:
   - Multiple cash deposits placed specifically between **$\$9,000$ and $\$9,999$** within short temporal windows (e.g., 30 days) across different branch locations and ATMs to deliberately evade the Currency Transaction Report (CTR) filing threshold of $\$10,000$.
2. **Rapid Cash-Out Velocity Spikes**:
   - High-value incoming wires immediately followed by multi-part outgoing transfers or cash withdrawals within 48 hours, leaving near-zero residual balance.
3. **Offshore FATF High-Risk Jurisdiction Funneling**:
   - Repeated wire transfers directed to non-cooperative offshore financial hubs (Cayman Islands `KY`, Panama `PA`, United Arab Emirates `AE`).

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Core Language** | Python 3.9+ | Primary runtime environment |
| **Web Framework** | FastAPI | High-performance asynchronous API framework |
| **ASGI Server** | Uvicorn | Production-ready HTTP server interface |
| **Data Validation** | Pydantic v2 | Strict schema validation & type enforcement |
| **Machine Learning** | XGBoost & Scikit-Learn | Supervised risk scoring (XGBoost) & unsupervised Isolation Forests |
| **Data Processing** | Pandas & NumPy | High-speed vector aggregation & matrix computations |
| **Model Persistence**| Joblib | Fast disk serialization for trained ML models & scalers |
| **Dataset Ingestion** | KaggleHub | Programmatic download of open-source benchmark datasets |
| **Frontend UI** | HTML5, CSS3, ES6+ JS | Custom Glassmorphism dashboard (Zero external UI frameworks) |
| **Data Visualization**| Chart.js | Interactive chart rendering for risk distributions & thresholds |
| **PDF Reporting** | jsPDF | Client-side export of audit-ready FinCEN SAR PDFs |
| **Testing** | Pytest | Unit and integration test suite |

---

## Installation & Local Setup

### Prerequisites

Ensure the following tools are installed on your local machine:
- **Python**: Version `3.9` or higher
- **pip**: Python package manager
- **git**: Version control system

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shravansumanthanan/sentinel-aml-agent.git
   cd sentinel-aml-agent
   ```

2. **Create and Activate a Virtual Environment**:
   - On macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - On Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Data Generation & Ingestion

Sentinel AML requires transaction and customer CSV files in the `data/` directory.

#### Option A: Generate Synthetic 5-Year Dataset (Recommended for instant setup)
Run the synthetic data generator to construct a 50,000 transaction dataset spanning 2021–2026:
```bash
python3 data_generator.py
```
*Outputs: `data/customers.csv` and `data/transactions.csv`.*

#### Option B: Download Kaggle IBM AML Benchmark Dataset
If you wish to benchmark against official IBM AML Kaggle data:
```bash
python3 download_kaggle.py
```

### Running the Application

Launch the FastAPI application server using Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Once running, access the web workstation in your browser at:
**`http://localhost:8000`**

---

## Usage Guide & Analyst Workflows

### Interactive Web Workstation

The web workstation provides a real-time compliance operations center featuring:
- **Natural Language Chat Console**: Issue plain English queries to inspect subjects or search for laundering patterns.
- **Interactive Threshold Stress-Tester**: Dynamically adjust structuring thresholds ($8,000–$9,900) to measure false positive vs. true positive detection bands.
- **SAR Narrative Generator & PDF Exporter**: View audit-ready SAR text and export official PDF compliance documentation with one click.
- **Telemetry & Traceability Modal**: Inspect exact execution DAG steps, tools invoked, tools skipped, and latency in milliseconds.

### Example Natural Language Queries

Below is a comprehensive collection of copy-pasteable sample queries covering all **12 core AML compliance categories**. Analysts and evaluator teams can copy and paste any of these directly into the web workstation chat console:

#### 1. Daily Monitoring & Alerts
```text
Show me all high-risk customers from today.
What suspicious activity was detected in the last 24 hours?
Summarise today's AML alerts.
Show today's highest-risk transactions.
Were any new suspicious customers identified today?
Show customers requiring immediate review.
```

#### 2. Customer Investigation & Subject Lookup
```text
Is customer C4521 suspicious?
Show me the complete profile for customer C4521.
Why was customer C4521 flagged?
What is customer C4521's current risk score?
Has customer C4521's behaviour changed recently?
Show all transactions made by customer C4521.
Has customer C4521 been investigated before?
Does customer C4521 have multiple suspicious transactions?
Show the transaction history for customer C4521 over the last 90 days.
Is customer CUST-0042 suspicious?
```

#### 3. Structuring & Smurfing Detection
```text
Find customers making repeated deposits below £10,000.
Show possible structuring cases.
Find customers splitting deposits to avoid reporting thresholds.
Which customers made more than 10 deposits below £10,000 this month?
Find structuring patterns in the last 30 days.
```

#### 4. Transaction Investigation & Threshold Filtering
```text
Show transactions above £50,000.
Find unusually large cash deposits.
Show transactions greater than $100000.
Show customers with transactions between £10,000 and £50,000.
```

#### 5. Time-Based & Range Analysis
```text
Analyse transactions from the last 7 days.
Show suspicious activity this month.
Analyse transactions between 1 January and 31 January.
Show high risk transactions in the last 90 days.
```

#### 6. Geographic & High-Risk Jurisdiction Analysis
```text
Show transactions involving high-risk countries.
Find customers sending money overseas.
Show transfers involving Cayman Islands, Panama, or UAE.
Find high risk country transfers to KY and PA.
```

#### 7. Cash Activity & Velocity Spikes
```text
Show unusually large cash deposits.
Find customers making frequent cash withdrawals.
Identify rapid cash deposits followed by withdrawals.
Show ATM and branch cash activity spikes.
```

#### 8. Behaviour Analysis & Volume Shift
```text
Which customers changed their transaction behaviour significantly?
Show customers whose transaction volume increased dramatically.
Find customers with sudden spikes in transaction velocity.
```

#### 9. Rule-Based & Numeric Range Queries
```text
Find customers with more than 20 transactions today.
Show customers whose daily transaction total exceeds £100,000.
Show customers with risk score greater than 50 and less than 70.
Find customers with composite score > 80.
```

#### 10. Risk Assessment & Explainability
```text
Why is customer C4521 considered high risk?
What factors contributed to customer C4521's risk score?
Explain why customer CUST-0042 was flagged as suspicious.
```

#### 11. Case Management & Actionable Escalation
```text
Should customer C4521 be reviewed?
Should customer C4521's case be escalated?
Recommend the next action for customer C4521.
Which cases require immediate escalation?
```

#### 12. FinCEN/FATF Reporting & SAR Generation
```text
Generate an investigation report for customer C4521.
Create a summary of today's suspicious activity.
Generate a FinCEN SAR narrative for customer C4521.
```

### Structuring Threshold Sensitivity Analysis

Navigate to the **Scenario Stress Test** tab or issue a query like `"Stress test at 8500"`. The engine computes:
- Subject count affected under the current boundary vs. lower boundary.
- Total transaction volume captured within the structuring window ($8,500 – $9,999).
- Differential volume and false-positive sensitivity curve rendered via Chart.js.

### Exporting SAR Narratives & Audit Reports

1. Run a search query or inspect a high-risk subject (e.g., `CUST-1042`).
2. Click **"Generate SAR Narrative"**.
3. Review the 5-part audit report on-screen.
4. Click **"Export SAR PDF"** to save an official compliance record locally.
5. Click **"Download Audit CSV"** to export structured telemetry and results.

---

## API Reference

### 1. Process Analyst Query
- **Endpoint**: `POST /api/chat`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "query": "Find structuring patterns in the last 30 days"
  }
  ```
- **Response Schema**:
  ```json
  {
    "query": "Find structuring patterns in the last 30 days",
    "parsed_intent": "STRUCTURING_SEARCH",
    "extracted_entities": {
      "customer_id": null,
      "time_window_days": 30,
      "min_amount": 9000.0,
      "max_amount": 9999.0,
      "min_count": 3,
      "country": null
    },
    "telemetry": {
      "execution_plan": [
        "Parse Query",
        "Compute Feature Baseline",
        "Filter Structuring Candidates",
        "Score Anomalies via XGBoost",
        "Generate SAR Narrative"
      ],
      "tools_called": [
        "AMLFeatureEngTool",
        "HybridAnomalyTool",
        "RiskClassifierTool",
        "SARGeneratorTool"
      ],
      "tools_skipped": [
        "EDATool",
        "SingleEntityLookupTool",
        "ThresholdStressTestTool"
      ],
      "latency_ms": 6.42
    },
    "results": {
      "anomalies_found": 14,
      "high_risk_subjects": ["CUST-1042", "CUST-3819"]
    },
    "explanations": [
      "Detected 14 subjects exhibiting multi-wave deposits between $9,000 and $9,999."
    ],
    "sar_narrative": "SUSPICIOUS ACTIVITY REPORT (SAR) NARRATIVE..."
  }
  ```

### 2. Scenario Stress Test
- **Endpoint**: `POST /api/stress-test`
- **Request Body**:
  ```json
  {
    "lower_bound": 8500.0
  }
  ```

### 3. Dataset Metrics Summary
- **Endpoint**: `GET /api/dataset/summary`
- **Response**: Returns dataset row counts, transaction volume, risk rating breakdown, and channel distribution.

### 4. Engine Health Check
- **Endpoint**: `GET /api/health`
- **Response**: `{"status": "ok", "version": "2.0.0", "startup_error": null}`

---

## Project Directory Structure

```
sentinel-aml-agent/
├── app/
│   ├── __init__.py
│   ├── agent.py            # Main Agent Orchestrator & Tool DAG Constructor
│   ├── kaggle_loader.py    # Merges & standardizes IBM AML & PaySim datasets
│   ├── main.py             # FastAPI REST endpoints & CORS configuration
│   ├── ml_model.py         # Supervised XGBoost & Isolation Forest risk scorer
│   ├── nlp_parser.py       # Zero-LLM regex & semantic intent extractor
│   └── tools.py            # Specialized AML tools (EDA, FeatureEng, SAR, etc.)
├── data/                   # Data storage directory
│   ├── customers.csv       # Standardized customer demographic records
│   ├── transactions.csv    # Standardized transaction ledger
│   └── model_cache/        # Cached joblib models & scalers
├── static/                 # Web workstation static assets
│   ├── index.html          # Compliance workstation single-page app
│   ├── style.css           # Glassmorphism design system & styles
│   └── app.js              # UI controller, Chart.js & jsPDF logic
├── tests/                  # Pytest automated test suite
│   ├── __init__.py
│   ├── test_agent.py       # Tests for orchestrator execution plans
│   ├── test_api.py         # FastAPI endpoint integration tests
│   └── test_nlp.py         # Regex & intent parser accuracy tests
├── Dockerfile              # Container deployment file
├── README.md               # Master technical documentation
├── data_generator.py       # 5-year synthetic AML ledger generator
├── download_kaggle.py      # KaggleHub dataset downloader utility
├── pyproject.toml          # Project configuration & dependencies
└── requirements.txt        # Python dependency manifest
```

---

## Testing & Quality Assurance

Sentinel AML includes a full test suite built with `pytest` / `unittest` covering NLP intent parsing, tool execution, ML scoring, and FastAPI endpoints.

Run tests locally:
```bash
python3 -m unittest discover tests
```
