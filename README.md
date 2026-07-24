# Sentinel AML

A high-performance, deterministic Anti-Money Laundering (AML) decision and orchestration engine built for compliance analytics.

## Overview

Sentinel AML provides dynamic query parsing, hybrid machine learning anomaly detection, and automated FinCEN Suspicious Activity Report (SAR) narrative generation for institutional transaction monitoring.

Unlike traditional rule engines with high false-positive rates or non-deterministic LLM-based approaches, Sentinel AML operates on a zero-LLM architecture. It maps natural language analyst queries to intent and entity contracts, constructs dynamic Tool Execution Graphs (DAGs), and executes sub-10ms localized anomaly scoring using scikit-learn Isolation Forests combined with deterministic regulatory rule sets.

## Key Capabilities

- **Query-Driven Tool DAG**: Parses analyst intent into structured execution contracts and dynamically invokes only required analytical components.
- **Deterministic Latency & Auditability**: Operates under 10ms execution latency with 100% mathematical trace log auditability.
- **Regulatory Narrative Generation**: Automatically constructs audit-ready FinCEN SAR narratives adhering to 31 C.F.R. § 1010.311.
- **Interactive Sensitivity Analysis**: Provides real-time threshold stress-testing for structuring anomaly bands ($9,000–$9,999).
- **Open-Source Dataset Support**: Ingests and standardizes Kaggle IBM Transactions for AML and PaySim Mobile Money benchmark feeds.

## System Architecture

```
                       +-----------------------------+
                       |    Analyst Query Request    |
                       +--------------+--------------+
                                      |
                                      v
                       +-----------------------------+
                       |  NLP Intent & Entity Parser |
                       +--------------+--------------+
                                      |
                                      v
                       +-----------------------------+
                       |   Agent Tool Orchestrator   |
                       |  (Dynamic Plan Construction) |
                       +--------------+--------------+
                                      |
         +----------------------------+----------------------------+
         |                            |                            |
         v                            v                            v
+------------------+        +------------------+        +------------------+
|     EDA Tool     |        | Feature Eng Tool |        | Lookup Tool      |
+------------------+        +--------+---------+        +------------------+
                                     |
                                     v
                            +------------------+
                            |  Hybrid Anomaly  |
                            |  Detection Tool  |
                            +--------+---------+
                                     |
                                     v
                            +------------------+
                            | Risk Classifier  |
                            +--------+---------+
                                     |
                                     v
                            +------------------+
                            |   SAR Generator  |
                            +------------------+
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shravansumanthanan/sentinel-aml-agent.git
   cd sentinel-aml-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize the transaction and customer datasets:
   ```bash
   python3 data_generator.py
   ```

### Running the Application

Start the FastAPI application server via Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Access the web interface by navigating to `http://localhost:8000`.

## API Reference

### 1. Process Analyst Query
- **Endpoint**: `POST /api/chat`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "query": "Find structuring patterns in the last 30 days"
  }
  ```
- **Response Schema**:
  ```json
  {
    "query": "string",
    "parsed_intent": "STRUCTURING_SEARCH | SINGLE_ENTITY_LOOKUP | THRESHOLD_AGGREGATION | FULL_EDA",
    "extracted_entities": {
      "customer_id": "string | null",
      "time_window_days": "integer | null",
      "max_amount": "float | null",
      "min_count": "integer | null"
    },
    "telemetry": {
      "execution_plan": ["array of step strings"],
      "tools_called": ["array of tool names"],
      "tools_skipped": ["array of tool names"],
      "latency_ms": "float"
    },
    "results": { ... },
    "explanations": ["array of strings"],
    "sar_narrative": "string | null"
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

## Data Schema & Sourcing

The system processes two core relational schemas, compatible with Kaggle IBM AML and PaySim benchmark specifications:

### `customers.csv`
| Column Name | Type | Description |
|-------------|------|-------------|
| `customer_id` | `STRING` | Primary key (e.g., `CUST-4521`) |
| `customer_name` | `STRING` | Account holder legal name |
| `risk_rating` | `STRING` | Subject baseline classification (`Low`, `Medium`, `High`) |
| `account_opened_date` | `DATE` | ISO 8601 creation date |
| `kyc_status` | `STRING` | KYC verification state (`Verified`, `Pending`, `Enhanced`) |
| `occupation` | `STRING` | Industry/occupation category |
| `country` | `STRING` | ISO country code |

### `transactions.csv`
| Column Name | Type | Description |
|-------------|------|-------------|
| `transaction_id` | `STRING` | Primary key (e.g., `TX-10042`) |
| `customer_id` | `STRING` | Foreign key referencing `customers.csv` |
| `timestamp` | `DATETIME` | ISO 8601 transaction timestamp |
| `amount` | `FLOAT` | Transaction amount in USD |
| `transaction_type` | `STRING` | Category (`Deposit`, `Transfer`, `Withdrawal`, `Wire`) |
| `channel` | `STRING` | Originating channel (`Online`, `Branch`, `ATM`, `Mobile`) |
| `destination_account` | `STRING` | Counterparty account identifier |
| `country_code` | `STRING` | Transaction origin/destination jurisdiction code |

## Data Attribution & Sourcing

This system ingests data structures compliant with the following open-source benchmark datasets:
- **IBM Transactions for Anti-Money Laundering (AML)** ([Kaggle Link](https://www.kaggle.com/datasets/ealenahmed/money-laundering-transaction-data))
- **PaySim Synthetic Financial Dataset for Fraud Detection** ([Kaggle Link](https://www.kaggle.com/datasets/ealams/paysim1))

## License

Distributed under the MIT License.
