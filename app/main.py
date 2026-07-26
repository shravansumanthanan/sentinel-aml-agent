from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, List, Optional
import os
import logging
import traceback as tb

from app.agent import AMLAgentOrchestrator

logger = logging.getLogger("sentinel-aml")

# Resolve paths relative to project root (one level up from app/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

app = FastAPI(title="SENTINEL-AML | Autonomous Compliance Decision System", version="2.0.0")

# CORS — restrict to known origins (Audit Step 6)
# Set CORS_ORIGINS env var as comma-separated list for production deployments
_cors_raw = os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
CORS_ALLOWED_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# ── Lazy-init orchestrator with retry (Audit Step 8) ───────────────────────
orchestrator: Optional[AMLAgentOrchestrator] = None
_startup_error: Optional[str] = None

def _get_orchestrator() -> AMLAgentOrchestrator:
    """Lazy-init with retry — a transient data failure no longer permanently bricks the server."""
    global orchestrator, _startup_error
    if orchestrator is not None:
        return orchestrator
    try:
        orchestrator = AMLAgentOrchestrator(data_dir=DATA_DIR)
        _startup_error = None
        logger.info("[SENTINEL-AML] Orchestrator initialised successfully.")
        return orchestrator
    except Exception as e:
        _startup_error = tb.format_exc()
        logger.error(f"[SENTINEL-AML] Orchestrator init failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Agent engine failed to initialise: {e}. Check server logs.",
        )

# Attempt eager init at import time (preserves current UX for happy path)
try:
    orchestrator = AMLAgentOrchestrator(data_dir=DATA_DIR)
except Exception as e:
    _startup_error = tb.format_exc()
    logger.warning(f"[SENTINEL-AML] Deferred init — startup failed: {e}")

class ChatRequest(BaseModel):
    query: str = Field(..., description="Natural language compliance analyst query")

    @model_validator(mode="before")
    @classmethod
    def accept_message_alias(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "query" not in values and "message" in values:
                values["query"] = values["message"]
        return values

class StressTestRequest(BaseModel):
    lower_bound: float = Field(..., description="Minimum transaction amount threshold lower bound")

class HealthResponse(BaseModel):
    status: str
    version: str

class ChatResponse(BaseModel):
    query: str
    parsed_intent: str
    extracted_entities: Dict[str, Any]
    telemetry: Dict[str, Any]
    results: Dict[str, Any]
    explanations: List[str]
    sar_narrative: Optional[str] = None

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    orch = _get_orchestrator()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        response = orch.process_query(req.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stress-test")
def stress_test_endpoint(req: StressTestRequest):
    orch = _get_orchestrator()
    result = orch.stress_test_threshold(req.lower_bound)
    return result

@app.get("/api/health")
def health_endpoint():
    return {
        "status": "ok" if orchestrator is not None else "degraded",
        "version": app.version,
        "startup_error": _startup_error[:500] if _startup_error else None,
    }

@app.get("/api/dataset/summary")
def summary_endpoint():
    orch = _get_orchestrator()
    eda_res = orch.eda_tool.run(orch.df_transactions, orch.df_customers)
    return eda_res

@app.get("/api/model/info")
def model_info_endpoint():
    """Return active ML model metadata — type, mode (supervised/unsupervised), and metrics."""
    orch = _get_orchestrator()
    return orch.get_model_info()

@app.post("/api/upload")
async def upload_dataset_endpoint(
    transactions_file: UploadFile = File(...),
    customers_file: Optional[UploadFile] = File(None)
):
    """
    Ingests custom CSV transaction & customer datasets.
    Re-runs feature engineering, ML anomaly scoring, and risk classification in-memory.
    """
    tx_name = getattr(transactions_file, "filename", None) or ""
    if not tx_name.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Transactions file must be a .csv file.")
        
    cust_name = getattr(customers_file, "filename", None) if customers_file else None
    if cust_name and not cust_name.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Customers file must be a .csv file.")

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tx_dest = os.path.join(DATA_DIR, "transactions.csv")
        
        # Save transactions CSV
        content = await transactions_file.read()
        with open(tx_dest, "wb") as f:
            f.write(content)

        # Optional customers CSV
        if customers_file and cust_name and cust_name.endswith(".csv"):
            cust_dest = os.path.join(DATA_DIR, "customers.csv")
            cust_content = await customers_file.read()
            with open(cust_dest, "wb") as f:
                f.write(cust_content)

        # Trigger orchestrator reload
        orch = _get_orchestrator()
        orch.load_data()

        eda_res = orch.eda_tool.run(orch.df_transactions, orch.df_customers)
        model_info = orch.get_model_info()

        return {
            "status": "success",
            "filename": transactions_file.filename,
            "total_transactions": eda_res["summary"]["total_transactions"],
            "unique_customers": eda_res["summary"]["unique_customers"],
            "total_volume": eda_res["summary"]["total_volume"],
            "active_model": model_info.get("model_type", "IsolationForest"),
            "is_supervised": model_info.get("is_supervised", False),
            "message": "Custom dataset uploaded and analyzed successfully!"
        }
    except Exception as e:
        logger.error(f"[SENTINEL-AML] CSV upload ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed ingesting CSV dataset: {str(e)}")

# Mount Static Files
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
