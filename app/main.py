from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from app.agent import AMLAgentOrchestrator

# Resolve paths relative to project root (one level up from app/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

app = FastAPI(title="SENTINEL-AML | Autonomous Compliance Decision System", version="2.0.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Orchestrator — wrap so a bad data file doesn't crash the whole app
try:
    orchestrator = AMLAgentOrchestrator(data_dir=DATA_DIR)
except Exception as e:
    import traceback
    traceback.print_exc()
    orchestrator = None
    print(f"[SENTINEL-AML] WARNING: Orchestrator failed to initialise: {e}")

class ChatRequest(BaseModel):
    query: str

class StressTestRequest(BaseModel):
    lower_bound: float

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent engine is not available. Check server logs.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        response = orchestrator.process_query(req.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stress-test")
def stress_test_endpoint(req: StressTestRequest):
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent engine is not available.")
    result = orchestrator.stress_test_threshold(req.lower_bound)
    return result

@app.get("/api/health")
def health_endpoint():
    return {"status": "ok" if orchestrator is not None else "degraded", "version": app.version}

@app.get("/api/dataset/summary")
def summary_endpoint():
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent engine is not available. Check server logs.")
    eda_res = orchestrator.eda_tool.run(orchestrator.df_transactions, orchestrator.df_customers)
    return eda_res

@app.get("/api/model/info")
def model_info_endpoint():
    """Return active ML model metadata — type, mode (supervised/unsupervised), and metrics."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent engine is not available. Check server logs.")
    return orchestrator.get_model_info()

# Mount Static Files
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
