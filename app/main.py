from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from app.agent import AMLAgentOrchestrator

app = FastAPI(title="AML Suspicious Activity Detection Agent", version="1.0.0")

# Initialize Orchestrator
orchestrator = AMLAgentOrchestrator(data_dir="/Users/sterlingsuman/Desktop/projectx/data")

class ChatRequest(BaseModel):
    query: str

class StressTestRequest(BaseModel):
    lower_bound: float

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    response = orchestrator.process_query(req.query)
    return response

@app.post("/api/stress-test")
def stress_test_endpoint(req: StressTestRequest):
    result = orchestrator.stress_test_threshold(req.lower_bound)
    return result

@app.get("/api/dataset/summary")
def summary_endpoint():
    eda_res = orchestrator.eda_tool.run(orchestrator.df_transactions, orchestrator.df_customers)
    return eda_res

# Mount Static Files
static_dir = "/Users/sterlingsuman/Desktop/projectx/static"
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_root():
    return FileResponse(os.path.join(static_dir, "index.html"))
