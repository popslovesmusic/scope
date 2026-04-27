from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import os
import json

from .run_native_platform import run_platform
from .run_state import (
    latest_jsonl, 
    tail_jsonl, 
    find_latest_feedback_trace, 
    load_memory_summary,
    clear_memory
)

app = FastAPI(title="Native Wave-Residue Platform API")

class RunParams(BaseModel):
    frames: int = 100
    nodes: int = 100
    engine_steps: Optional[int] = None
    feedback_enabled: Optional[bool] = None

@app.get("/status")
async def get_status():
    pattern = "logs/feedback_trace_*.jsonl"
    latest = latest_jsonl(pattern)
    if not latest:
        return {"status": "idle", "message": "No runs recorded yet."}
    return {"status": "active", "latest_frame": latest}

@app.get("/trace/feedback")
async def get_feedback_trace(n: int = 20):
    path = find_latest_feedback_trace()
    if not path:
        raise HTTPException(status_code=404, detail="No feedback trace found")
    return tail_jsonl(path, n)

@app.get("/memory")
async def get_memory_summary():
    return load_memory_summary()

@app.post("/run")
async def start_run(params: RunParams):
    # For now, we run synchronously for simplicity
    summary = run_platform(
        num_frames=params.frames,
        num_nodes=params.nodes,
        engine_steps_per_frame=params.engine_steps,
        feedback_enabled=params.feedback_enabled
    )
    return summary

@app.post("/memory/reset")
async def reset_memory():
    success = clear_memory()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset memory")
    return {"status": "success", "message": "Residue memory cleared and backup created"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
