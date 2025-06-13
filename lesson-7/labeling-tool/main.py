import csv
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Define the paths relative to the current file
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '..', 'data')
static_dir = os.path.join(current_dir, 'static')
traces_file_path = os.path.join(data_dir, 'traces.csv')
labeled_traces_file_path = os.path.join(data_dir, 'labeled_traces_latest.jsonl')

# Mount the static directory to serve frontend files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class Label(BaseModel):
    trace_id: str
    feedback: str
    failure_modes: List[str]

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, 'index.html'))

@app.get("/api/traces")
async def get_traces():
    try:
        with open(traces_file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            traces = [row for row in reader]
        return JSONResponse(content=traces)
    except FileNotFoundError:
        return JSONResponse(content={"error": "traces.csv not found"}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/label")
async def save_label(label: Label):
    try:
        with open(labeled_traces_file_path, mode='a', encoding='utf-8') as jsonlfile:
            jsonlfile.write(json.dumps(label.dict()) + '\n')
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
