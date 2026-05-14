"""FastAPI server exposing latency metrics for the frontend."""

from __future__ import annotations

import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from latency import LATENCY_HUB


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/api/latency")
def get_latency() -> dict:
    return {"ok": True, "data": LATENCY_HUB.get_latest()}


@app.get("/api/history")
def get_history() -> dict:
    return {"ok": True, "items": LATENCY_HUB.get_history()}


def start_api(host: str = "127.0.0.1", port: int = 8000) -> Optional[uvicorn.Server]:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server
