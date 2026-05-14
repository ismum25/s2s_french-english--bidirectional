"""Latency metrics broadcaster for the UI."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol


class LatencyHub:
    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._clients: Set[WebSocketServerProtocol] = set()
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._history: Deque[Dict[str, Any]] = deque(maxlen=50)

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        with self._lock:
            self._clients.add(ws)
        try:
            await ws.send(json.dumps({"type": "hello", "ts": time.time()}))
            async for _ in ws:
                pass
        finally:
            with self._lock:
                self._clients.discard(ws)

    async def _broadcast(self, payload: str) -> None:
        if not self._clients:
            return
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    def start(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        if self._thread is not None:
            return

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            server = websockets.serve(self._handler, host, port)
            self._loop.run_until_complete(server)
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _record(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._latest = payload
            self._history.appendleft(payload)

    def publish(self, stages: Dict[str, float], meta: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "type": "latency",
            "ts": time.time(),
            "stages": stages,
            "total_ms": sum(stages.values()),
        }
        if meta:
            payload["meta"] = meta
        self._record(payload)
        if self._loop is None:
            return
        data = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._broadcast(data), self._loop)

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._latest) if self._latest else None

    def get_history(self) -> list[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._history]


LATENCY_HUB = LatencyHub()
