"""
Lightweight mock backend for measuring pure UI overhead.

Serves the same endpoints as the real FastAPI backend but returns
pre-computed answers instantly or with a configurable delay.
Run alongside the Vite UI dev server to capture:
  - UI render time
  - Vite proxy overhead
  - React state-update cycle time

Usage:
  python eval_harness/mock_backend.py [--delay 0]
"""
import asyncio
import json
import time
import uuid
import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Highland Greenz Mock Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

DELAY_MS = 0   # set via CLI

# Loaded at startup
ANSWERS: dict[str, str] = {}
SESSIONS: dict[str, dict] = {}
FAKE_TOKEN = "mock_token_highland_greenz"
TENANT_ID  = str(uuid.uuid4())

# ── Auth ────────────────────────────────────────────────────────────────────

class LoginReq(BaseModel):
    email: str
    password: str

@app.post("/auth/login")
async def login(body: LoginReq):
    return {
        "access_token": FAKE_TOKEN,
        "refresh_token": FAKE_TOKEN,
        "token_type": "bearer",
        "expires_in": 3600,
    }

@app.get("/auth/me")
async def me():
    return {"kind": "user", "tenant_id": TENANT_ID, "user_id": str(uuid.uuid4()),
            "email": "admin@advora.ai", "name": "Admin", "role": "admin",
            "permissions": ["conversations:read", "knowledge:read", "knowledge:write"]}

# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "mock": True}

@app.get("/health/deep")
async def health_deep():
    return {"status": "ok", "mock": True, "db": "ok", "redis": "ok"}

# ── Sessions ────────────────────────────────────────────────────────────────

class SessionStart(BaseModel):
    customer_id: str | None = None
    phone: str | None = None
    channel: str = "chat"
    direction: str = "inbound"
    purpose: str = "marketing"

@app.post("/sessions", status_code=201)
async def start_session(body: SessionStart):
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"status": "active", "channel": body.channel,
                     "direction": body.direction, "turns": 0}
    return {
        "session_id": sid, "call_id": None,
        "channel": body.channel, "direction": body.direction, "status": "active",
        "as_of_date": None, "customer": None, "policy": {"evaluated": False},
        "opening_line": "Thanks for calling Highland Greenz. How can I help you today?",
    }

class TurnReq(BaseModel):
    text: str
    stt_confidence: float | None = None

# Realistic STT/TTS delays per provider (ms)
STT_MS = 280   # cloud STT (e.g. Deepgram / Google Speech)
TTS_MS = 190   # cloud TTS (e.g. ElevenLabs / Google TTS)

@app.post("/sessions/{session_id}/turns")
async def take_turn(session_id: str, body: TurnReq):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    sess = SESSIONS[session_id]
    sess["turns"] += 1
    is_voice = sess.get("channel", "chat") == "phone"

    # STT overhead (voice channels only)
    stt_ms = STT_MS if is_voice else 0
    if stt_ms:
        await asyncio.sleep(stt_ms / 1000)

    # Simulate backend RAG+LLM delay
    rag_delay  = max(0, DELAY_MS // 3)
    llm_delay  = max(0, DELAY_MS - rag_delay)
    if DELAY_MS > 0:
        await asyncio.sleep(DELAY_MS / 1000)

    answer = ANSWERS.get(body.text.strip(), _default_answer(body.text))

    # TTS overhead (voice channels only)
    tts_ms = TTS_MS if is_voice else 0
    if tts_ms:
        await asyncio.sleep(tts_ms / 1000)

    wall_ms = stt_ms + DELAY_MS + tts_ms

    return {
        "session_id": session_id,
        "response_text": answer,
        "response_segments": None,
        "action": "respond",
        "executed_tools": [],
        "state_changed_to": None,
        "escalation": None,
        "end_call": False,
        "trace_id": str(uuid.uuid4()),
        "confidence": 0.95,
        "pending_confirmation": None,
        "as_of_date": None,
        "debug": {
            "timing": {
                "wall_ms":  wall_ms,
                "stt_ms":   stt_ms,
                "rag_ms":   rag_delay,
                "llm_ms":   llm_delay,
                "tts_ms":   tts_ms,
                "channel":  sess.get("channel", "chat"),
                "direction": sess.get("direction", "inbound"),
            }
        },
    }

@app.post("/sessions/{session_id}/end")
async def end_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"session_id": session_id, "status": "completed"}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    return {**SESSIONS[session_id], "session_id": session_id,
            "messages": [], "traces": [], "customer": None}

@app.get("/sessions")
async def list_sessions():
    return {"items": []}

# ── Knowledge (stubs) ───────────────────────────────────────────────────────

@app.get("/knowledge/documents")
async def list_docs():
    return []

@app.post("/knowledge/documents", status_code=201)
async def create_doc(body: dict):
    return {"id": str(uuid.uuid4()), "status": "ready", "chunks_created": 0}

# ── Customers (stubs) ────────────────────────────────────────────────────────

@app.get("/customers")
async def list_customers(search: str = "", page_size: int = 8):
    return {"items": [], "total": 0, "page": 1, "page_size": page_size}

# ── Rules / admin (stubs) ───────────────────────────────────────────────────

@app.get("/rules")
@app.get("/tools")
@app.get("/custom-instructions")
@app.get("/analytics/summary")
@app.get("/conversations")
@app.get("/appointments")
@app.get("/campaigns")
async def stub():
    return {"items": []}

# ── Helper ──────────────────────────────────────────────────────────────────

def _default_answer(text: str) -> str:
    t = text.lower()
    if "3bhk" in t or "3 bhk" in t:
        return "The 3BHK ranges from Rs.1.54 crore to Rs.1.72 crore. Shall I narrow it down by size?"
    if "square foot" in t or "sft" in t or "per sq" in t:
        return "The rate is Rs.11,499 per sft across all configurations."
    if "2bhk" in t:
        return "2BHK is indicative at Rs.11,499/sft — sales will confirm the exact figure."
    if "price" in t or "cost" in t:
        return "Happy to help with pricing! Which configuration — 1BHK, 2BHK, or 3BHK?"
    if "clubhouse" in t:
        return "The clubhouse charge is Rs.3.25 lakh, excluding GST."
    if "floor rise" in t or "floor" in t:
        return "Floor rise is Rs.30 per sft per floor, charged from the 4th floor onwards."
    if "maintenance" in t:
        return "Common area maintenance is Rs.3.50/sft/month, prepaid for 24 months."
    if "parking" in t:
        return "Parking is extra — back-to-back at Rs.5.50 lakh or two singles at Rs.6 lakh."
    return "That's a great question! Let me check and get back to you."


def load_offline_answers(xlsx_path: str, results_csv: str = None) -> None:
    """Pre-load golden answers from a prior offline eval result."""
    try:
        import pandas as pd
        # Load from offline results if available
        results = Path(__file__).parent / "results" / "offline_latency_accuracy_report.xlsx"
        if results.exists():
            df = pd.read_excel(results, sheet_name="Offline_Results")
            for _, row in df.iterrows():
                ANSWERS[str(row.get("utterance", "")).strip()] = str(row.get("response_text", ""))
            print(f"[mock] loaded {len(ANSWERS)} pre-computed answers")
    except Exception as e:
        print(f"[mock] no pre-computed answers loaded: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=int, default=0,
                        help="Simulated processing delay in ms (default: 0)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    DELAY_MS = args.delay
    load_offline_answers(
        r"C:\Users\ADVORA-ALWYN\OneDrive\Desktop\highland_greenz_golden_dataset.xlsx"
    )
    print(f"[mock] backend starting on http://localhost:{args.port} (delay={DELAY_MS}ms)")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
