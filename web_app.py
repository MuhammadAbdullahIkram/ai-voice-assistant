"""Web interface for the AI Voice Assistant.

Run:  python web_app.py      (opens http://127.0.0.1:8000 in your browser)
"""
import socket
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from assistant.nlp import THRESHOLD, IntentClassifier
from assistant.skills import handle
from assistant.speech import listen

HOST = "127.0.0.1"
BASE = Path(__file__).resolve().parent      # works no matter which folder you run it from
app = FastAPI(title="AI Voice Assistant")
clf = IntentClassifier()


class Command(BaseModel):
    text: str


def process(text: str) -> dict:
    """Same pipeline as main.py: intent classification -> skill -> reply."""
    text = text.strip().lower()
    intent, conf = clf.predict(text)
    stop = False
    if conf < THRESHOLD:
        reply = "Sorry, I did not understand that. Say help to see what I can do."
    else:
        try:
            reply, stop = handle(intent, text)
        except Exception as e:  # never crash the server
            reply = f"Sorry, something went wrong ({type(e).__name__})."
    return {"text": text, "reply": reply, "intent": intent, "confidence": round(conf, 2),
            "understood": conf >= THRESHOLD, "exit": stop}


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.post("/command")
def command(cmd: Command):
    if not cmd.text.strip():
        raise HTTPException(status_code=400, detail="Empty command")
    return process(cmd.text[:500])


@app.post("/listen")
def listen_route():
    """Fallback for browsers without speech recognition: record with the Python microphone code."""
    try:
        return {"text": listen()}
    except (ImportError, OSError) as e:
        raise HTTPException(status_code=503, detail=str(e))


def free_port(start: int = 8000) -> int:
    """First free port from `start` upwards (in case 8000 is already in use)."""
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((HOST, port)) != 0:
                return port
    return start


if __name__ == "__main__":
    port = free_port()
    url = f"http://{HOST}:{port}"
    print(f"\nAI Voice Assistant is running at {url}\nPress Ctrl+C in this window to stop it.\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=HOST, port=port)
