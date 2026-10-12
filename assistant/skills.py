"""Actions the assistant can perform for each intent."""
import ast
import datetime as dt
import operator
import random
import re
import subprocess
import webbrowser
from pathlib import Path
from urllib.parse import quote_plus

from .nlp import (extract_note, extract_open_target, extract_play_query,
                  extract_search_query, extract_topic)

NOTES = Path(__file__).resolve().parent.parent / "notes" / "notes.txt"

SITES = {"youtube": "youtube.com", "google": "google.com", "github": "github.com",
         "facebook": "facebook.com", "linkedin": "linkedin.com", "kaggle": "kaggle.com",
         "wikipedia": "wikipedia.org", "gmail": "mail.google.com", "twitter": "x.com"}
APPS = {"notepad": "notepad", "calculator": "calc", "paint": "mspaint",
        "command prompt": "cmd", "file explorer": "explorer"}
JOKES = ["Why do programmers prefer dark mode? Because light attracts bugs.",
         "Why did the data scientist break up with the graph? It had too many outliers.",
         "I told my computer I needed a break, and it said: no problem, I'll go to sleep."]


def _greeting():
    h = dt.datetime.now().hour
    part = "morning" if h < 12 else "afternoon" if h < 18 else "evening"
    return f"Good {part}! How can I help you?"


# ---------------------------------------------------------------- safe calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    raise ValueError("unsupported expression")


def calculate(text: str) -> str:
    t = text.lower()
    for word, sym in [("multiplied by", "*"), ("divided by", "/"), ("plus", "+"), ("minus", "-"),
                      ("times", "*"), ("into", "*"), ("over", "/"), (" x ", "*")]:
        t = t.replace(word, f" {sym} ")
    tokens = re.findall(r"\d+\.?\d*|[+\-*/]", t)
    if len(tokens) < 3:
        return "Sorry, I could not find a calculation. Try: calculate 12 times 4."
    try:
        result = _eval(ast.parse(" ".join(tokens), mode="eval"))
    except ZeroDivisionError:
        return "I cannot divide by zero."
    except Exception:
        return "Sorry, I could not calculate that."
    result = int(result) if float(result).is_integer() else round(result, 4)
    return f"The answer is {result}."


# ---------------------------------------------------------------- skills
def open_target(text: str) -> str:
    target = extract_open_target(text)
    if not target:
        return "What should I open?"
    if target in APPS:
        try:
            subprocess.Popen(APPS[target], shell=True)
            return f"Opening {target}."
        except Exception:
            return f"Sorry, I could not open {target} on this system."
    if target in SITES:
        webbrowser.open("https://" + SITES[target])
        return f"Opening {target}."
    webbrowser.open("https://www.google.com/search?q=" + quote_plus(target))
    return f"I did not know {target}, so I searched Google for it."


def wikipedia_answer(text: str) -> str:
    topic = extract_topic(text)
    try:
        import wikipedia
    except ImportError:
        return "The wikipedia package is not installed. Run: pip install wikipedia"
    try:
        return wikipedia.summary(topic, sentences=2, auto_suggest=False)
    except Exception as e:
        name = type(e).__name__
        if name == "DisambiguationError":
            return f"{topic} could mean many things. Please be more specific."
        if name == "PageError":
            return f"I could not find a Wikipedia page for {topic}."
        return "I could not reach Wikipedia. Please check your internet connection."


def joke() -> str:
    try:
        import pyjokes
        return pyjokes.get_joke()
    except Exception:
        return random.choice(JOKES)


def save_note(text: str) -> str:
    note = extract_note(text)
    if not note:
        return "What should I note? Say: take a note, followed by your note."
    NOTES.parent.mkdir(exist_ok=True)
    with open(NOTES, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M}  {note}\n")
    return f"Noted: {note}."


def read_notes() -> str:
    if not NOTES.exists():
        return "You have no notes yet."
    lines = NOTES.read_text(encoding="utf-8").strip().splitlines()[-3:]
    return "Your latest notes: " + ". ".join(l[17:].strip() for l in lines)


ABOUT = ("I am a voice assistant. I can tell the time and date, search Google, answer questions from "
         "Wikipedia, open websites and apps, play YouTube, calculate, tell jokes and take notes.")


def handle(intent: str, text: str):
    """Run the skill for an intent. Returns (reply, should_exit)."""
    if intent == "exit":
        return "Goodbye! Have a great day.", True
    if intent == "greeting":
        return _greeting(), False
    if intent == "how_are_you":
        return "I am doing great, thank you for asking!", False
    if intent == "about":
        return ABOUT, False
    if intent == "thanks":
        return "You are welcome!", False
    if intent == "time":
        return f"It is {dt.datetime.now():%I:%M %p}.", False
    if intent == "date":
        return f"Today is {dt.datetime.now():%A, %d %B %Y}.", False
    if intent == "search_web":
        q = extract_search_query(text)
        webbrowser.open("https://www.google.com/search?q=" + quote_plus(q))
        return f"Here is what I found on Google for {q}.", False
    if intent == "wikipedia":
        return wikipedia_answer(text), False
    if intent == "open":
        return open_target(text), False
    if intent == "play":
        q = extract_play_query(text)
        webbrowser.open("https://www.youtube.com/results?search_query=" + quote_plus(q))
        return f"Playing {q} on YouTube.", False
    if intent == "joke":
        return joke(), False
    if intent == "note":
        return save_note(text), False
    if intent == "read_notes":
        return read_notes(), False
    if intent == "calculate":
        return calculate(text), False
    return "Sorry, I cannot do that yet.", False
