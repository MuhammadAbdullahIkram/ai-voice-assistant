"""NLP: text preprocessing, intent classification and slot (entity) extraction."""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

DATA = Path(__file__).resolve().parent.parent / "data" / "intents.json"
THRESHOLD = 0.35


def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\d+(\.\d+)?", " num ", text)   # numbers -> one token
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class IntentClassifier:
    def __init__(self, path=DATA):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        X, y = [], []
        for tag, examples in data.items():
            for ex in examples:
                X.append(preprocess(ex))
                y.append(tag)
        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                                   LogisticRegression(C=20, max_iter=2000))
        self.model.fit(X, y)

    def predict(self, text: str):
        """Return (intent, confidence)."""
        probs = self.model.predict_proba([preprocess(text)])[0]
        i = probs.argmax()
        return str(self.model.classes_[i]), float(probs[i])


# ---------------------------------------------------------------- slot extraction
_FILLER_START = {"search", "for", "google", "on", "the", "web", "look", "up", "find", "online", "please", "about"}


def extract_search_query(text: str) -> str:
    words = text.lower().strip().split()
    while words and words[0] in _FILLER_START:
        words.pop(0)
    q = " ".join(words)
    return re.sub(r"\s+(on google|online|on the web|on the internet)$", "", q).strip()


def extract_topic(text: str) -> str:
    q = text.lower().strip()
    lead = r"^(who is|who was|who are|what is|what are|what was|tell me about|tell me more about|explain|wikipedia|search wikipedia for|define|a|an|the)\s+"
    while True:
        new = re.sub(lead, "", q)
        if new == q:
            return q
        q = new


def extract_open_target(text: str) -> str:
    m = re.match(r"^(open|launch|start|go to|take me to)\s+(the\s+)?(.+?)(\s+(app|website|site|application))?$",
                 text.lower().strip())
    return m.group(3).strip() if m else ""


def extract_play_query(text: str) -> str:
    m = re.match(r"^(play|youtube search|search youtube for)\s*(.*?)(\s+on youtube)?$", text.lower().strip())
    q = m.group(2).strip() if m else ""
    return q if q and q not in {"a song", "some music", "music"} else "lofi music"


def extract_note(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"^(take a note|write a note|note down|remember to|save a note|make a note)\s*(that|to)?\s*", "", t)
    return t.strip()
