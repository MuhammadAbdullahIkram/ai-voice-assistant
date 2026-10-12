# Code Explanation

This document explains how the AI Voice Assistant works, file by file.

## 1. The big picture
Every command, spoken or typed, follows the same path:

```
text  ->  preprocess  ->  intent classifier  ->  (confident?)  ->  slot extraction  ->  skill  ->  reply
```
The terminal (`main.py`) and the web UI (`web_app.py`) share the same NLP and skill code. Only the input/output differs.

---

## 2. `data/intents.json` (training data)
A dictionary: intent name -> example phrases.
```json
{ "time": ["what time is it", "tell me the time", "current time"],
  "calculate": ["calculate 5 plus 3", "what is 12 times 4"] }
```
The classifier learns from these phrases. More and more varied phrases mean better accuracy.

---

## 3. `assistant/nlp.py` (the NLP brain)

### Preprocessing
```python
def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r"\d+(\.\d+)?", " num ", text)   # numbers -> one token
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()
```
Lowercasing and removing punctuation reduces superficial differences. Replacing every number with the token `num` makes
"5 plus 3" and "12 plus 4" look the same to the model, so it learns the *pattern* of a calculation.

### Intent classifier
```python
self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                           LogisticRegression(C=20, max_iter=2000))
```
- **TF-IDF** turns a sentence into numbers. A word gets a high weight if it is frequent in the sentence but rare across
  all training phrases ("calculate" is informative, "the" is not). `ngram_range=(1, 2)` also uses word pairs such as "what time".
- **Logistic Regression** takes those numbers and outputs a probability for every intent.

```python
probs = self.model.predict_proba([preprocess(text)])[0]
i = probs.argmax()
return str(self.model.classes_[i]), float(probs[i])     # (intent, confidence)
```

### Confidence threshold
`THRESHOLD = 0.35`. If the best probability is lower, the assistant says it did not understand instead of doing
something wrong. This matters because the model must always pick *some* intent, even for nonsense input.

### Slot extraction
The intent says *what* to do; slots say *with what*. These functions use simple rules:
| Function | Example input | Result |
|---|---|---|
| `extract_search_query` | "search python tutorials on google" | `python tutorials` |
| `extract_topic` | "who is albert einstein" | `albert einstein` |
| `extract_open_target` | "open the notepad app" | `notepad` |
| `extract_play_query` | "play despacito on youtube" | `despacito` |
| `extract_note` | "take a note buy milk" | `buy milk` |

Example (strip leading filler words, then a trailing "on google"):
```python
words = text.lower().strip().split()
while words and words[0] in _FILLER_START:
    words.pop(0)
```

---

## 4. `assistant/skills.py` (actions)
`handle(intent, text)` is a dispatcher. It runs the right function and returns `(reply, should_exit)`.

```python
if intent == "time":
    return f"It is {dt.datetime.now():%I:%M %p}.", False
if intent == "search_web":
    q = extract_search_query(text)
    webbrowser.open("https://www.google.com/search?q=" + quote_plus(q))
    return f"Here is what I found on Google for {q}.", False
```

### Safe calculator
Spoken words are converted to symbols ("times" -> `*`), numbers and operators are extracted, and the expression is
evaluated with Python's `ast` module, **not** `eval()`:
```python
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
```
Only numbers and `+ - * /` are allowed, so no code can be injected. Division by zero is handled.

### Other skills
- **Open:** known apps (`notepad`, `calculator`...) start through `subprocess`; known sites open in the browser; unknown names become a Google search.
- **Wikipedia:** `wikipedia.summary(topic, sentences=2)`, with handling for ambiguous or missing pages.
- **Notes:** appended with a timestamp to `notes/notes.txt`; "read my notes" returns the last three.

---

## 5. `assistant/speech.py` (speech input/output)
- `speak(text)` prints the reply and speaks it offline with `pyttsx3` (silently skipped if unavailable).
- `listen()` records audio and sends it to the free Google Web Speech API (`recognize_google`).
  - If **PyAudio** is installed, it uses `speech_recognition.Microphone`.
  - Otherwise it uses **sounddevice** with a small voice-activity detector: it measures background noise for 0.5 s, sets a volume
    threshold (3x the noise level), starts recording when you speak, and stops after about 1 second of silence.

---

## 6. `main.py` (terminal app)
```python
intent, conf = clf.predict(text)
if conf < THRESHOLD:
    reply, stop = "Sorry, I did not understand that...", False
else:
    reply, stop = handle(intent, text)
```
`--text` replaces the microphone with `input()`, and `--debug` prints the intent and confidence. The skill call is wrapped
in `try/except`, so one failure never crashes the loop.

---

## 7. `web_app.py` (FastAPI backend)
| Endpoint | Purpose |
|---|---|
| `GET /` | serves `static/index.html` |
| `POST /command` | takes `{"text": ...}`, runs the same pipeline, returns `reply`, `intent`, `confidence`, `understood`, `exit` |
| `POST /listen` | fallback: records with the Python microphone code (for browsers without speech recognition) |

The `IntentClassifier` is trained once when the server starts, so every request is fast.

---

## 8. `static/index.html` (web UI)
One self-contained file (HTML, CSS, JavaScript), no external libraries.
- **Orb states:** `data-state` on the app container is `idle`, `listening`, `thinking` or `speaking`; CSS animations change with it.
- **Speech-to-text:** the browser's `SpeechRecognition` API (Chrome/Edge) with live interim results; falls back to `/listen` otherwise.
- **Text-to-speech:** `speechSynthesis` reads the reply (switchable with "Voice replies").
- **Chat:** messages are added with `textContent` (not `innerHTML`), which prevents HTML injection.

---

## 9. `tests/test_intents.py`
Checks 24 phrases that are *not* in the training data. A case passes only if the intent is correct **and** the confidence is above
the threshold. It also unit-tests the slot extractors and the calculator.

---

## 10. Tutorial: add a new intent ("flip a coin")
1. **Add training phrases** to `data/intents.json`:
   ```json
   "coin": ["flip a coin", "toss a coin", "heads or tails", "flip coin please", "toss a coin for me"]
   ```
2. **Write the skill** in `assistant/skills.py`:
   ```python
   def coin_flip() -> str:
       return f"It is {random.choice(['heads', 'tails'])}!"
   ```
3. **Connect it** inside `handle(...)`:
   ```python
   if intent == "coin":
       return coin_flip(), False
   ```
4. Restart and test: `python main.py --text --debug`, then type `flip a coin`.
5. Add a few unseen phrases to `tests/test_intents.py` and re-run it.

---

## 11. Design decisions
- **ML instead of if/else rules:** the classifier handles phrases it has never seen ("can you tell me the current time").
- **TF-IDF + Logistic Regression:** fast, needs little data, runs on any laptop, and is easy to explain and debug.
- **Confidence threshold:** better to say "I did not understand" than to run the wrong action.
- **Two microphone backends:** PyAudio is hard to install on newer Python versions; sounddevice works everywhere.
- **Shared core:** terminal and web UI reuse the same `assistant` package.

## 12. Limitations and ideas
Small training set; English only; online speech recognition; Windows-only app launching.
Next steps: more training phrases, Vosk offline recognition, wake word, LLM fallback, Urdu support.
