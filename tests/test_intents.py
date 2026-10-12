"""Checks the intent classifier on phrases that are NOT in the training data.
Run from the project folder:  python tests/test_intents.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant.nlp import (THRESHOLD, IntentClassifier, extract_note, extract_open_target,
                           extract_play_query, extract_search_query, extract_topic)
from assistant.skills import calculate

CASES = [
    ("hello there", "greeting"), ("good morning assistant", "greeting"),
    ("what time is it right now", "time"), ("can you tell me the current time", "time"),
    ("what is the date today", "date"), ("tell me today's date", "date"),
    ("search for python courses on google", "search_web"), ("google best phones", "search_web"),
    ("who is cristiano ronaldo", "wikipedia"), ("tell me about the moon", "wikipedia"),
    ("open whatsapp", "open"), ("launch notepad", "open"),
    ("play shape of you", "play"), ("play some naat on youtube", "play"),
    ("tell me something funny", "joke"), ("make me laugh please", "joke"),
    ("take a note call mom tomorrow", "note"), ("read my notes please", "read_notes"),
    ("calculate 45 plus 55", "calculate"), ("what is 9 times 7", "calculate"),
    ("thanks for the help", "thanks"), ("how are you doing today", "how_are_you"),
    ("goodbye assistant", "exit"), ("what can you do", "about"),
]

clf = IntentClassifier()
correct = 0
for text, expected in CASES:
    got, conf = clf.predict(text)
    ok = got == expected and conf >= THRESHOLD  # must also be confident enough to act
    correct += ok
    print(f"{'PASS' if ok else 'FAIL'}  {text!r:45} -> {got} ({conf:.2f})")
print(f"\nIntent accuracy on unseen phrases: {correct}/{len(CASES)} = {correct / len(CASES):.0%}")

assert extract_search_query("search python tutorials on google") == "python tutorials"
assert extract_topic("who is albert einstein") == "albert einstein"
assert extract_open_target("open the notepad app") == "notepad"
assert extract_play_query("play despacito on youtube") == "despacito"
assert extract_note("take a note buy milk") == "buy milk"
assert calculate("what is 12 times 4") == "The answer is 48."
assert calculate("calculate 10 divided by 4") == "The answer is 2.5."
assert calculate("5 divided by 0") == "I cannot divide by zero."
print("Slot extraction and calculator tests passed.")
