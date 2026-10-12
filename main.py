"""AI Voice Assistant (NLP & Speech Processing).

Voice mode :  python main.py
Text mode  :  python main.py --text        (type instead of speaking, no microphone needed)
Options    :  --mute (no spoken replies)   --debug (show detected intent and confidence)
"""
import argparse

from assistant.nlp import THRESHOLD, IntentClassifier
from assistant.skills import handle
from assistant.speech import listen, speak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true", help="type commands instead of speaking")
    ap.add_argument("--mute", action="store_true", help="do not speak replies aloud")
    ap.add_argument("--debug", action="store_true", help="show intent and confidence")
    args = ap.parse_args()

    clf = IntentClassifier()
    speak("Hello! I am your voice assistant. Say help to see what I can do, or say exit to stop.", args.mute)

    while True:
        if args.text:
            try:
                text = input("You: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
        else:
            try:
                text = listen()
            except (ImportError, OSError) as e:
                print(f"Microphone/speech library problem: {e}\nSee README, or run with --text.")
                break
            if text:
                print(f"You: {text}")
        if not text:
            continue

        intent, conf = clf.predict(text)
        if args.debug:
            print(f"[intent={intent} confidence={conf:.2f}]")
        if conf < THRESHOLD:
            reply, stop = "Sorry, I did not understand that. Say help to see what I can do.", False
        else:
            try:
                reply, stop = handle(intent, text)
            except Exception as e:  # never crash the loop
                reply, stop = f"Sorry, something went wrong ({type(e).__name__}).", False
        speak(reply, args.mute)
        if stop:
            break


if __name__ == "__main__":
    main()
