"""Speech input (speech-to-text) and output (text-to-speech).

Microphone backends: PyAudio if installed, otherwise sounddevice (works on any Python version).
"""
_engine = None


def speak(text: str, mute: bool = False) -> None:
    """Print the reply and speak it aloud (offline, using pyttsx3)."""
    global _engine
    print(f"Assistant: {text}")
    if mute:
        return
    try:
        import pyttsx3

        if _engine is None:
            _engine = pyttsx3.init()
            _engine.setProperty("rate", 170)
        _engine.say(text)
        _engine.runAndWait()
    except Exception:
        pass  # speaking is optional; the text is already printed


def _record_pyaudio(sr, timeout, phrase_limit):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        try:
            return recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        except sr.WaitTimeoutError:
            return None


def _rms(data):
    import numpy as np

    return float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))


def _record_sounddevice(sr, timeout, phrase_limit):
    """Record until the speaker stops (about 1 second of silence), using simple volume detection."""
    import numpy as np
    import sounddevice as sd

    rate, step = 16000, 0.1
    block = int(rate * step)
    frames, started, waited, silent = [], False, 0.0, 0.0
    with sd.InputStream(samplerate=rate, channels=1, dtype="int16", blocksize=block) as stream:
        print("Listening...")
        noise = [_rms(stream.read(block)[0]) for _ in range(5)]      # 0.5 s background-noise calibration
        threshold = max(400.0, 3 * float(np.mean(noise)))
        while True:
            data = stream.read(block)[0]
            loud = _rms(data) > threshold
            if not started:
                if loud:
                    started = True
                    frames.append(data)
                else:
                    waited += step
                    if waited >= timeout:
                        return None
            else:
                frames.append(data)
                silent = 0.0 if loud else silent + step
                if silent >= 1.0 or len(frames) * step >= phrase_limit:
                    break
    return sr.AudioData(np.concatenate(frames).tobytes(), rate, 2)


def listen(timeout: int = 6, phrase_limit: int = 10) -> str:
    """Record from the microphone and return the recognised text ('' if nothing understood)."""
    import speech_recognition as sr

    try:
        import pyaudio  # noqa: F401
        have_pyaudio = True
    except ImportError:
        have_pyaudio = False

    if have_pyaudio:
        audio = _record_pyaudio(sr, timeout, phrase_limit)
    else:
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            raise ImportError("no microphone library found. Run: pip install sounddevice")
        audio = _record_sounddevice(sr, timeout, phrase_limit)
    if audio is None:
        return ""
    try:
        return sr.Recognizer().recognize_google(audio).lower()   # free Google Web Speech API
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        print("(Speech service unreachable - check your internet connection)")
        return ""
