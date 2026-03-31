# audio_feedback.py
import pyttsx3
import threading
import queue

engine = pyttsx3.init()
engine.setProperty("rate", 192)

speech_queue = queue.Queue()
engine_lock = threading.Lock()

current_generation = 0


def next_generation():
    global current_generation
    current_generation += 1

    with engine_lock:
        engine.stop()

    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break

    print(f"[TTS] New generation {current_generation}")
    return current_generation


def enqueue_sentence(text, generation_id):
    print(f"[TTS] enqueue: gen={generation_id} text={text}")
    speech_queue.put((text, generation_id))


def tts_worker():
    print("[TTS] worker started")
    while True:
        text, gen_id = speech_queue.get()

        if gen_id != current_generation:
            print("[TTS] skipped stale sentence")
            continue

        with engine_lock:
            if gen_id != current_generation:
                continue
            print("[TTS] speaking:", text)
            engine.say(text)
            engine.runAndWait()


threading.Thread(target=tts_worker, daemon=True).start()