import whisper
 
# load once at startup
model = whisper.load_model("base")
 
 
def transcribe_audio(audio_path):
 
    if audio_path is None:
        return "", None
 
    result = model.transcribe(audio_path)
    text = result["text"].strip()
 
    return text, None
 