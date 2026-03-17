import whisper
from gradation_logger import logger
 
model = whisper.load_model("base")
 
 
def transcribe_audio(audio_path):

 
    if audio_path is None:
        return "", None
    
    logger.start_stage("speech_to_text")
 
    result = model.transcribe(audio_path, fp16=False)
    text = result["text"].strip()

    logger.end_stage("speech_to_text")
 
    return text, None
 