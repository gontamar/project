import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import tempfile
import uuid
import os
 
from transformers import TextIteratorStreamer
import threading
 
MODEL_PATH = "models/BioMistral-7B-DARE"
 
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
 
class LLMHandler:
    def __init__(self, model_id="BioMistral/BioMistral-7B-DARE"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                                       local_files_only=True,
                                                       trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)
 
        self.model.eval()
 
    def text_to_speech(self, text):
        tts = gTTS(text=text, lang='en')
        unique_filename = f"response_{uuid.uuid4().hex}.mp3"
        temp_path = os.path.join(tempfile.gettempdir(), unique_filename)
        tts.save(temp_path)
        return temp_path
    
    def generate_response(self, query, imaging_text, blood_text):
        prompt = f"""<s>[INST] <<SYS>>
    You are a Senior Surgical Consultant. Provide a direct clinical conclusion.
    RULES:
    1. Provide short answers. No introductory filler (e.g., "Based on the data").
    2. If asking about a specific test, use only that evidence.
    3. If blood values are abnormal, state the abnormality and the diagnosis directly.
    4. If imaging is >50%, state the diagnosis. If <50% or "No Finding", state the scan is clear.
    5. If combining findings, provide a single unified conclusion.
<<SYS>>
 
[EVIDENCE]
{imaging_text}
{blood_text}
 
QUESTION: {query} [/INST]
Answer:"""
 
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            early_stopping=True
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
 
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
        else:
            answer = full_text.split("[/INST]")[-1].strip()
            
        return answer
    
    def generate_response_stream(self, query, imaging_text, blood_text, history=""):
        prompt = f"""<s>[INST] <<SYS>>
You are a Senior Surgical Consultant. Provide a direct clinical conclusion.
RULES:
1. One sentence only. No introductory filler (e.g., "Based on the data").
2. If asking about a specific test, use only that evidence.
3. If blood values are abnormal, state the abnormality and the diagnosis directly.
4. If imaging is >50%, state the diagnosis. If <50% or "No Finding", state the scan is clear.
5. If combining findings, provide a single unified conclusion.
<</SYS>>
 
[EVIDENCE]
{imaging_text}
{blood_text}
[CHAT HISTORY]
{history}
 
QUESTION: {query} [/INST]
Answer:"""
 
 
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
 
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
 
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            early_stopping=True
        )
 
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs
        )
        thread.start()
 
        for token in streamer:
            yield token
 
