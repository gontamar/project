import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
 
from rag_pipeline import RAGEngine
from session_memory import memory_manager
import os

os.environ['TRANSFORMERS-OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
 

MODEL_PATH = "models/Qwen2.5-7B-Instruct"
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only = True, trust_remote_code = True)
 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",   
    local_files_only = True,
    trust_remote_code = True
)
 
model.eval()
 
 
rag = RAGEngine()
 

def ask_llm(part_name, question, history):
 
    session = memory_manager.get_session("manual_chat")
 
    if history is None:
        history = []
 
    # store user message in memory
    session.add_user(question)
 
    # if no part matched
    if part_name is None:
        history.append({"role": "assistant", "content": "Upload a part image first."})
        yield history
        return
 
    context = rag.retrieve_context(part_name, question)
 

    past_messages = session.get_messages()
 
    conversation = ""
    for m in past_messages:
        role = "User" if m.type == "human" else "Assistant"
        conversation += f"{role}: {m.content}\n"
 
    
    prompt = f"""
You are a technical assistant helping identify and explain engine components.
DO NOT add introductory filler words like: "According to the manual", "Based on the manual context, etc.
 
The part being discussed is: {part_name}
 
Conversation History:
{conversation}
 
Manual Context:
{context}
 
Question:
{question}
 
Answer:
"""
 
   
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
 
    # streamer for real-time output
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
 
    # generation kwargs
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9
    )
 
    # run generation in separate thread (for streaming)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
 
    # -------------------------------
    # STREAM OUTPUT
    # -------------------------------
    answer = ""
 
    for token in streamer:
        answer += token
        history[-1]["content"] = answer
        yield history
 
    # -------------------------------
    # STORE IN MEMORY
    # -------------------------------
    session.add_ai(answer)
 