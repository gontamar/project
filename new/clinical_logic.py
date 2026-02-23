import json
import re
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
 
MODEL_PATH = "models/BioMistral-7B-DARE"
 
# Load Model once (Optimized for M4 Max)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
 
# Initialize LangChain Memory
memory = ConversationBufferMemory(return_messages=True)
 
def reset_clinical_memory():
    memory.clear()
    return "Memory cleared"
 
def get_history_as_string():
    messages = memory.chat_memory.messages
    history_str = ""
    for msg in messages:
        prefix = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{prefix}: {msg.content}\n"
    return history_str
 
def get_streamer_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=400, temperature=0.2)
    Thread(target=model.generate, kwargs=generation_kwargs).start()
    return streamer
 
# --- ROUTER ---
def route_question(user_input):
    # --- FIX 1: EXTRACT RAW STRING FROM GRADIO DICT ---
    if isinstance(user_input, dict):
        user_question = user_input.get("text", "")
    elif isinstance(user_input, list) and len(user_input) > 0:
        # Handle cases where it might be a list of dicts
        item = user_input[0]
        user_question = item.get("text", "") if isinstance(item, dict) else str(item)
    else:
        user_question = str(user_input)
 
    # --- FIX 2: MEMORY CHECK (SPEED OPTIMIZATION) ---
    # If the user asks a follow-up, we need the context
    history = memory.load_memory_variables({})["history"]
    
    # YOUR EXACT PROMPT
    prompt = f"""
Please analyze the following user question and segregate it into a JSON object.
 
### Schema Requirements:
 
Return a JSON object containing only the relavant keys from the set: ["brain","chest","blood"]
 
1. "brain": A single string containing all questions regarding tumours(or tumors), neurology, cognition, psychiatry, or head anatomy.
 
2. "chest": A single string containing all questions regarding chest conditions, cardiology, pulmonology, or thoracic issues.
 
3. "blood": A single string containing all questions regarding blood report abnormalities, blood report variables, hematology, lab results, or systemic circulation.
 
Constraints:
 
1. Each key must contain one single string.
 
2. If multiple questions fit a category, combine them into that single string, separated by a space.
 
3. If a category has no matching questions, DO NOT include that key in the json.
 
4. Return ONLY the raw JSON object. No conversational filler or markdown code blocks.
 
User Input: {user_question}
JSON:"""
 
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    
    full_response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(full_response)
    
    # Parsing the JSON from the response
    try:
        start = full_response.find("{")
        end = full_response.rfind("}") + 1
        return json.loads(full_response[start:end]), user_question
    except Exception as e:
        print(f"Routing Error: {e}")
        return {}, user_question
 
# --- BRAIN LOGIC ---
def get_brain_prompt(question, predictions):
    tumor_classes = ["Glioma", "Meningioma", "Pituitary"]
    tumor_prob = max(predictions.get(c, 0) for c in tumor_classes)
    most_likely = max(predictions, key=predictions.get)
    
    likelihood = "Extremely high" if tumor_prob > 0.9 else "High" if tumor_prob > 0.5 else "Moderate" if tumor_prob > 0.2 else "Low"
    summary = f"Most likely: {most_likely}\nTumor Likelihood: {likelihood} ({tumor_prob:.4f})\nNo Tumor: {predictions.get('No Tumor', 0):.4f}"
    
    return f"""You are a clinical AI assistant.
 
You are given structured results from a brain MRI classification model.
The probabilities have already been processed and summarized.\nBRAIN MRI SUMMARY:\n{summary}\nQUESTION:\n{question}\nAnswer concisely using only the summary above.
Do not invent additional information."""
 
# --- CHEST LOGIC ---
def get_chest_prompt(question, predictions):
    formatted = "\n".join([f"- {k}: {v:.2%} probability" for k, v in predictions.items() if k.strip()])
    return f"""You are a clinical AI assistant.
You are given chest X-ray model predictions.
These are probabilistic outputs, not confirmed diagnoses.\nCHEST X-RAY PREDICTIONS:\n{formatted}\nQUESTION:\n{question}\nAnswer concisely and only using the information above."""
 
# --- BLOOD LOGIC ---
def interpret_lab(value, low, high):
    try:
        v, l, h = float(value), float(low), float(high)
        return "LOW" if v < l else "HIGH" if v > h else "NORMAL"
    except: return "UNKNOWN"
 
def get_blood_prompt(question, retrieved_context):
    pattern = re.compile(r"([A-Za-z\s%]+)\s+(\d+\.?\d*)\s*(Low|High|Borderline)?\s*(\d+\.?\d*)-(\d+\.?\d*)", re.IGNORECASE)
    structured = []
    for match in pattern.findall(retrieved_context):
        name, val, _, low, high = match
        status = interpret_lab(val, low, high)
        structured.append(f"{name.strip()}: {val} (Ref {low}-{high}) -> {status}")
    
    summary = "\n".join(structured)
    return f"""
    You are a clinical AI assistant reviewing a patient's blood test report.
    The laboratory values have already been evaluated against their reference ranges.
    Abnormal results are labeled as LOW or HIGH.
    Answer the question clearly and professionally as a clinician would.
    If the report alone is insufficient to determine the answer, state that clinical correlation is required.\nBlood Test Data:\n{summary}\nContext:\n{retrieved_context}\nQuestion:\n{question}"""
