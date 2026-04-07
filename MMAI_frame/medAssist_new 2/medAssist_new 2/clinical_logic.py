import json
import re
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
 
from session_memory import memory_manager
 
MODEL_PATH = "models/BioMistral-7B-DARE"
 
# ---------------- MODEL LOAD ---------------- #
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
 
# ---------------- MEMORY ---------------- #
 
def reset_clinical_memory(session_id="default"):
    memory_manager.clear_session(session_id)
    return "Memory cleared"
 
 
def get_history_as_string(session_id="default"):
    session = memory_manager.get_session(session_id)
    messages = session.get_messages()
 
    history_str = ""
    for msg in messages:
        prefix = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
        history_str += f"{prefix}: {msg.content}\n"
 
    return history_str
 
 
def store_interaction(user_input, ai_output, session_id="default"):
    session = memory_manager.get_session(session_id)
    session.add_user(user_input)
    session.add_ai(ai_output)
 
 
# ---------------- STREAMING ---------------- #
 
def get_streamer_response(prompt, session_id="default"):
    messages = [{"role": "user", "content": prompt}]
 
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
 
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
 
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
 
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=400,
        temperature=0.2
    )
 
    Thread(target=model.generate, kwargs=generation_kwargs).start()
 
    return streamer
 
 
# ---------------- ROUTER ---------------- #
 
def route_question(user_input, session_id="default"):
 
    # Extract text safely
    if isinstance(user_input, dict):
        user_question = user_input.get("text", "")
    elif isinstance(user_input, list) and len(user_input) > 0:
        item = user_input[0]
        user_question = item.get("text", "") if isinstance(item, dict) else str(item)
    else:
        user_question = str(user_input)
 
    # Get conversation history
    history = get_history_as_string(session_id)
 
    prompt = f"""
You are a medical query router.
 
Your job is to classify the USER'S LATEST QUESTION into categories.
 
### Conversation History (for context only):
{history}
 
### Current User Question:
{user_question}
 
### Instructions:
 
Return a JSON object containing ONLY relevant keys from:
["brain","chest","blood"]
 
Definitions:
 
- "brain": tumors, neurology, cognition, psychiatry, head anatomy
- "chest": lungs, heart, thoracic issues, X-rays
- "blood": lab reports, blood values, hematology
 
Rules:
1. Use ONLY the current question for classification
2. Use history ONLY if the question is a follow-up
3. Each key must have ONE string
4. Combine multiple questions into one string
5. DO NOT include empty categories
6. RETURN ONLY RAW JSON
 
JSON:
"""
 
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
 
    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_new_tokens=150, do_sample=False)
 
    full_response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(full_response)
 
    try:
        start = full_response.find("{")
        end = full_response.rfind("}") + 1
        return json.loads(full_response[start:end]), user_question
    except Exception as e:
        print(f"Routing Error: {e}")
        return {}, user_question
 
 
# ---------------- BRAIN ---------------- #
 
def get_brain_prompt(question, predictions, session_id="default"):
    history = get_history_as_string(session_id)
 
    tumor_classes = ["Glioma", "Meningioma", "Pituitary"]
    tumor_prob = max(predictions.get(c, 0) for c in tumor_classes)
    most_likely = max(predictions, key=predictions.get)
 
    likelihood = (
        "Extremely high" if tumor_prob > 0.9 else
        "High" if tumor_prob > 0.5 else
        "Moderate" if tumor_prob > 0.2 else
        "Low"
    )
 
    summary = f"""Most likely: {most_likely}
Tumor Likelihood: {likelihood} ({tumor_prob:.4f})
No Tumor: {predictions.get('No Tumor', 0):.4f}"""
 
    return f"""
You are a clinical AI assistant.
 
Conversation History:
{history}
 
BRAIN MRI SUMMARY:
{summary}
 
Current Question:
{question}
 
Answer concisely using ONLY the summary above.
Do not add external assumptions.
"""
 
 
# ---------------- CHEST ---------------- #
 
def get_chest_prompt(question, predictions, session_id="default"):
    history = get_history_as_string(session_id)
 
    formatted = "\n".join(
        [f"- {k}: {v:.2%} probability" for k, v in predictions.items() if k.strip()]
    )
 
    return f"""
You are a clinical AI assistant.
 
Conversation History:
{history}
 
CHEST X-RAY PREDICTIONS:
{formatted}
 
Current Question:
{question}
 
Answer concisely using ONLY the information above.
"""
 
 
# ---------------- BLOOD ---------------- #
 
def interpret_lab(value, low, high):
    try:
        v, l, h = float(value), float(low), float(high)
        return "LOW" if v < l else "HIGH" if v > h else "NORMAL"
    except:
        return "UNKNOWN"
 
 
def get_blood_prompt(question, retrieved_context, session_id="default"):
    history = get_history_as_string(session_id)
 
    pattern = re.compile(
        r"([A-Za-z\s%]+)\s+(\d+\.?\d*)\s*(Low|High|Borderline)?\s*(\d+\.?\d*)-(\d+\.?\d*)",
        re.IGNORECASE
    )
 
    structured = []
 
    for match in pattern.findall(retrieved_context):
        name, val, _, low, high = match
        status = interpret_lab(val, low, high)
        structured.append(f"{name.strip()}: {val} (Ref {low}-{high}) -> {status}")
 
    summary = "\n".join(structured)
 
    return f"""
You are a clinical AI assistant.
 
Conversation History:
{history}
 
Blood Test Data:
{summary}
 
Context:
{retrieved_context}
 
Current Question:
{question}
 
Answer clearly and professionally.
If insufficient data, say clinical correlation is required.
"""
