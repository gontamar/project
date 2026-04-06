import torch
import time
import os
from threading import Thread
 
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
 
from rag_pipeline import RAGEngine
from session_memory import memory_manager
from gradation_logger import logger
 
from audio_feedback import enqueue_sentence
 
# Offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
 
 
MODEL_PATH = "models/Qwen2.5-7B-Instruct"
 
 
# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)
 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
    trust_remote_code=True
)
 
model.eval()
 
 
rag = RAGEngine()
 
 
chat_token_cache = {}
 
 
def build_system_and_rag(part_name, context):
    """
    Build system + RAG tokens (fresh every query)
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical assistant.\n"
                "Answer in a direct and concise manner.\n"
                "Do NOT use filler phrases.\n"
                "Do NOT say things like 'based on context/manual'.\n"
                "Keep answers short (max 3-4 lines unless necessary).\n"
                "Be precise and technical."
            )
        },
        {
            "role": "system",
            "content": f"Part: {part_name}"
        },
        {
            "role": "system",
            "content": f"Manual Context:\n{context}"
        }
    ]
 
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    print("This is the text after applying chat template", text)
 
    tokens = tokenizer(text, return_tensors="pt").to(model.device)["input_ids"]
    print("These are the token ids for the above chat template", tokens)
    return tokens
 
 
def build_user_tokens(question):
    """
    Tokenize ONLY user message
    """
    messages = [
        {"role": "user", "content": question}
    ]
 
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print("This is the text after applying chat template", text)
 
    tokens = tokenizer(text, return_tensors="pt").to(model.device)["input_ids"]
    print("These are the token ids for the above chat template", tokens)
    return tokens
 
 
def ask_llm(part_name, question, history, generation_id):
    session = memory_manager.get_session("manual_chat")
 
    if history is None:
        history = []
 

    session.add_user(question)
 

    if part_name is None:
        history.append({
            "role": "assistant",
            "content": "Upload a part image first."
        })
        yield history
        return
 
    session_id = "manual_chat"
 

    logger.start_stage("rag_retrieval")
    context = rag.retrieve_context(part_name, question)
    logger.end_stage("rag_retrieval")
 

    logger.start_stage("tokenization")
    start_tok = time.perf_counter()
 
    system_tokens = build_system_and_rag(part_name, context)
 
    history_tokens = chat_token_cache.get(session_id, None)
 
    user_tokens = build_user_tokens(question)
 
    if history_tokens is None:
        final_input_ids = torch.cat([system_tokens, user_tokens], dim=1)
    else:
        final_input_ids = torch.cat(
            [system_tokens, history_tokens, user_tokens],
            dim=1
        )
 
    inputs = {"input_ids": final_input_ids}
 
    end_tok = time.perf_counter()
    logger.end_stage("tokenization")
 
    input_tokens = final_input_ids.shape[1]
    logger.log(f"Input Tokens: {input_tokens}")
 
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
 
    generation_kwargs = dict(
        input_ids=final_input_ids,
        streamer=streamer,
        max_new_tokens=150,
        temperature=0.3,
        top_p=0.8,
        repetition_penalty=1.1
    )
 
    def generate():
        logger.start_stage("generation")
        output = model.generate(**generation_kwargs)
        logger.end_stage("generation")
 
        generated_tokens = output[:, final_input_ids.shape[1]:]
 

        assistant_text = tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True
        )
 
        assistant_messages = [
            {"role": "assistant", "content": assistant_text}
        ]
 
        assistant_text_formatted = tokenizer.apply_chat_template(
            assistant_messages,
            tokenize=False,
            add_generation_prompt=False
        )
 
        assistant_tokens = tokenizer(
            assistant_text_formatted,
            return_tensors="pt"
        ).to(model.device)["input_ids"]
 
        if session_id not in chat_token_cache:
            chat_token_cache[session_id] = torch.cat(
                [user_tokens, assistant_tokens], dim=1
            )
        else:
            chat_token_cache[session_id] = torch.cat(
                [chat_token_cache[session_id], user_tokens, assistant_tokens],
                dim=1
            )
 
    thread = Thread(target=generate)
    thread.start()
 

    # answer = ""
 
    # history.append({"role": "assistant", "content": ""})
 
    # for token in streamer:
    #     answer += token
    #     history[-1]["content"] = answer.strip()
    #     yield history

    import re

    answer = ""
    tts_buffer = ""
    spoken_anything = False

    history.append({"role": "assistant", "content": ""})
    sentence_pattern = re.compile(r"[.!?]")

    for token in streamer:
        answer += token
        tts_buffer += token

        history[-1]["content"] = answer.strip()
        yield history

        if sentence_pattern.search(tts_buffer):
            sentence = tts_buffer.strip()

            if len(sentence.split()) >= 3:
                enqueue_sentence(sentence, generation_id)
                spoken_anything = True

            tts_buffer = ""

    final_text = answer.strip()

    # ✅ If nothing was spoken yet, speak the final answer anyway
    if final_text and not spoken_anything:
        enqueue_sentence(final_text, generation_id)



 
    answer = answer.strip()
 
    if len(answer.split("\n")) > 6:
        answer = "\n".join(answer.split("\n")[:6])
 

    output_tokens = len(tokenizer.encode(answer))
    logger.log(f"Output Tokens: {output_tokens}")
 

    session.add_ai(answer)
 
    logger.log_chat_memory(session)
 