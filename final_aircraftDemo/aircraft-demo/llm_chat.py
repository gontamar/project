import ollama
 
from rag_pipeline import RAGEngine
from session_memory import memory_manager
 
 
rag = RAGEngine()

 
def ask_llm(part_name, question, history):
 
    session = memory_manager.get_session("manual_chat")
 
    if history is None:
        history = []
 
    # store user message in memory
    session.add_user(question)
 
    if part_name is None:
 
        history.append({"role": "user", "content": question})
        history.append({
            "role": "assistant",
            "content": "Upload a part image first."
        })
 
        return history
 
    context = rag.retrieve_context(part_name, question)
 
    # get past messages from memory
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
 
    stream = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
 
    answer = ""
 
    for chunk in stream:
 
        token = chunk["message"]["content"]
        answer += token
 
        history[-1]["content"] = answer
 
        yield history
 
    # store assistant reply
    session.add_ai(answer)
 