import ollama
 
from rag_pipeline import RAGEngine
 
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, HumanMessage
 
 
rag = RAGEngine()
 
 
 
class State(MessagesState):
    pass
 
 

def call_model(state: State):
 
    messages = state["messages"]
 
    prompt = "\n".join([m.content for m in messages])
 
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
 
    return {"messages": [AIMessage(content=response["message"]["content"])]}
 
 
checkpointer = InMemorySaver()
 
builder = StateGraph(State)
 
builder.add_node("call_model", call_model)
 
builder.add_edge(START, "call_model")
 
graph = builder.compile(checkpointer=checkpointer)
 
config = {"configurable": {"thread_id": "manual_chat"}}
 
 
def ask_llm(part_name, question, history):
 
    if history is None:
        history = []
 
    if part_name is None:
 
        history.append({"role": "user", "content": question})
        history.append({
            "role": "assistant",
            "content": "Upload a part image first."
        })
 
        return history
 
 
    context = rag.retrieve_context(part_name, question)
 
 
    prompt = f"""
You are a technical assistant helping identify and explain engine components.
 
The part being discussed is: {part_name}

DO NOT REPEAT CONVERSATION HISTORY.
 
Use the provided manual excerpts to answer the question accurately.
Do not mention filler introductory words like according to the manual. Do not mention source of information.
 
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
 