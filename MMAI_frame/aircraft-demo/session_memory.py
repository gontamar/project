from typing import Dict, List
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
 
 
class SessionMemory:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.history = ChatMessageHistory()
        self.token_ids = []   # ✅ NEW
 
    def add_user(self, message: str):
        self.history.add_message(HumanMessage(content=message))
 
    def add_ai(self, message: str):
        self.history.add_message(AIMessage(content=message))
 
    def get_messages(self):
        return self.history.messages
 
    def clear(self):
        self.history.clear()
        self.token_ids = []   
 

    def append_tokens(self, new_tokens):
        self.token_ids.extend(new_tokens)
 
    def get_tokens(self):
        return self.token_ids
 
 
 
class MemoryManager:
    """Handles multiple sessions"""
 
    def __init__(self):
        self.sessions: Dict[str, SessionMemory] = {}
 
    def get_session(self, session_id: str = "default"):
 
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionMemory(session_id)
 
        return self.sessions[session_id]
 
    def clear_session(self, session_id="default"):
 
        if session_id in self.sessions:
            self.sessions[session_id].clear()
 
 
memory_manager = MemoryManager()
 