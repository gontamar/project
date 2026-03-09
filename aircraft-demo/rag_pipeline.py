import os
import pdfplumber
import chromadb
import ollama
 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
 
MANUALS_FOLDER = "manuals"
 
 
class RAGEngine:
 
    def __init__(self, db_path="manual_db"):
 
        self.db_path = db_path
 
        self.client = chromadb.PersistentClient(path=self.db_path)
 
        self.vector_db = None
 
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
 
 
    def get_manual_path(self, part_name):
 
        pdf_name = f"{part_name}.pdf"
        pdf_path = os.path.join(MANUALS_FOLDER, pdf_name)
 
        if os.path.exists(pdf_path):
            return pdf_path
 
        return None
 
 
    def load_pdf(self, pdf_path):
 
        text = ""
 
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
 
                if page_text:
                    text += page_text + "\n"
 
        return text
 
 
    def embed_text(self, text):
 
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
 
        return response["embedding"]
 
 
    def store_manual(self, part_name):
 
        pdf_path = self.get_manual_path(part_name)
 
        if pdf_path is None:
            return False
 
        text = self.load_pdf(pdf_path)
 
        chunks = self.text_splitter.split_text(text)
 
        collection = self.client.get_or_create_collection(part_name)
 
        for i, chunk in enumerate(chunks):
 
            embedding = self.embed_text(chunk)
 
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{part_name}_{i}"]
            )
 
        return True
 
 
    def retrieve_context(self, part_name, query, k=3):
 
        collection = self.client.get_collection(part_name)
 
        query_embedding = self.embed_text(query)
 
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
 
        docs = results["documents"][0]
 
        context = "\n\n".join(docs)
 
        return context
 