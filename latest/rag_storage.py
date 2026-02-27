import os
import pdfplumber
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
 
class RAGEngine:
    def __init__(self, db_path="patient_db"):
        self.db_path = db_path
        # Using the local path provided
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="models/all-MiniLM-L6-v2"
        )
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.vector_db = None
 
    def initialize_session(self, pdf_path=None):
        try:
            self.client.delete_collection("blood_data")
        except:
            pass
        
        documents = []
        if pdf_path and os.path.exists(pdf_path):
            with pdfplumber.open(pdf_path) as pdf:
                full_text = "".join([(page.extract_text() or "") + "\n" for page in pdf.pages])
 
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
            for chunk in splitter.split_text(full_text):
                documents.append(Document(page_content=chunk))
 
        if documents:
            self.vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.db_path,
                collection_name="blood_data",
                client=self.client
            )
            return True
        return False
 
    def query_case(self, question):
        if not self.vector_db:
            return "No blood report data available."
        docs = self.vector_db.similarity_search(question, k=4)
        return "\n".join([d.page_content for d in docs])
 
 
