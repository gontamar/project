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
        self.embeddings = SentenceTransformerEmbeddings(model_name="models/all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.vector_db = None
 
    def initialize_session(self, pdf_path=None):
        try: self.client.delete_collection("blood_data")
        except: pass
        
        documents = []
        if pdf_path and os.path.exists(pdf_path):
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += (page.extract_text() or "") + "\n"
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            full_text += " | ".join([str(c) for c in row if c]) + "\n"
            
            if full_text.strip():
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                for chunk in splitter.split_text(full_text):
                    documents.append(Document(page_content=chunk))
 
        if documents:
            self.vector_db = Chroma.from_documents(
                documents=documents, embedding=self.embeddings,
                persist_directory=self.db_path, collection_name="blood_data", client=self.client
            )
 
    def query_case(self, question):
 
        if not self.vector_db: return "No blood report data available."
        docs = self.vector_db.similarity_search(question, k=3)
        return "\n[EXTRACTED BLOOD REPORT DATA]:\n" + "\n".join([d.page_content for d in docs])
    
    def get_major_readings(self):
        """Fetches the most relevant blood report segments for summary display."""
        if not self.vector_db:
            return "No blood report uploaded."
        
        docs = self.vector_db.similarity_search("blood test results reference range levels", k=5)
        summary_text = "\n".join([d.page_content for d in docs])
        return f"### 🧪 Major Blood Report Readings\n{summary_text}"
    
 
    def get_blood_table(self, pdf_path):
        if not pdf_path or not os.path.exists(pdf_path):
            return "No blood report data available."
 
        markdown_table = "| Reading Name | Result | Standard Range |\n| :--- | :--- | :--- |\n"
        rows_found = False
        junk_filter = ["low", "high", "reading name", "result", "standard range"]
 
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
 
                lines = {}
                for word in words:
                    top = round(word['top'], 1)
                    if top not in lines:
                        lines[top] = []
                    lines[top].append(word)
 
                for top in sorted(lines.keys()):
                    line_words = sorted(lines[top], key=lambda x: x['x0'])
                    
                    current_row = []
                    if line_words:
                        temp_text = line_words[0]['text']
                        for i in range(1, len(line_words)):
                            if line_words[i]['x0'] - line_words[i-1]['x1'] > 15:
                                current_row.append(temp_text)
                                temp_text = line_words[i]['text']
                            else:
                                temp_text += " " + line_words[i]['text']
                        current_row.append(temp_text)
 
                    if len(current_row) >= 2:
                        name = current_row[0].strip()
                        
                        if name.lower() in junk_filter or len(name) < 2:
                            continue
                        
                        result = current_row[1].strip()
                        ref_range = current_row[2].strip() if len(current_row) > 2 else "N/A"
 
                        if any(char.isdigit() for char in result):
                            markdown_table += f"| {name} | {result} | {ref_range} |\n"
                            rows_found = True
 
        return markdown_table if rows_found else "⚠️ **Extraction failed.** Try a different PDF format."
 
    def _simple_fallback_extractor(self, pdf_path):
        """If the complex pattern fails, just grab lines with numbers."""
        markdown_table = "| Reading Name | Value/Details |\n| :--- | :--- |\n"
        with pdfplumber.open(pdf_path) as pdf:
            full_text = pdf.pages[0].extract_text()
            for line in full_text.split('\n'):
                if any(char.isdigit() for char in line) and len(line) > 10:
                    parts = line.split()
                    name = " ".join([p for p in parts if p.isalpha()])
                    data = " ".join([p for p in parts if not p.isalpha()])
                    markdown_table += f"| {name} | {data} |\n"
        return markdown_table
