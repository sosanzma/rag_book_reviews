from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from config import ACTIVELOOP_ID, OPENAI_API_KEY
import os

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

class VectorDB:
    def __init__(self, dataset_name):
        self.embeddings = OpenAIEmbeddings()
        self.dataset_path = f"hub://{ACTIVELOOP_ID}/{dataset_name}"
        self.db = DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def add_reports(self, reports, book_title):
        for report_type, content in reports.items():
            chunks = self.text_splitter.split_text(content)
            metadatas = [{"source": f"{book_title}_{report_type}", "book_title": book_title} for _ in chunks]
            self.db.add_texts(chunks, metadatas)

    def get_retriever(self):
        return self.db.as_retriever()

# Example usage
# vector_db = VectorDB("book_chat_db")
# vector_db.add_reports(book_reports, "The Catcher in the Rye")