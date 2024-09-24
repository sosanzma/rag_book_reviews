from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings as OpenAIEmbeddingsNew
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['ACTIVELOOP_TOKEN'] = os.getenv("ACTIVELOOP_TOKEN")
ACTIVELOOP_ID = os.getenv("ACTIVELOOP_ID")

class VectorDB:
    def __init__(self, dataset_name):
        self.embeddings = OpenAIEmbeddingsNew()
        self.dataset_path = f"hub://{ACTIVELOOP_ID}/{dataset_name}"
        self.db = DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def add_reports(self, reports):
        for report_type, content in reports.items():
            chunks = self.text_splitter.split_text(content)
            metadatas = [{"source": f"{report_type}"} for _ in chunks]
            self.db.add_texts(chunks, metadatas)

    def get_retriever(self):
        return self.db.as_retriever()

# Example usage
# vector_db = VectorDB("book_chat_db")
# vector_db.add_reports(book_reports, "The Catcher in the Rye")