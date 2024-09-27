from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings as OpenAIEmbeddingsNew
import os
import re
from typing import Dict, List
from deeplake import delete

from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['ACTIVELOOP_TOKEN'] = os.getenv("ACTIVELOOP_TOKEN")
ACTIVELOOP_ID = os.getenv("ACTIVELOOP_ID")

class VectorDB:
    def __init__(self, dataset_name, overwrite=False):
        self.embeddings = OpenAIEmbeddingsNew()
        self.dataset_path = f"hub://{ACTIVELOOP_ID}/{dataset_name}"
        
        if overwrite:
            self.recreate_db()
        else:
            self.db = DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings, read_only=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.book_metadata = {}  # New dictionary to store book metadata

    def recreate_db(self):
        try:
            delete(self.dataset_path)
            print(f"Remote dataset {self.dataset_path} has been deleted.")
        except Exception as e:
            print(f"Error deleting remote dataset: {str(e)}")
        
        self.db = DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings, read_only=False)
        print(f"New dataset {self.dataset_path} has been created.")

    def preprocess_reports(self, reports: Dict[str, str]):
        for report_type, content in reports.items():
            book_sections = re.split(r'\d+\.\s+\*\*', content)[1:]  # Split by numbered book entries
            for section in book_sections:
                book_title = re.match(r'(.*?)\*\*', section)
                if book_title:
                    title = book_title.group(1).strip()
                    link_match = re.search(r'\*\*Link to Reviews\*\*:\s+\[(.*?)\]\((https?://.*?)\)', section)
                    if link_match:
                        self.book_metadata[title] = {
                            'source': report_type,
                            'link': link_match.group(2),
                            'title': title
                        }

    def add_reports(self, reports: Dict[str, str]):
        self.preprocess_reports(reports)
        for report_type, content in reports.items():
            chunks = self.text_splitter.split_text(content)
            metadatas = []
            for chunk in chunks:
                metadata = {"source": report_type}
                for title, book_data in self.book_metadata.items():
                    if title in chunk:
                        metadata.update(book_data)
                        metadata.update({"source": book_data['link']})
                        print(metadata)
                        break
                metadatas.append(metadata)
            print("METADATOS",metadatas) 
            self.db.add_texts(chunks, metadatas)
        print(f"Added {len(chunks)} chunks to the database.")

    def get_retriever(self):
        return self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1}
        )
    def extract_book_title(self, text: str) -> str:
        match = re.search(r'"([^"]+)"', text)
        return match.group(1) if match else None