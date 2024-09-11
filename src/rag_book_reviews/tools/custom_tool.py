from crewai_tools import BaseTool

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter

class BookRAGTool(BaseTool):
    name: str = "RAG tool"
    description: str = (
        "Allows the agent to search and store for book reviews in a vector database."
    )
    def __init__(self, activeloop_id, dataset_name):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.dataset_path = f"hub://{activeloop_id}/{dataset_name}"
        self.db = DeepLake(dataset_path=self.dataset_path, embedding_function=self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever()
        )

    def add_opinions(self, opinions):
        all_texts, all_metadatas = [], []
        for opinion in opinions:
            chunks = self.text_splitter.split_text(opinion["text"])
            for chunk in chunks:
                all_texts.append(chunk)
                all_metadatas.append({"source": opinion["source"]})
        self.db.add_texts(all_texts, all_metadatas)

    def get_response(self, question):
        response = self.chain({"question": question})
        return response