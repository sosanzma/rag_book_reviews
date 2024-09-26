from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from rag_book_reviews.vector_db import VectorDB
from typing import Dict, List


class BookChatInterface:
    def __init__(self, database_name):
        self.vector_db = VectorDB(database_name)
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.get_retriever()
        )

    def get_response(self, question: str) -> Dict[str, str]:
        response = self.chain.invoke({"question": question})
        print(response)
        return response