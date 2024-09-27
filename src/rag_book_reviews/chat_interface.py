from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from rag_book_reviews.vector_db import VectorDB
from typing import Dict, List


class BookChatInterface:
    def __init__(self, database_name):
        self.vector_db = VectorDB(database_name)
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        # Define a custom prompt template
        template = """Given the following extracted parts of a long document and a question, create a final answer with references.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        

        QUESTION: {question}
        =========
        {summaries}
        =========
        FINAL ANSWER:"""

        PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])
        
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.get_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def get_response(self, question: str) -> Dict[str, str]:
        response = self.chain({"question": question})
        
        answer = response['answer'].strip()
        
        # Check if the answer indicates lack of knowledge
        if answer.lower().startswith("i don't know") or answer.lower() == "i don't know.":
            return {
                "answer": answer,
                "sources": ""
            }
        
        sources = []
        if 'source_documents' in response:
            for doc in response['source_documents']:
                if 'source' in doc.metadata:
                    sources.append(doc.metadata['source'])
        
        return {
            "answer": answer,
            "sources": "\n".join(set(sources)) if sources else ""
        }