from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
class BookChatInterface:
    def __init__(self, vector_db):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_db.get_retriever()
        )

    def get_response(self, question):
        response = self.chain.invoke({"question": question})
        return response

# Example usage
# chat_interface = BookChatInterface(vector_db)
# response = chat_interface.get_response("What are the main themes in The Catcher in the Rye?")
# print(response['answer'])
# print(response['sources'])