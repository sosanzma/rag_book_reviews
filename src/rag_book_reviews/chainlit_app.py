import chainlit as cl
from rag_book_reviews.vector_db import VectorDB
from rag_book_reviews.chat_interface import BookChatInterface


chat_interface = BookChatInterface("book_chat_db")
@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to the Book Chat System! Ask me anything about the books in our database.").send()

@cl.on_message
async def main(message: cl.Message):
    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()

    try:
        response = chat_interface.get_response(message.content)
        
        answer_msg = cl.Message(content=response)
        await answer_msg.send()
        
        await thinking_msg.remove()
        answer2_msg = cl.Message(content=response['sources'])
        await answer2_msg.send()
    except Exception as e:
        error_msg = cl.Message(content=f"An error occurred: {str(e)}")
        await error_msg.send()
        await thinking_msg.remove()

if __name__ == "__main__":
    cl.run()