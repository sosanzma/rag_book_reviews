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
        
        # Send the main answer
        answer_msg = cl.Message(content=response['answer'])
        await answer_msg.send()
        
        # If sources are available, send them as a separate message
        if response['sources']:
            sources_msg = cl.Message(content=f"Sources:\n{response['sources']}")
            await sources_msg.send()
        else:
            no_sources_msg = cl.Message(content="No specific sources found for this information.")
            await no_sources_msg.send()

        await thinking_msg.remove()
    except Exception as e:
        error_msg = cl.Message(content=f"An error occurred: {str(e)}")
        await error_msg.send()
        await thinking_msg.remove()

if __name__ == "__main__":
    cl.run()