import chainlit as cl
from rag_book_reviews.vector_db import VectorDB
from rag_book_reviews.chat_interface import BookChatInterface
from rag_book_reviews.read_reports import read_reports

vector_db = None
chat_interface = None

@cl.on_chat_start
async def start():
    global vector_db, chat_interface
    
    # Initialize VectorDB
    vector_db = VectorDB("book_chat_db")
    
    # Read reports
    reports = read_reports()
    
    if not reports:
        await cl.Message(content="No reports found. Please check the book titles and try again.").send()
        return
    
    # Add reports to VectorDB
    vector_db.add_reports(reports)
    
    # Initialize chat interface
    chat_interface = BookChatInterface(vector_db)
    
    await cl.Message(content="Welcome to the Book Chat System! Ask me anything about the books in our database.").send()

@cl.on_message
async def main(message: cl.Message):
    if chat_interface is None:
        await cl.Message(content="Chat interface is not initialized. Please try restarting the chat.").send()
        return

    # Send a thinking message
    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()

    try:
        # Get response from our chat interface
        response = chat_interface.get_response(message.content)
        
        # Create a new message with the answer
        answer_msg = cl.Message(content=response['answer'])
        await answer_msg.send()
        
        # Remove the thinking message
        await thinking_msg.remove()
        
        # Send the sources as a separate message
        sources = response.get('sources', '')
        if sources:
            await cl.Message(content=f"Sources: {sources}").send()
    except Exception as e:
        # Handle any errors
        error_msg = cl.Message(content=f"An error occurred: {str(e)}")
        await error_msg.send()
        await thinking_msg.remove()

if __name__ == "__main__":
    cl.run()