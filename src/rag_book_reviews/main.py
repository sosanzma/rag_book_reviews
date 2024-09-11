from rag_book_reviews.vector_db import VectorDB
from rag_book_reviews.chat_interface import BookChatInterface
from rag_book_reviews.read_reports import read_reports

def main():
    print("Welcome to the Book Chat System!")
    
    # Initialize VectorDB
    vector_db = VectorDB("book_chat_db")
    
    # Get book title from user
    book_title = input("Enter the title of the book you want to chat about: ")
    
    # Read reports
    reports = read_reports(book_title)
    
    if not reports:
        print(f"No reports found for '{book_title}'. Please check the book title and try again.")
        return
    
    # Add reports to VectorDB
    vector_db.add_reports(reports, book_title)
    
    # Initialize chat interface
    chat_interface = BookChatInterface(vector_db)
    
    print(f"\nChat about {book_title}. Type 'exit' to end the chat.")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break
        
        response = chat_interface.get_response(question)
        print("\nResponse:")
        print(response["answer"])
        print("\nSources:")
        for source in response["sources"].split(", "):
            print("- " + source)

if __name__ == "__main__":
    main()