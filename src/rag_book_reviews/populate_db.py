from rag_book_reviews.vector_db import VectorDB
from rag_book_reviews.read_reports import read_reports
import deeplake

def populate_database(overwrite=True):
    print(f"DeepLake version: {deeplake.__version__}")
    
    vector_db = VectorDB("book_chat_db", overwrite=overwrite)
    
    reports = read_reports()
    
    if not reports:
        print("No reports found. Please check the report files.")
        return
    
    
    vector_db.add_reports(reports)
    
    print("Database populated successfully!")

if __name__ == "__main__":
    populate_database()