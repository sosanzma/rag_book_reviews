# RAG Book Reviews Chatbot

This project implements a Retrieval-Augmented Generation (RAG) system to create a chatbot that can answer questions about book reviews and opinions. The chatbot uses reports from Goodreads and Reddit to provide comprehensive information about popular psychology books.

## Project Structure

```
README.md
poetry.lock
pyproject.toml
reports/
    ├── goodreads_report.txt
    └── reddit_report.txt
src/
    └── rag_book_reviews/
        ├── __init__.py
        ├── vector_db.py
        ├── chat_interface.py
        ├── main.py
        ├── populate_db.py
        └── read_reports.py
```

## Components

1. `vector_db.py`: Handles the creation and management of the vector database using DeepLake.
2. `chat_interface.py`: Implements the chatbot interface using LangChain and OpenAI's GPT model.
3. `read_reports.py`: Reads and processes the Goodreads and Reddit reports.
4. `populate_db.py`: Populates the vector database with the processed reports.
5. `main.py`: The entry point of the application, integrating Chainlit for the chat interface.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-book-reviews-chatbot.git
   cd rag-book-reviews-chatbot
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ACTIVELOOP_TOKEN=your_activeloop_token
   ACTIVELOOP_ID=your_activeloop_id
   ```

## Usage

1. Populate the vector database:
   ```
   python src/rag_book_reviews/populate_db.py
   ```

2. Run the chatbot using Chainlit:
   ```
   chainlit run src/rag_book_reviews/main.py
   ```

This will start the Chainlit server and open the chat interface in your default web browser.

## How it works

1. The `populate_db.py` script reads the Goodreads and Reddit reports, processes them, and stores them in a DeepLake vector database.
2. The `chat_interface.py` uses LangChain and OpenAI's GPT model to create a retrieval-based question-answering system.
3. When a user asks a question, the system uses a RetrievalQAWithSourcesChain to retrieve relevant information from the vector database.
4. The retrieved information is used to generate an informed response, and the sources of the information are also returned.
5. The Chainlit interface in `main.py` provides a user-friendly chat experience, allowing users to interact with the chatbot seamlessly and view the sources of the information provided.

## Features

- Retrieval-Augmented Generation for accurate and context-aware responses
- Use of RetrievalQAWithSourcesChain for advanced retrieval and answer generation
- Return of source information for each response, enhancing transparency and credibility
- Integration with DeepLake for efficient vector storage and retrieval
- User-friendly chat interface powered by Chainlit
- Comprehensive book reviews and opinions from Goodreads and Reddit

## Next Steps

1. Expand the database with more book reviews and genres
2. Implement user authentication for personalized experiences
3. Add features like book recommendations based on user preferences
4. Optimize the retrieval process for faster response times
5. Implement caching mechanisms to improve performance
6. Enhance the display of source information in the Chainlit interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
