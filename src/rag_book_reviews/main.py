#!/usr/bin/env python
from datetime import datetime
import os
import sys
from rag_book_reviews.crew import BookReviewCrew

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding necessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information
if len(sys.argv) > 1:
    book_title = sys.argv[1]
else:
    raise ValueError("No book_title provided. Please specify a book_title as a command line argument.")

def run():
    """
    Run the crew.
    """

    crew_instance = BookReviewCrew()
    result = crew_instance.crew().kickoff(inputs={
        'book_title': book_title
    })
    print(result)
    print(type(result))

    # Create a reports directory if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Generate a filename based on the genre and current timestamp
    filename = f"reports/{book_title.replace(' ', '_').lower()}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


    result_str = str(result)

    # Write the result to a text file
    with open(filename, 'w') as f:
        f.write(result_str)

    print(f"Report saved to {filename}")

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "book_title": book_title
    }
    try:
        BookReviewCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        BookReviewCrew().crew().replay(task_id=sys.argv[1], inputs={"genre": book_title})

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "book_title": book_title
    }
    try:
        BookReviewCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")