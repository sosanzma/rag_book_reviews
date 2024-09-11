import os
import yaml
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from crewai_tools import SerperDevTool
from langchain.tools import tool, StructuredTool
import chainlit as cl
import json
from pydantic import BaseModel, Field

global_user_query = None

class AskHumanInput(BaseModel):
    question: str = Field(..., description="The question to ask the human")

async def ask_human(question: str) -> str:
    global global_user_query
    if global_user_query:
        response = global_user_query
        global_user_query = None  # Reset after use
        return response
    else:
        human_response = await cl.AskUserMessage(content=f"{question}").send()
        if human_response:
            return human_response["content"]
        return "No response received from the user."

ask_human_tool = StructuredTool(
    name="Ask Human follow up questions",
    description="Ask human follow up questions",
    func=ask_human,
    args_schema=AskHumanInput
)

class CrewOutputEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return {key: str(value) for key, value in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        return str(obj)

@CrewBase
class BookReviewCrew:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.agents_config = self.load_config('agents.yaml')
        self.tasks_config = self.load_config('tasks.yaml')
        self.book_info = None

    def load_config(self, filename):
        config_path = os.path.join(self.base_path, 'config', filename)
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def get_llm(self, model_name):
        if model_name.startswith('gpt'):
            return ChatOpenAI(model_name=model_name)
        elif model_name.startswith('claude'):
            return ChatAnthropic(model=model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @agent
    def scraper_agent(self) -> Agent:
        config = self.agents_config['scraper_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=[SerperDevTool()],
            verbose=True,
            llm=self.get_llm('gpt-4o-mini'),
            allow_delegation=False
        )

    @agent
    def qa_agent(self) -> Agent:
        config = self.agents_config['qa_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            verbose=True,
            llm=self.get_llm('gpt-4o-mini'),
            allow_delegation=True,
            tools=[ask_human_tool],
        )

    @task
    def scrape_book_info(self) -> Task:
        config = self.tasks_config['scrape_book_info']
        return Task(
            description=config['description'],
            agent=self.scraper_agent(),
            expected_output=config['expected_output']
        )

    @task
    def interactive_qa(self) -> Task:
        config = self.tasks_config['interactive_qa']
        return Task(
            description=config['description'],
            agent=self.qa_agent(),
            expected_output=config['expected_output']
        )

    @crew
    def initial_crew(self) -> Crew:
        return Crew(
            agents=[self.scraper_agent()],
            tasks=[self.scrape_book_info()],
            process=Process.sequential,
            verbose=True
        )

    @crew
    def qa_crew(self) -> Crew:
        return Crew(
            agents=[self.qa_agent()],
            tasks=[self.interactive_qa()],
            process=Process.sequential,
            verbose=True
        )

    def initial_run(self, book_title):
        self.book_info = self.initial_crew().kickoff(inputs={'book_title': book_title})
        return self.book_info

    def qa_run(self, user_query):
        global global_user_query
        global_user_query = user_query  # Set the global variable
        
        result = self.qa_crew().kickoff(inputs={
            'user_query': user_query,
            'book_info': self.book_info,
            'book_title': cl.user_session.get("book_title")
        })
        
        # Extract relevant information from CrewOutput
        if hasattr(result, 'result'):
            extracted_result = {
                'result': str(result.result),
                'task_id': str(getattr(result, 'task_id', '')),
                'agent_id': str(getattr(result, 'agent_id', '')),
                'task_output': str(getattr(result, 'task_output', ''))
            }
        else:
            extracted_result = {'result': str(result)}
        
        return extracted_result  # Remove JSON serialization here

# Chainlit interface

@cl.on_chat_start
async def chat_start():
    book_title_msg = await cl.AskUserMessage(content="What book would you like to review?", timeout=60).send()
    if book_title_msg is None:
        await cl.Message(content="No input received. Ending the conversation.").send()
        return

    book_title = book_title_msg['content']
    crew_instance = BookReviewCrew()
    
    await cl.Message("Gathering information about the book...").send()
    initial_info = await cl.make_async(crew_instance.initial_run)(book_title)
    
    cl.user_session.set("crew", crew_instance)
    cl.user_session.set("book_title", book_title)
    
    await cl.Message(f"Great! I've gathered information about '{book_title}'. What would you like to know?").send()
    await cl.Message(content=f"Initial book info: {initial_info}").send()
@cl.on_message
async def main(message: cl.Message):
    crew_instance = cl.user_session.get("crew")
    if crew_instance is None:
        await cl.Message(content="Crew instance not found. Please start a new conversation.").send()
        return

    book_title = cl.user_session.get("book_title")
    try:
        result = await cl.make_async(crew_instance.qa_run)(message.content)
        
        # Extract the result from the output
        result_content = result.get('result', str(result))
        
        await cl.Message(content=result_content).send()

        # Save report
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        filename = f"reports/{book_title.replace(' ', '_').lower()}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'a') as f:
            f.write(f"User Query: {message.content}\n")
            f.write(f"AI Response: {result_content}\n\n")
        
        await cl.Message(f"Response added to report: {filename}").send()
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await cl.Message(content=error_message).send()
        print(f"Error in main function: {e}")
if __name__ == "__main__":
    cl.run()