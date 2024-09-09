import os
import yaml
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from crewai_tools import SerperDevTool, RagTool

@CrewBase
class BookReviewCrew:
    """Book Review Crew for gathering and analyzing book opinions"""
    
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.agents_config = self.load_config('agents.yaml')
        self.tasks_config = self.load_config('tasks.yaml')
        self.rag_tool = self.create_rag_tool()

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

    def create_rag_tool(self):
        return RagTool(
            name="Book Knowledge Base",
            description="A knowledge base for storing and retrieving book-related information.",
            summarize=True
        )

    @agent
    def searcher_goodreads(self) -> Agent:
        config = self.agents_config['searcher_goodreads']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=[SerperDevTool()],
            verbose=True,
            llm=self.get_llm('gpt-4')
        )

    @agent
    def reddit_reviewer(self) -> Agent:
        config = self.agents_config['reddit_reviewer']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=[SerperDevTool()],
            verbose=True,
            llm=self.get_llm('gpt-4')
        )

    @agent
    def data_processor(self) -> Agent:
        config = self.agents_config['data_processor']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            verbose=True,
            llm=self.get_llm('gpt-4')
        )

    @agent
    def rag_system_manager(self) -> Agent:
        config = self.agents_config['rag_system_manager']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=[self.rag_tool],
            verbose=True,
            llm=self.get_llm('gpt-4')
        )

    @agent
    def chat_interface(self) -> Agent:
        config = self.agents_config['chat_interface']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            verbose=True,
            llm=self.get_llm('gpt-4')
        )

    @task
    def gather_goodreads_reviews_task(self, book_title: str) -> Task:
        config = self.tasks_config['gather_goodreads_reviews_task']
        return Task(
            description=config['description'].format(book_title=book_title),
            agent=self.searcher_goodreads(),
            expected_output=config['expected_output']
        )

    @task
    def gather_reddit_reviews_task(self, book_title: str) -> Task:
        config = self.tasks_config['gather_reddit_reviews_task']
        return Task(
            description=config['description'].format(book_title=book_title),
            agent=self.reddit_reviewer(),
            expected_output=config['expected_output']
        )

    @task
    def process_opinion_data(self) -> Task:
        config = self.tasks_config['process_opinion_data']
        return Task(
            description=config['description'],
            agent=self.data_processor(),
            expected_output=config['expected_output']
        )

    @task
    def insert_data_task(self, data: str) -> Task:
        config = self.tasks_config['insert_data_task']
        return Task(
            description=f"{config['description']}: {data}",
            agent=self.rag_system_manager(),
            expected_output=config['expected_output']
        )

    @task
    def query_database_task(self, query: str) -> Task:
        config = self.tasks_config['query_database_task']
        return Task(
            description=f"{config['description']}: '{query}'",
            agent=self.rag_system_manager(),
            expected_output=config['expected_output']
        )

    @task
    def update_data_task(self, identifier: str, new_data: str) -> Task:
        config = self.tasks_config['update_data_task']
        return Task(
            description=f"{config['description']}: Update '{identifier}' with {new_data}",
            agent=self.rag_system_manager(),
            expected_output=config['expected_output']
        )

    @task
    def delete_data_task(self, identifier: str) -> Task:
        config = self.tasks_config['delete_data_task']
        return Task(
            description=f"{config['description']}: Delete data with identifier '{identifier}'",
            agent=self.rag_system_manager(),
            expected_output=config['expected_output']
        )

    @task
    def handle_user_query(self, query: str) -> Task:
        config = self.tasks_config['handle_user_query']
        return Task(
            description=f"{config['description']}: '{query}'",
            agent=self.chat_interface(),
            expected_output=config['expected_output']
        )

    @crew
    def run_crew(self, book_title: str, user_query: str = None) -> Crew:
        """Assemble and run the Book Review Crew"""
        tasks = [
            self.gather_goodreads_reviews_task(book_title),
            self.gather_reddit_reviews_task(book_title),
            self.process_opinion_data(),
            self.insert_data_task("Processed opinion data")
        ]
        
        if user_query:
            tasks.append(self.query_database_task(user_query))
            tasks.append(self.handle_user_query(user_query))

        return Crew(
            agents=[
                self.searcher_goodreads(),
                self.reddit_reviewer(),
                self.data_processor(),
                self.rag_system_manager(),
                self.chat_interface()
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )