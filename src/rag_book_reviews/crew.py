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
        self.conversation_active = True

    def load_config(self, filename):
        config_path = os.path.join(self.base_path, 'config', filename)
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def get_llm(self, model_name):
        print(f"Attempting to get LLM for model: {model_name}")
        if model_name.startswith('gpt'):
            llm = ChatOpenAI(model_name=model_name)
        elif model_name.startswith('claude'):
            llm = ChatAnthropic(model=model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        if llm is None:
            print(f"LLM for model {model_name} is None")
        return llm

    def create_rag_tool(self):
        return RagTool(
            name="Book Knowledge Base",
            description="A knowledge base for storing and retrieving book-related information as well as opinions from Goodreads and Reddit.",
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
            llm=self.get_llm('gpt-4o'),
            max_iterations=100,
            timeout=120
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
            llm=self.get_llm('gpt-4o'),
            max_iterations=100,
            timeout=120
        )

    @agent
    def data_processor(self) -> Agent:
        config = self.agents_config['data_processor']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            verbose=True,
            llm=self.get_llm('gpt-4o-mini'),
            max_iterations=100,
            timeout=120
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
            llm=self.get_llm('gpt-4o'),
            max_iterations=100,
            timeout=120
        )

    @agent
    def chat_interface(self) -> Agent:
        config = self.agents_config['chat_interface']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=[self.rag_tool],
            verbose=True,
            llm=self.get_llm('gpt-4o'),
            max_iterations=100,
            timeout=120
        )

    @task
    def gather_goodreads_reviews_task(self) -> Task:
        config = self.tasks_config['gather_goodreads_reviews_task']
        return Task(
            description=config['description'],
            agent=self.searcher_goodreads(),
            expected_output=config['expected_output']
        )

    @task
    def gather_reddit_reviews_task(self) -> Task:
        config = self.tasks_config['gather_reddit_reviews_task']
        return Task(
            description=config['description'],
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
    def insert_data_task(self) -> Task:
        config = self.tasks_config['insert_data_task']
        return Task(
            description=config['description'],
            agent=self.rag_system_manager(),
            expected_output=config['expected_output']
        )

    @task
    def query_database_task(self) -> Task:
        config = self.tasks_config['query_database_task']
        return Task(
            description=config['description'],
            agent=self.rag_system_manager(),
            expected_output=config['expected_output']
        )

    @task
    def handle_user_query(self) -> Task:
        config = self.tasks_config['handle_user_query']
        return Task(
            description=config['description'],
            agent=self.chat_interface(),
            expected_output=config['expected_output']
        )

    @task
    def continuous_chat(self) -> Task:
        config = self.tasks_config['continuous_chat']
        return Task(
            description=config['description'],
            agent=self.chat_interface(),
            expected_output=config['expected_output'],
            human_input=True
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Book Review Crew"""
        agent_methods = [
            self.searcher_goodreads,
            self.reddit_reviewer,
            self.data_processor,
            self.rag_system_manager,
            self.chat_interface
        ]
        
        agents = []
        for i, agent_method in enumerate(agent_methods):
            agent = agent_method()
            if agent is None:
                print(f"Agent at index {i} ({agent_method.__name__}) is None.")
            agents.append(agent)

        # Check if any agent is None
        for i, agent in enumerate(agents):
            if agent is None:
                raise ValueError(f"Agent at index {i} is None. Check the agent creation method: {agent_methods[i].__name__}")

        tasks = [
            self.gather_goodreads_reviews_task(),
            self.gather_reddit_reviews_task(),
            self.process_opinion_data(),
            self.insert_data_task(),
            self.query_database_task(),
            self.handle_user_query(),
            self.continuous_chat()
        ]

        # Check if any task is None
        for i, task in enumerate(tasks):
            if task is None:
                raise ValueError(f"Task at index {i} is None. Check the task creation methods.")

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

    async def run(self):
        crew_result = await self.crew().run()
        print("Initial crew tasks completed.")
        print(crew_result)

        chat_task = self.continuous_chat()
        
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                print("Chat ended.")
                break
                        
            response = await chat_task.run()
            print(f"Assistant: {response}")

    # Remove or comment out the generate_response method if it's not being used
    # async def generate_response(self, user_input):
    #     ...