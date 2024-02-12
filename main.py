import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain.agents import Tool
from langchain.agents import load_tools
from crewai import Agent, Task, Process, Crew
from langchain_community.utilities import GoogleSerperAPIWrapper

# to get your api key for free, visit and signup: https://serper.dev/
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

search = GoogleSerperAPIWrapper()

search_tool = Tool(
    name="Scrape google searches",
    func=search.run,
    description="useful for when you need to search the internet",
)

# Loading Human Tools
human_tools = load_tools(["human"])

# To Load GPT-4
api = os.getenv("OPENAI_API_KEY")

# Define your agents with roles and goals
researcher = Agent(
    role="Content Researcher",
    goal="Find and explore the most exciting and relevant content and consumer trends to ensure that content is relevant and appealing, focusing on content published in 2024",
    backstory="""You are an expert strategist who knows how to spot emerging trends and companies. You know how 
    to discern between actual projects and companies and factious pages used for advertising. You're great at finding interesting, exciting content on the web. 
    You source scraped data to create detailed summaries containing names of the most exciting projects, products, trends and companies on the web. 
    ONLY use scraped text from the web for your summary report.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]+human_tools,
)

writer = Agent(
    role="Senior Writer",
    goal="Write an engaging and interesting blog post using the researchers summary report in simple, layman's vocabulary",
    backstory="""You create engaging content for blogs. You will create highly informative, relevant, valuable, and engaging blog posts on various topics. 
    You will be expected to write five or more paragraph articles, proofread your articles before submitting the drafts, and revise articles after receiving feedback from the editor. The articles must be unique 
    100% factually correct, easy to read, and SEO friendly. You only use data that was provided to you from the researcher for the blog and site your sources.""",
    verbose=True,
    allow_delegation=True,
    tools=human_tools,
)

editor = Agent(
    role="Expert Writing Critic and Editor",
    goal="Provide feedback and criticize blog post drafts. Make sure that the tone and writing style is compelling, simple and concise",
    backstory="""You are an Expert at providing feedback to the writers. You can tell when a blog text isn't concise,
    simple, or engaging enough. You know how to provide helpful feedback that can improve any text. You know how to ensure that text 
    stays relevant and insightful by using layman's terms.""",
    verbose=True,
    allow_delegation=True,
)

# Create tasks for your agents
# Being explicit on the task to ask for human feedback.
task_report = Task(
    description="""Use and summarize scraped data from the internet to make a detailed report on the latest content based on the theme provided by the human. Use ONLY 
    scraped data from the web to generate the report. Your final answer MUST be a full analysis report, text only; ignore any code or anything that 
    isn't text. The report has to have bullet points and 5-10 defining the scraped content. Write the names of every tool and project. 
    Each bullet point MUST contain 4 sentences that refer to one specific company, product, model, or anything you found online related to the theme provided by the human. 
    Include text from the scraped data in the report as quoted.""",
    agent=researcher,
)

task_blog = Task(
    description="""Write a blog article with text only and with a short but impactful headline and at least 10 paragraphs. The blog should summarize 
    the content the human has chosen from the researcher's report. The content in the blog post should be factual from the report. Style and tone should be compelling, concise, fun, and technical, but also use 
    layman's terms for the general public. Name specific new, exciting projects, apps, and companies in the report. Don't 
    write "**Paragraph [number of the paragraph]:**"; instead, start the new paragraph in a new line. Write the names of projects and tools in BOLD.
    ALWAYS include links to projects/tools/research papers. ONLY include information from the report.""",
    agent=writer,
)

task_critique = Task(
    description="""The Output MUST be concise, simple, and engaging. Provide helpful feedback that can improve the writing. Ensure that the text 
    stays relevant and insightful by using layman's terms.
    Make sure that it does; if it doesn't, have the writer rewrite it accordingly.
    """,
    agent=editor,
)

# instantiate crew of agents
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[task_report, task_blog, task_critique],
    verbose=2,
    process=Process.sequential,  # Sequential process will have tasks executed one after the other, and the outcome of the previous one is passed as extra content into this next.
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)

