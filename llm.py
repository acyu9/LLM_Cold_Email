from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os
from chroma_database import load_data_to_chromadb, query_chromadb

load_dotenv()
api_key = os.getenv("GROQ_CLOUD_API_KEY")

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# Scrape the website
loader = WebBaseLoader(
    web_path = "https://jobs.intuit.com/job/bengaluru/software-engineer-2/27595/77913719936"
)

page_data = loader.load().pop().page_content
# print(page_data)

prompt_extract = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTION:
    The scraped text is from the career's page of a website.
    Your job is to extract the job postings and return them in JSON format containing
    the following keys: 'role', 'experience', 'skills', and 'description'.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE):
    """
)

# Chain/pipeline | that passes the prompt to llm
chain_extract = prompt_extract | llm
result = chain_extract.invoke(input={'page_data':page_data})
# print(result.content)

# Change result from str to json (dict)
json_parser = JsonOutputParser()
job = json_parser.parse(result.content)
# print(job)
# print(job['skills'])

# Get links with query texts (Python, React)
load_data_to_chromadb('my_portfolio.csv', 'portfolio')
links = query_chromadb('portfolio', ['Experience in Python', 'Expertise in React'])
# print(links)

prompt_email = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}

    ### INSTRUCTION:
    You are Bob, a business development executive at XYZ. XYZ is
    an AI & Software consulting company dedicated to the seamless
    integration of business processes through automated tools.
    Your job is to write a cold email to the client regarding the job
    mentioned above describing the capability in fulfilling their needs.
    Also add the most relevant ones from the following links to 
    showcase XYZ's portfolio: {link_list}.
    Remember you are Bob, BDE at XYZ.
    Do not provide a preamble.

    ### EMAIL (NO PREAMBLE):
    """
)

# Generate email
chain_email = prompt_email | llm
res = chain_email.invoke({'job_description': str(job), 'link_list': links})
print(res.content)