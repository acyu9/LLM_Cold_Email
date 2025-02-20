from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_CLOUD_API_KEY")

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# response = llm.invoke("The first person to land on the moon was...")
# print(response.content)

# Scrape the website
loader = WebBaseLoader(
    web_path = "https://careers.nike.com/software-engineer/job/R-50095"
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
# result is str in json format
# print(result.content)

json_parser = JsonOutputParser()
json_result = json_parser.parse(result.content)
print(json_result)