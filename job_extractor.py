from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

def extract_job_details(url: str, llm: ChatGroq):
    """Scrape job details from a URL and return extracted job info in JSON format."""
    loader = WebBaseLoader(web_path=url)
    page_data = loader.load().pop().page_content
    
    # Define what to extract from the website
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
    result = chain_extract.invoke(input={'page_data': page_data})

    # Change result from str to json (dict)
    json_parser = JsonOutputParser()
    job = json_parser.parse(result.content)

    if isinstance(job, list):
        job = job[0]

    return job
