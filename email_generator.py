from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

def generate_email(job_description: dict, link_list: list, llm: ChatGroq):
    """Generate an email based on the job description and relevant links."""

    # Email template
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
    res = chain_email.invoke({'job_description': str(job_description), 'link_list': link_list})
    
    return res.content
