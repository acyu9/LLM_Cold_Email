from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from chroma_database import load_data_to_chromadb, query_chromadb
from job_extractor import extract_job_details
from email_generator import generate_email

load_dotenv()
api_key = os.getenv("GROQ_CLOUD_API_KEY")

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# Step 1: Extract job details from the website
job_url = "https://jobs.intuit.com/job/bengaluru/software-engineer-2/27595/77913719936"
job = extract_job_details(job_url, llm)

# Step 2: Load portfolio data to ChromaDB for efficient and semantic search
load_data_to_chromadb('my_portfolio.csv', 'portfolio')

# Step 3: Query ChromaDB using skills from the job description for the relevant links
job_skills = job['skills']
links = query_chromadb('portfolio', job_skills)

# Step 4: Generate cold email
email = generate_email(job, links, llm)
print(email)