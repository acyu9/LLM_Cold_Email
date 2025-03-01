import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# To run: streamlit run main.py

def create_streamlit_app(llm, portfolio, clean_text):
    st.title("Cold Email Generator")
    url_input = st.text_input("Enter a URL: ", value = 'https://jobs.intuit.com/job/bengaluru/software-engineer-2/27595/77913719936')
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_data_to_chromadb()
            jobs = llm.extract_jobs(data)
            
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_email(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f'An Error Occured: {e}')


if __name__ == '__main__':
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout='wide', page_title='Cold Email Generator')
    create_streamlit_app(chain, portfolio, clean_text)
