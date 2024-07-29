import argparse
import os
import openai
import pdfplumber
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
CHROMA_PATH = "chroma"

def main():
    parser = argparse.ArgumentParser(description="Match CVs to a job description")
    parser.add_argument("job_description_path", type=str, help="Path to the job description file in PDF format")
    args = parser.parse_args()

    # Extract text from job description PDF
    job_description_text = extract_text_from_pdf(args.job_description_path)

    # Query the Chroma database
    results = query_chroma_database(job_description_text)

    # Print results
    for doc, score in results:
        print(f"Score: {score:.2f}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content: {doc.page_content[:500]}...")  # Print first 500 characters of content
        print("-----")

def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or '') + ' '
    return text

def query_chroma_database(query_text):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text, k=5)  # Adjust k as needed
    return results

if __name__ == "__main__":
    main()
