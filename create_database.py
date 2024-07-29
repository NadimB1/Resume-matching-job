from langchain_community.document_loaders import DirectoryLoader
import openai
import pdfplumber
import os
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import shutil

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
CHROMA_PATH = "chroma"
DATA_PATH = "data/cvs"  # Assuming PDF CVs are stored here

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or '') + ' '
    return text

def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.pdf'):
            filepath = os.path.join(DATA_PATH, filename)
            text = extract_text_from_pdf(filepath)
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents

def split_text(documents):
    # Assuming we don't split into smaller chunks for simplicity
    return documents

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
    db.persist()
    print(f"Saved {len(chunks)} documents to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
