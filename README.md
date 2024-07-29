# CV Matching with Job Descriptions

This project uses OpenAI embeddings to create a system for matching CVs (in PDF format) with job descriptions (also in PDF format). The system extracts text from PDF files, generates embeddings, stores them in a Chroma vector database, and allows querying to find the most relevant CVs for a given job description.

## Features

- Extract text from CV and job description PDFs.
- Generate and store embeddings for CVs in a Chroma vector database.
- Query the database with a job description to find the most relevant CVs based on semantic similarity.

## Prerequisites

- Python 3.11 or later
- OpenAI API Key (You need to sign up at [OpenAI](https://www.openai.com/) to get an API key)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/cv-matching.git
    cd cv-matching
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the spaCy language model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

5. Set up your environment variables:
    - Create a `.env` file in the project directory and add your OpenAI API key:
        ```
        OPENAI_API_KEY=your_openai_api_key_here
        ```

## Usage

### Step 1: Create the Database

To populate the Chroma database with CVs:

```bash
python create_database.py
