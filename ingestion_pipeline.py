import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PDF_PATH = "irc.gov.in.108.1996.pdf"

CHROMA_DB_DIR = "chroma_db"

# --------------------------------------------------
# Load PDF
# --------------------------------------------------

def load_documents():

    print(f"\nLoading PDF: {PDF_PATH}")

    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    print(f"Total Pages Loaded: {len(documents)}")

    return documents

# --------------------------------------------------
# Chunk Documents
# --------------------------------------------------

def split_documents(documents):

    print("\nSplitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total Chunks Created: {len(chunks)}")

    return chunks

# --------------------------------------------------
# Create Embeddings + Vector Store
# --------------------------------------------------

def create_vector_store(chunks):

    print("\nCreating embeddings...")

    embeddings = OpenAIEmbeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print("Embeddings created successfully")

    print(f"Vector database stored in: {CHROMA_DB_DIR}")

    return vector_store

# --------------------------------------------------
# Main Function
# --------------------------------------------------

def main():

    documents = load_documents()

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    print("\nIngestion Pipeline Completed Successfully")

# --------------------------------------------------

if __name__ == "__main__":
    main()