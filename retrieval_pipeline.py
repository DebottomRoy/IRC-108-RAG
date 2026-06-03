import os
from dotenv import load_dotenv

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

CHROMA_DB_DIR = "chroma_db"

# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

print("Loading embedding model...")

embeddings = OpenAIEmbeddings()

# --------------------------------------------------
# Load Existing Chroma Database
# --------------------------------------------------

print("Loading vector database...")

vector_store = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
)

print("Vector database loaded successfully!")

# --------------------------------------------------
# Retrieval Function
# --------------------------------------------------

def retrieve_documents(query, k=3):

    results = vector_store.similarity_search(
        query=query,
        k=k
    )

    return results

# --------------------------------------------------
# Main Query Loop
# --------------------------------------------------

def main():

    print("\n===================================")
    print(" IRC Code Retrieval System")
    print("===================================")

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("\nExiting...")
            break

        documents = retrieve_documents(query)

        print("\nTop Retrieved Chunks:\n")

        for index, doc in enumerate(documents, start=1):

            print(f"\n{'='*20} Chunk {index} {'='*20}")

            print(doc.page_content)

            print(f"\nMetadata: {doc.metadata}")

            print(f"\n{'='*50}")

# --------------------------------------------------

if __name__ == "__main__":
    main()