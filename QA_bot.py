import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

CHROMA_DB_DIR = "chroma_db"

print("Loading embeddings...")
embeddings = OpenAIEmbeddings()

print("Loading vector database...")
vector_store = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
)

print("Loading GPT model...")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def ask_question(question):
    docs = vector_store.similarity_search(question, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an IRC Code AI Assistant.
Answer the question using only the given context.
If the answer is not found in the context, say:
"The answer is not available in the provided IRC document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown")
        })

    return response.content, sources

def main():
    print("\n======================================")
    print(" IRC Code AI Assistant")
    print("======================================")

    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        answer, sources = ask_question(question)

        print("\nANSWER:\n")
        print(answer)

        print("\nSOURCES:")
        for i, source in enumerate(sources, start=1):
            print(f"{i}. {source['source']} | Page {source['page']}")

if __name__ == "__main__":
    main()