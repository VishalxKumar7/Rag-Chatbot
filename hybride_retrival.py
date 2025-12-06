from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from functools import partial
from langchain_chroma import Chroma

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

# Load all documents from 'docs' folder for BM25  
def load_all_docs(docs_path="docs"):
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=partial(TextLoader, encoding="utf8"),
    )
    return loader.load()

documents = load_all_docs("docs")

# Setup vector store (Chroma)
persistent_directory = "chroma_db"
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)
vector_retriever = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_name="rag_chatbot"
)

# BM25 retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 15

# Vector retriever
vector_retriever = vector_retriever.as_retriever(search_kwargs={"k": 4})

# Hybrid retriever (BM25 + vector)
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

model = ChatOpenAI(model="gpt-4o")
chat_history = []

def ask_question(user_query):
    print(f"\n--- Your Question: {user_query} ---")

    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable."),
            *chat_history,
            HumanMessage(content=f"Rewrite the question: {user_query}")
        ]
        result = model.invoke(messages)
        search_query = result.content.strip()
        print("Search for:", search_query)
    else:
        search_query = user_query

    docs = hybrid_retriever.invoke(search_query)

    # --- truncate doc text before building prompt ---
    max_chars = 1000  # adjust as needed
    truncated_texts = []
    for doc in docs:
        text = doc.page_content
        if len(text) > max_chars:
            text = text[:max_chars]
        truncated_texts.append(text)

    combined_input = (
        f"Based on the following documents, answer the question: {user_query}\n\n"
        + "\n".join(f"- {text}" for text in truncated_texts)
        + "\n\nIf you can't find the answer, say \"I don't have enough information\"."
    )

    messages = [
        SystemMessage(content="You are a helpful assistant answering based on provided documents."),
        *chat_history,
        HumanMessage(content=combined_input),
    ]

    result = model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=answer))

    print("Answer:", answer)
    return answer

def start_chat():
    print("Ask your question! Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()
