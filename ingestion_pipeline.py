# filename: chroma_demo.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from functools import partial
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load .env to get OPENAI_API_KEY
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# 1) Load your documents (all .txt files in "docs" folder)
loader = DirectoryLoader(
    path="docs",                # your docs folder
    glob="*.txt",
    loader_cls=partial(TextLoader, encoding="utf8")
)
documents = loader.load()

if len(documents) == 0:
    raise ValueError("No text documents found in docs/")

# 2) Optionally split them into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_chunks = text_splitter.split_documents(documents)

# 3) Create embedding function
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              openai_api_key=OPENAI_API_KEY)

# 4) Create (or overwrite) persistent Chroma vector store from chunks
persist_dir = "chroma_db"
collection_name = "my_collection"

vector_store = Chroma.from_documents(
    documents=docs_chunks,
    embedding=embeddings,
    persist_directory=persist_dir,
    collection_name=collection_name
)

print(">>> Stored", len(docs_chunks), "chunks into Chroma vector store.")

# --- now reload and test retrieval from same store ---

db = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings,
    collection_name=collection_name  # important: match the same collection name
)

query = "Google"  # or another query you expect matches for
print("Querying:", query)
results = db.similarity_search(query, k=3)
print("Number of results:", len(results))
for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Source:", doc.metadata.get("source"))
    print("Preview:", doc.page_content[:200].replace("\n", " "), "...")
