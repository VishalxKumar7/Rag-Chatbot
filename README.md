Chroma Hybrid RAG Chatbot (README Instructions)
This project demonstrates how to build a hybrid Retrieval-Augmented Generation (RAG) chatbot using LangChain, Chroma, and OpenAI models.
It combines dense embeddings (vector similarity) with BM25 keyword search to improve document retrieval quality.

1. Project Setup
Create a project folder and place your text documents inside a subfolder named docs/.

Create a Python virtual environment and install required dependencies:

text
pip install langchain langchain_community langchain_chroma langchain_openai python-dotenv
Create a .env file in the project root and add your OpenAI API key:

text
OPENAI_API_KEY=your_openai_api_key_here
2. How the Script Works
Step 1 — Load Environment Variables
The script loads the .env file to access the OPENAI_API_KEY.
If the key is missing, it raises an error and stops execution.

Step 2 — Load Text Documents
All .txt files in the docs/ folder are loaded using DirectoryLoader and TextLoader.
If no documents are found, the script raises an error.

Step 3 — Split Documents (First Section)
Large text files are split into smaller overlapping chunks using CharacterTextSplitter
(chunk size = 1000 characters, overlap = 200). This makes them easier to embed for retrieval.

Step 4 — Generate Embeddings and Build Chroma Store
Embeddings are created using the OpenAI text-embedding-3-small model via OpenAIEmbeddings.
These embeddings are stored in a Chroma vector database at the folder chroma_db/ under a chosen collection_name.
If the folder doesn’t exist, Chroma creates it automatically.

Step 5 — Verify Stored Chunks
After storing, the script reconnects to the same Chroma database using the same collection_name.
It runs a sample test query (e.g., “Google”) using similarity search to verify retrieval.

Step 6 — Create Dual Retrievers
The chatbot section sets up two different retrieval systems:

Vector Retriever (semantic search) — based on embeddings stored in Chroma.

BM25 Retriever (keyword-based search) — built directly from the text documents.

These are combined using EnsembleRetriever with equal weights (0.5, 0.5) to form a Hybrid Retriever.

Step 7 — Define the Chat Model
A ChatOpenAI model (gpt-4o) is created to answer user queries based on retrieved document context.
This model powers the actual Q&A responses.

Step 8 — Implement Query Rewriting (Context Awareness)
If previous chat history exists, a helper prompt asks the model to rewrite the new query as a standalone question.
This ensures clarity and relevance in multi-turn conversations.

Step 9 — Retrieve and Prepare Document Context
When the user asks a question:

The hybrid retriever finds the top relevant chunks (k=15 for BM25, k=4 for vector).

Long chunks are truncated to a safe character limit (max_chars = 1000).

These chunks are combined into a formatted prompt that includes the question and the supporting text snippets.

Step 10 — Generate Response
The chat model (gpt-4o) receives a prompt like:

"Based on the following documents, answer the question..."

It then crafts an answer, or replies with “I don’t have enough information” if necessary context is missing.

Step 11 — Maintain Chat History
The interaction history is stored in a chat_history list that contains alternating HumanMessage and AIMessage objects.
This ensures that future questions can use previous context for rewriting and coherence.

Step 12 — Run the Chat Interface
The start_chat() function provides a simple command-line chat loop.

Type a question to query the hybrid system.

Type exit or quit to end the conversation.

3. Summary of Components
Component	Role
DirectoryLoader, TextLoader	Load and read .txt documents
CharacterTextSplitter	Split documents into overlapping chunks
OpenAIEmbeddings	Convert chunks into dense vector embeddings
Chroma	Store and query document embeddings
BM25Retriever	Perform keyword-based retrieval
EnsembleRetriever	Combine semantic + keyword retrieval
ChatOpenAI	Generate context-aware answers
dotenv	Securely manage API keys
4. How to Run
Place your documents in the docs/ folder.

Run the script:

text
python chroma_demo.py
Wait for embeddings to generate and Chroma store to persist.

When the CLI prompt appears, type your question.

Review generated answers and retrieved document sources printed in the console.
