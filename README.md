# Retrieval-Augmented Chatbot (LangChain + Chroma + BM25) 📚🤖

## What is this

This project builds a Retrieval-Augmented Generation (RAG) chatbot over a custom document collection.  
It lets you load `.txt` documents (placed in a `docs/` folder), embed them into a vector store using Chroma, and then use combined semantic + keyword retrieval (hybrid of embedding-based and keyword-based search) to answer user queries via a Large-Language Model (LLM).  

In short — you get a chatbot that answers questions *based on your own documents*.

## Repository Structure

```
/docs/                   ← Folder containing your .txt documents  
chroma_db/              ← Persisted Chroma vector-store data (local DB)  
hybride_search.py       ← Main script: handles chat interface + retrieval + answering  
ingestion_pipeline.py   ← Script to load, split, embed and store documents  
requirements.txt        ← Python dependencies  
.env                     ← Environment variables (e.g. API key) — NOT to commit  
venv/                    ← Virtual environment folder — ignored  
.gitignore              ← Specifies files/folders to ignore (venv, .env, chroma_db, etc.)  
```

## Prerequisites & Setup

1. Python 3.10+ (or compatible)  
2. Create & activate a virtual environment  
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux / macOS
   source venv/bin/activate
   ```  
3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```  
4. Create a `.env` file in the project root and set your API key:  
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```  
5. Add your `.txt` documents into the `docs/` folder.

## How to Use

### 1. Build / Ingest documents  
Run:  
```bash
python ingestion_pipeline.py
```  
This will:  
- Load all `.txt` files from `docs/`  
- Split them into chunks  
- Embed chunks and store them in the `chroma_db/` vector database  

### 2. Start Chat / Ask Questions  
Run:  
```bash
python hybride_search.py
```  
Then you can ask questions via CLI. The system will retrieve relevant document chunks (hybrid retrieval), send them to the LLM, and print the answer — grounded in your documents.

## What it does (Features)

- Semantic vector-based retrieval (embeddings) for conceptual similarity  
- Keyword-based retrieval (BM25) for lexical matching  
- Hybrid retrieval combining both — improves retrieval recall and accuracy  
- Document chunking — handles large documents efficiently  
- Easy ingestion: just `.txt` files, no need for fine-tuning  
- LLM-backed QA on your document knowledge base  

## Limitations & What to Keep in Mind

- Currently supports only `.txt` documents  
- If many documents / chunks are retrieved, prompt size may be large — watch out for LLM token limits  
- You must manage/update `docs/` and rerun ingestion if you add/modify documents  
- Don’t commit / share `.env` (contains API key), `venv/`, or `chroma_db/` — they are in `.gitignore`

## Possible Improvements / Future Work

- Support for other document formats (PDF, DOCX, HTML)  
- Add a web or GUI interface instead of CLI (e.g. via a web server or web framework)  
- Automate incremental ingestion (auto-detect new docs and embed)  
- Add configuration options: chunk size, retriever weights, embedding model, etc.  
- Add tests, error handling, better prompt design, and caching  

## Author & License

- Author: *Your Name or GitHub Username*  
- License: MIT (or choose appropriate license)

---

**Thanks** for checking out the project — feel free to fork, extend, or give feedback!  
