# Chat With Your Documents — Codomax Module 4

A Streamlit document question-answering application. Upload PDF/TXT, extract text, split it into overlapping chunks, vectorize chunks using **TF-IDF**, retrieve the closest chunks by cosine similarity, and ask Gemini to answer with references to the retrieved excerpts.

## Important scope note
This is a **lightweight RAG prototype**. It uses sparse TF-IDF vectors held in memory, **not neural embeddings or a dedicated persistent vector database**. To fully meet a strict embeddings/vector-database requirement, upgrade retrieval to a sentence-transformer or embedding API and Chroma/FAISS (with persistence where needed).

## Features
- PDF and TXT upload
- Overlapping text chunks
- TF-IDF vectorization and similarity retrieval
- Gemini grounded answers and source excerpts
- Basic session chat history
- Configurable model and top-k retrieval
- Error handling and retry for temporary API issues

## Deploy on Streamlit Cloud
1. Upload the files in this folder to a new public GitHub repository.
2. On Streamlit Community Cloud, select the repository and `app.py` as the main file.
3. Under app Settings → Secrets, add `GEMINI_API_KEY = "your_actual_key"`.
4. Save, wait for deployment, upload a text-based PDF or TXT, and test a question.
5. Never commit your API key or `secrets.toml`.

Scanned PDFs without selectable text are not supported. Gemini API quotas, availability and pricing vary by account/model.
