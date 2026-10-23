
import streamlit as st
import requests
import chromadb
import hashlib
import time
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Chat With Your Documents",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat With Your Documents")
st.caption("Codomax Module 4 | RAG, Embeddings & Vector Databases")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Initialize vector database
@st.cache_resource
def load_database():
    return chromadb.EphemeralClient()

def extract_text(uploaded_file):
    uploaded_file.seek(0)

    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
    else:
        text = uploaded_file.getvalue().decode(
            "utf-8", errors="replace"
        )

    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def create_vector_database(chunks, document_id):
    model = load_embedding_model()
    client = load_database()

    collection_name = "doc_" + document_id[:20]

    collection = client.get_or_create_collection(
        name=collection_name
    )

    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"chunk_number": i + 1}
            for i in range(len(chunks))
        ]
    )

    return collection_name

def retrieve_context(question, collection_name, top_k=3):
    model = load_embedding_model()
    client = load_database()

    collection = client.get_collection(collection_name)

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas"]
    )

    documents = results["documents"][0]
    metadata = results["metadatas"][0]

    context = "\n\n".join(
        f"[Source {i+1}, Chunk {meta['chunk_number']}]\n{doc}"
        for i, (doc, meta) in enumerate(
            zip(documents, metadata)
        )
    )

    return context, documents

def ask_gemini(question, context, api_key):
    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-3.5-flash-lite:generateContent"
    )

    prompt = f"""
You are a document-based AI assistant.

Answer the user's question using ONLY the provided context.
If the answer is not present, clearly say:
"I could not find this information in the uploaded document."

Preserve code formatting when the question asks for code.
Mention relevant source numbers when answering.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 1500,
            "temperature": 0.2
        }
    }

    for attempt in range(3):
        try:
            response = requests.post(
                url,
                headers={"x-goog-api-key": api_key},
                json=payload,
                timeout=45
            )

            if response.status_code in (
                429, 500, 502, 503, 504
            ) and attempt < 2:
                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()

            data = response.json()
            parts = data["candidates"][0]["content"]["parts"]

            return "\n".join(
                part.get("text", "")
                for part in parts
            )

        except requests.exceptions.Timeout:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=5,
        value=3
    )

    st.info(
        "This app uses Sentence Transformers, "
        "ChromaDB, and Gemini for RAG."
    )

# File upload
uploaded_file = st.file_uploader(
    "Upload a PDF or TXT document",
    type=["pdf", "txt"]
)

api_key = st.secrets.get("GEMINI_API_KEY", "")

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    document_id = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.get("document_id") != document_id:
        with st.spinner("Processing document and creating embeddings..."):
            try:
                text = extract_text(uploaded_file)

                if not text.strip():
                    st.error("No readable text found in this document.")
                    st.stop()

                chunks = chunk_text(text)

                collection_name = create_vector_database(
                    chunks, document_id
                )

                st.session_state.document_id = document_id
                st.session_state.collection_name = collection_name
                st.session_state.messages = []

            except Exception as e:
                st.error(f"Document processing failed: {e}")
                st.stop()

    st.success("Document processed successfully!")

    st.write(
        "You can now ask questions about your document."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                context, sources = retrieve_context(
                    question,
                    st.session_state.collection_name,
                    top_k
                )

                if not api_key:
                    st.error(
                        "Please configure GEMINI_API_KEY "
                        "in Streamlit Secrets."
                    )
                else:
                    with st.spinner("Generating answer..."):
                        answer = ask_gemini(
                            question, context, api_key
                        )

                    st.markdown(answer)

                    with st.expander("📖 Retrieved Sources"):
                        for i, source in enumerate(
                            sources, start=1
                        ):
                            st.markdown(f"**Source {i}**")
                            st.text(source)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

            except requests.exceptions.HTTPError as e:
                st.error(
                    f"Gemini API error: "
                    f"{e.response.status_code}"
                )
                st.code(e.response.text[:500])

            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    st.info("Upload a document to begin asking questions.")
