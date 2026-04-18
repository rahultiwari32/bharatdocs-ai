import os
import tempfile
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from groq import Groq
import streamlit as st

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Helper Functions ---
def load_and_index_pdf(uploaded_file):
    """Load uploaded PDF, chunk it and create FAISS index."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    reader = PdfReader(tmp_path)
    chunks = []
    metadatas = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            page_chunks = splitter.split_text(text)
            for chunk in page_chunks:
                chunks.append(chunk)
                metadatas.append({"page": page_num + 1,
                                  "source": uploaded_file.name})

    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")
    os.unlink(tmp_path)
    return vector_store, len(chunks), len(reader.pages)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def load_existing_index():
    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

def ask_groq(question: str, context: str) -> str:
    prompt = f"""You are an intelligent enterprise document assistant.
Use ONLY the context below to answer the question accurately.
Mention the page number if available.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.set_page_config(
    page_title="Enterprise Doc Assistant",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/document.png", width=80)
    st.title("📁 Document Assistant")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload any PDF to chat with it"
    )

    if uploaded_file:
        with st.spinner("⏳ Indexing document..."):
            vector_store, num_chunks, num_pages = load_and_index_pdf(uploaded_file)
            st.session_state.vector_store = vector_store
            st.session_state.messages = []
        st.success(f"✅ Indexed {num_pages} pages, {num_chunks} chunks")
        st.info(f"📄 **File:** {uploaded_file.name}")

    elif os.path.exists("faiss_index/index.faiss"):
        if "vector_store" not in st.session_state:
            with st.spinner("Loading existing index..."):
                st.session_state.vector_store = load_existing_index()
            st.success("✅ Existing document loaded!")

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Ask any question")
    st.markdown("3. Get AI answers with sources")
    st.markdown("---")
    st.caption("Built with Groq + HuggingFace + FAISS")

# Main area
st.title("🤖 Enterprise Document Assistant")
st.caption("Chat with your documents using AI — powered by Groq LLM")

if "vector_store" not in st.session_state:
    st.info("👈 Please upload a PDF from the sidebar to get started!")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask anything about your document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = st.session_state.vector_store.similarity_search(question, k=3)
            context = "\n\n".join([
                f"[Page {d.metadata.get('page', '?')}]: {d.page_content}"
                for d in docs
            ])
            answer = ask_groq(question, context)
            st.markdown(answer)

            with st.expander("📄 Source chunks used"):
                for i, doc in enumerate(docs):
                    page = doc.metadata.get('page', '?')
                    st.markdown(f"**Chunk {i+1} — Page {page}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})