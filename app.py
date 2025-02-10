import os
import json
import faiss
import numpy as np
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from sentence_transformers import SentenceTransformer

# Custom CSS Styling
st.markdown("""
<style>
    .main { background-color: #1a1a1a; color: #ffffff; }
    .sidebar .sidebar-content { background-color: #2d2d2d; }
    .stTextInput textarea { color: #ffffff !important; }
    .stSelectbox div[data-baseweb="select"] { color: white !important; background-color: #3d3d3d !important; }
</style>
""", unsafe_allow_html=True)

st.title("👶 Therapy Bot: Your Parenting Guide 🍼")
st.caption("🎭 The Ultimate AI Therapist for Sleep-Deprived Parents!")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Choose AI Model", ["deepseek-r1:7b", "deepseek-r1:1.5b"], index=0)
    st.divider()
    st.markdown("### 🤹‍♂️ What Therapy Bot Can Do?")
    st.markdown("""
    - 🍼 Sleep Training Advice
    - 🍎 Healthy Baby Food Tips
    - 😰 Parenting Anxiety Support
    - 🎨 Fun Activities for Kids
    """)
    st.divider()
    st.markdown("Built with 💙 [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Load FAISS Index and Metadata
faiss_index_path = "faiss_index.bin"
metadata_path = "faiss_metadata.json"
pdf_texts_path = "pdf_texts.json"

if os.path.exists(faiss_index_path) and os.path.exists(metadata_path) and os.path.exists(pdf_texts_path):
    index = faiss.read_index(faiss_index_path)
    with open(metadata_path, "r") as f:
        doc_names = json.load(f)
    with open(pdf_texts_path, "r") as f:
        pdf_texts = json.load(f)
else:
    st.error("❌ FAISS index or metadata files not found. Please run the indexing script first.")
    st.stop()

# Load Sentence Transformer for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_articles(query, top_k=3):
    """Finds the most relevant articles based on user query."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(doc_names[idx])  # Get matching document names
    
    return results

# Initiate AI Engine
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

def build_prompt_chain(user_query):
    """Builds the AI prompt chain with retrieved context from FAISS."""
    
    # Retrieve relevant articles
    related_articles = search_articles(user_query, top_k=3)
    retrieved_content = "\n\n".join([pdf_texts[doc] for doc in related_articles])

    # AI System Prompt with context from FAISS
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are Therapy Bot, an expert AI parenting assistant. "
        "Provide warm, empathetic, and fun responses with helpful tips for parents. "
        "Use the following relevant information from parenting research papers:\n\n"
        f"{retrieved_content[:1500]}"  # Limit to avoid token overflow
    )

    # Create Chat Prompt Chain
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))

    return ChatPromptTemplate.from_messages(prompt_sequence)

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

# Session State Management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "👋 Hey there, Super Parent! How can I help you today? 🍼"}]

# Chat Container
chat_container = st.container()

# Display Chat Messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat Input & Processing
user_query = st.chat_input("Ask your parenting question here...👶")

if user_query:
    # Add User Message
    st.session_state.message_log.append({"role": "user", "content": user_query})

    # Generate AI Response
    with st.spinner("🎭 Thinking..."):
        prompt_chain = build_prompt_chain(user_query)
        ai_response = generate_ai_response(prompt_chain)

    # Add AI Response
    st.session_state.message_log.append({"role": "ai", "content": ai_response})

    # Rerun to Update Chat
    st.rerun()
