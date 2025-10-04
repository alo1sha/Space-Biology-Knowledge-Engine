# ========================================
# Streamlit App with Persistent State (Fixed)
# ========================================

import streamlit as st
import pandas as pd
import pickle
import os
import re
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = "/content/Search Engine Data.csv"
PAPERS_PATH = "/content/papers.pkl"

df = pd.read_csv(DATA_PATH)
if os.path.exists(PAPERS_PATH):
    with open(PAPERS_PATH, "rb") as f:
        papers = pickle.load(f)
else:
    papers = {}
    st.warning("⚠️ papers.pkl not found — summaries and graphs will be unavailable.")

# -----------------------------
# Generate Interests
# -----------------------------
titles = " ".join(df["Title"].dropna().tolist()).lower()
words = re.findall(r"\b[a-z]{5,}\b", titles)
common = Counter(words).most_common(20)
auto_interests = [w.capitalize() for w, _ in common]

# -----------------------------
# App Layout
# -----------------------------
st.set_page_config(page_title="Research Paper Explorer", layout="wide")
st.title("🔍 Research Paper Explorer")
st.markdown("Explore papers, read summaries, ask questions, and visualize graphs.")

# -----------------------------
# Initialize session state
# -----------------------------
if "selected_interest" not in st.session_state:
    st.session_state.selected_interest = None
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "current_summary" not in st.session_state:
    st.session_state.current_summary = ""
if "current_graph" not in st.session_state:
    st.session_state.current_graph = None

# -----------------------------
# Search functions
# -----------------------------
OUTPUT_DIR = "/content/vector_store_output"
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_index = FAISS.load_local(OUTPUT_DIR, embedding_model, allow_dangerous_deserialization=True)

def paper_search(keyword, top_k=5):
    results = faiss_index.as_retriever(search_kwargs={"k": top_k}).get_relevant_documents(keyword)
    found = []
    for doc in results:
        found.append({
            "title": doc.metadata.get("title", "Untitled"),
            "source": doc.metadata.get("source", "Unknown"),
            "preview": doc.page_content[:300] + "..."
        })
    return found

def show_summary(paper_id_or_url):
    if paper_id_or_url in papers:
        return papers[paper_id_or_url]["summary"]
    return "No summary available for this paper."

def show_graph(paper_id_or_url):
    if paper_id_or_url not in papers:
        return None
    return papers[paper_id_or_url].get("graph_image")

def answer_question(paper_id_or_url, question):
    if paper_id_or_url not in papers:
        return "No data found for this paper."
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    QA_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_index.as_retriever(search_kwargs={"k": 3})
    )
    title = papers[paper_id_or_url]["title"]
    query = f"Based on the paper titled '{title}', answer: {question}"
    return QA_chain.run(query)

# -----------------------------
# Interface
# -----------------------------
st.session_state.selected_interest = st.selectbox(
    "🎯 Choose your research interest:", auto_interests, 
    index=auto_interests.index(st.session_state.selected_interest)
    if st.session_state.selected_interest in auto_interests else 0
)

if st.button("Search Papers"):
    st.session_state.search_results = paper_search(st.session_state.selected_interest, top_k=5)

if st.session_state.search_results:
    st.success(f"Found {len(st.session_state.search_results)} related papers.")
    for i, paper in enumerate(st.session_state.search_results):
        st.markdown(f"### 📄 {i+1}. {paper['title']}")
        st.write(f"🔗 [Open Paper Source]({paper['source']})")
        st.caption(paper['preview'])

        col1, col2, col3 = st.columns(3)

        # Summary
        with col1:
            if st.button(f"📝 Summary #{i+1}"):
                st.session_state.current_summary = show_summary(paper["source"])
        if st.session_state.current_summary:
            st.text_area("Summary", st.session_state.current_summary, height=250)

        # Question
        with col2:
            user_q = st.text_input(f"❓ Ask about paper #{i+1}")
            if st.button(f"Get Answer #{i+1}") and user_q:
                st.info("Answering... please wait ⏳")
                answer = answer_question(paper["source"], user_q)
                st.success(answer)

        # Graph
        with col3:
            if st.button(f"📊 Graph #{i+1}"):
                st.session_state.current_graph = show_graph(paper["source"])
        if st.session_state.current_graph and os.path.exists(st.session_state.current_graph):
            st.image(st.session_state.current_graph, caption="Entity Graph", use_container_width=True)
