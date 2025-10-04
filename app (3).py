# ========================================
#  Space Biology Knowledge Engine
# ========================================

import streamlit as st
import pandas as pd
import pickle
import os
import re
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = "Search Engine Data.csv"
PAPERS_PATH = "papers.pkl"

# Load CSV
df = pd.read_csv(DATA_PATH)

# Load papers dictionary
if os.path.exists(PAPERS_PATH):
    with open(PAPERS_PATH, "rb") as f:
        papers = pickle.load(f)
else:
    papers = {}
    st.warning("⚠️ papers.pkl not found — summaries and graphs may be unavailable.")

# -----------------------------
# Auto-generate Interests from Titles
# -----------------------------
titles = " ".join(df["Title"].dropna().tolist()).lower()
words = re.findall(r"\b[a-z]{5,}\b", titles)
common = Counter(words).most_common(20)
auto_interests = [w.capitalize() for w, _ in common]

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="🚀 Space Biology Knowledge Engine", layout="wide")
st.title("🧬 Space Biology Knowledge Engine")
st.markdown("Explore **space-related research papers**: choose your interest, read summaries, ask questions, and view entity graphs 🌌")

# -----------------------------
# Select interest
# -----------------------------
selected_interest = st.selectbox("🌍 Choose your research interest:", auto_interests)

# -----------------------------
# Load FAISS index
# -----------------------------
OUTPUT_DIR = "vector_store_output"
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_index = FAISS.load_local(OUTPUT_DIR, embedding_model, allow_dangerous_deserialization=True)

# -----------------------------
# Helper functions
# -----------------------------
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

def show_summary(paper_id_or_url, papers_dict):
    if paper_id_or_url in papers_dict:
        return papers_dict[paper_id_or_url]["summary"]
    return f"No summary available for {paper_id_or_url}"

def show_graph(paper_id_or_url, papers_dict):
    if paper_id_or_url not in papers_dict:
        return None
    return papers_dict[paper_id_or_url].get("graph_image")

def answer_question(paper_id_or_url, question, papers_dict):
    if paper_id_or_url not in papers_dict:
        return f"No data found for {paper_id_or_url}"

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

    title = papers_dict[paper_id_or_url]["title"]
    query = f"Based on the paper titled '{title}', answer: {question}"
    return QA_chain.run(query)

# -----------------------------
# Search button
# -----------------------------
if st.button("🔍 Search Papers"):
    results = paper_search(selected_interest, top_k=5)

    if not results:
        st.warning("No matching papers found for this interest.")
    else:
        st.success(f"Found {len(results)} related papers 🛰️")
        for i, paper in enumerate(results):
            st.markdown(f"### 📄 {i+1}. {paper['title']}")
            st.write(f"🔗 [Open Paper Source]({paper['source']})")
            st.caption(paper['preview'])

            col1, col2, col3 = st.columns(3)

            # Summary
            with col1:
                if st.button(f"🧾 Summary #{i+1}"):
                    summary = show_summary(paper["source"], papers)
                    st.text_area("Summary", summary, height=250)

            # Q&A
            with col2:
                user_q = st.text_input(f"❓ Ask about paper #{i+1}")
                if user_q:
                    st.info("Answering... please wait ⏳")
                    answer = answer_question(paper["source"], user_q, papers)
                    st.success(answer)

            # Graph
            with col3:
                if st.button(f"🪐 Graph #{i+1}"):
                    graph_path = show_graph(paper["source"], papers)
                    if graph_path and os.path.exists(graph_path):
                        st.image(graph_path, caption="Entity Graph", use_container_width=True)
                    else:
                        st.warning("No graph available for this paper.")
