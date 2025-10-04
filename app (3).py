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

import zipfile

if os.path.exists("graph_images.zip"):
    with zipfile.ZipFile("graph_images.zip", 'r') as zip_ref:
        zip_ref.extractall("graph_images")

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = "Search Engine Data.csv"
PAPERS_PATH = "papers.pkl"

df = pd.read_csv(DATA_PATH)

if os.path.exists(PAPERS_PATH):
    with open(PAPERS_PATH, "rb") as f:
        papers = pickle.load(f)
else:
    papers = {}
    st.warning("⚠️ papers.pkl not found — summaries and graphs may be unavailable.")

# -----------------------------
# Generate Interests
# -----------------------------
titles = " ".join(df["Title"].dropna().tolist()).lower()
words = re.findall(r"\b[a-z]{5,}\b", titles)
auto_interests = [w.capitalize() for w, _ in Counter(words).most_common(20)]

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="🚀 Space Biology Knowledge Engine", layout="wide")
st.title(" Space Biology Knowledge Engine")
st.markdown("Explore **space-related research papers**: choose your interest, read summaries, ask questions, and view entity graphs 🌌")

# -----------------------------
# Interest selection
# -----------------------------
selected_interest = st.selectbox("🌍 Choose your research interest:", auto_interests)

# Keep state across reruns
if "search_results" not in st.session_state:
    st.session_state.search_results = None

if st.button("🔍 Search Papers"):
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_index = FAISS.load_local("vector_store_output", embedding_model, allow_dangerous_deserialization=True)

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

    st.session_state.search_results = paper_search(selected_interest, top_k=5)

# -----------------------------
# Display results if already loaded
# -----------------------------
if st.session_state.search_results:
    results = st.session_state.search_results
    st.success(f"Found {len(results)} related papers 🛰️")

    for i, paper in enumerate(results):
        st.markdown(f"### 📄 {i+1}. {paper['title']}")
        st.write(f"🔗 [Open Paper Source]({paper['source']})")
        st.caption(paper['preview'])

        # Unique keys per paper to prevent reset
        summary_key = f"summary_{i}"
        question_key = f"question_{i}"
        graph_key = f"graph_{i}"

        col1, col2, col3 = st.columns(3)

        #  Summary button
        with col1:
            if st.button(f"🧾 Summary #{i+1}", key=summary_key):
                summary = papers.get(paper["source"], {}).get("summary", "No summary available.")
                st.text_area("Summary", summary, height=250, key=f"summary_area_{i}")

        #  Question input
        with col2:
            user_q = st.text_input(f" Ask about paper❓ #{i+1}", key=question_key)
            if user_q:
                model_name = "google/flan-t5-base"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
                llm = HuggingFacePipeline(pipeline=qa_pipeline)

                QA_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=FAISS.load_local("vector_store_output", SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 3})
                )

                title = paper["title"]
                query = f"Based on the paper titled '{title}', answer: {user_q}"
                with st.spinner("Answering... please wait ⏳"):
                    answer = QA_chain.run(query)
                    st.success(answer)

        #  Graph
        with col3:
            if st.button(f"🪐 Graph #{i+1}", key=graph_key):
                graph_path = papers.get(paper["source"], {}).get("graph_image")
                if graph_path and os.path.exists(graph_path):
                    st.image(graph_path, caption="Entity Graph", use_container_width=True)
                else:
                    st.warning("No graph available for this paper.")
