# ========================================
# Streamlit App: Space Biology Knowledge Engine
# ========================================

import streamlit as st
import pandas as pd
import pickle
import os
import re
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================
# Setup paths
# ==============================
DATA_PATH = "Search Engine Data.csv"
PAPERS_PATH = "papers.pkl"
OUTPUT_DIR = "vector_store_output"

# ==============================
# Load data
# ==============================
if not os.path.exists(DATA_PATH):
    st.error("❌ Missing dataset file: Search Engine Data.csv — upload it to the repo.")
    st.stop()

df = pd.read_csv(DATA_PATH)

if os.path.exists(PAPERS_PATH):
    with open(PAPERS_PATH, "rb") as f:
        papers = pickle.load(f)
else:
    papers = {}
    st.warning("⚠️ papers.pkl not found — summaries & graphs will be unavailable.")

# ==============================
# Generate interests from titles
# ==============================
titles = " ".join(df["Title"].dropna().tolist()).lower()
words = re.findall(r"\b[a-z]{5,}\b", titles)
auto_interests = [w.capitalize() for w, _ in Counter(words).most_common(20)]

# ==============================
# Streamlit page setup
# ==============================
st.set_page_config(page_title="Research Paper Explorer", layout="wide")
st.title("🔍 Research Paper Explorer")
st.markdown("Explore research papers by your **interest**, read summaries, ask questions, and visualize entity graphs.")

# ==============================
# Persist state (to prevent reset)
# ==============================
if "results" not in st.session_state:
    st.session_state.results = []
if "selected_interest" not in st.session_state:
    st.session_state.selected_interest = None

# ==============================
# Select research interest
# ==============================
selected_interest = st.selectbox("🎯 Choose your research interest:", auto_interests)

if st.button("Search Papers"):
    st.session_state.selected_interest = selected_interest

if st.session_state.selected_interest:
    # Load FAISS index
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_index = FAISS.load_local(OUTPUT_DIR, embedding_model, allow_dangerous_deserialization=True)

    def paper_search(keyword, top_k=5):
        results = faiss_index.as_retriever(search_kwargs={"k": top_k}).get_relevant_documents(keyword)
        return [{
            "title": doc.metadata.get("title", "Untitled"),
            "source": doc.metadata.get("source", "Unknown"),
            "preview": doc.page_content[:300] + "..."
        } for doc in results]

    def show_summary(paper_id_or_url):
        return papers.get(paper_id_or_url, {}).get("summary", "No summary available.")

    def show_graph(paper_id_or_url):
        return papers.get(paper_id_or_url, {}).get("graph_image")

    def answer_question(paper_id_or_url, question):
        if not paper_id_or_url in papers:
            return "No data found for this paper."
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        llm = HuggingFacePipeline(pipeline=qa_pipeline)
        QA_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=faiss_index.as_retriever(search_kwargs={"k": 3})
        )
        title = papers[paper_id_or_url]["title"]
        query = f"Based on the paper titled '{title}', answer: {question}"
        return QA_chain.run(query)

    # Run search once and cache it
    if not st.session_state.results:
        st.session_state.results = paper_search(st.session_state.selected_interest, top_k=5)

    results = st.session_state.results

    if results:
        st.success(f"Found {len(results)} related papers for '{st.session_state.selected_interest}'.")

        for i, paper in enumerate(results):
            st.markdown(f"### 📄 {i+1}. {paper['title']}")
            st.write(f"🔗 [Open Paper Source]({paper['source']})")
            st.caption(paper['preview'])

            with st.expander("📝 Summary"):
                st.write(show_summary(paper["source"]))

            user_q = st.text_input(f"❓ Ask about paper #{i+1}", key=f"q{i}")
            if user_q:
                st.info("Generating answer, please wait...")
                answer = answer_question(paper["source"], user_q)
                st.success(answer)

            graph_path = show_graph(paper["source"])
            if graph_path and os.path.exists(graph_path):
                st.image(graph_path, caption="Entity Graph", use_container_width=True)
            st.divider()
    else:
        st.warning("No matching papers found.")

