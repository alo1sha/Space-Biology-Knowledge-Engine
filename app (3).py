# ========================================
# Streamlit App: Space Biology Knowledge Engine
# ========================================

import streamlit as st
import pandas as pd
import pickle
import os
import re
import zipfile
from collections import Counter

# -----------------------------
# 1. Setup paths & unzip data
# -----------------------------
DATA_PATH = "Search Engine Data.csv"      # CSV data file in GitHub
PAPERS_PATH = "papers.pkl"                # Summaries + graph paths
VECTOR_STORE_PATH = "vector_store_output" # Folder for FAISS index
GRAPH_PATH = "graph_images"               # Folder for graphs

# Unzip vector store if needed
if os.path.exists("vector_store_output.zip") and not os.path.exists(VECTOR_STORE_PATH):
    with zipfile.ZipFile("vector_store_output.zip", 'r') as zip_ref:
        zip_ref.extractall(VECTOR_STORE_PATH)

# Unzip graphs if needed
if os.path.exists("graph_images.zip") and not os.path.exists(GRAPH_PATH):
    with zipfile.ZipFile("graph_images.zip", 'r') as zip_ref:
        zip_ref.extractall(GRAPH_PATH)

# -----------------------------
# 2. Load data
# -----------------------------
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# Load papers dictionary
if os.path.exists(PAPERS_PATH):
    with open(PAPERS_PATH, "rb") as f:
        papers = pickle.load(f)
else:
    papers = {}
    st.warning("⚠️ 'papers.pkl' not found — summaries and graphs will be unavailable.")

# -----------------------------
# 3. Auto-generate interests from titles
# -----------------------------
titles = " ".join(df["Title"].dropna().tolist()).lower()
words = re.findall(r"\b[a-z]{5,}\b", titles)
common = Counter(words).most_common(20)
auto_interests = [w.capitalize() for w, _ in common]

# -----------------------------
# 4. Page setup
# -----------------------------
st.set_page_config(page_title="Space Biology Knowledge Engine", layout="wide")
st.title("🧬 Space Biology Knowledge Engine")
st.markdown("Explore **NASA** research papers, get **summaries**, ask **questions**, and view **entity graphs** 🌌")

# -----------------------------
# 5. Select interest
# -----------------------------
selected_interest = st.selectbox("🎯 Choose your research interest:", auto_interests)

if st.button("Search Papers"):
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.llms import HuggingFacePipeline
    from langchain.chains import RetrievalQA
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    # -----------------------------
    # Load FAISS index
    # -----------------------------
    try:
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        faiss_index = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ Error loading FAISS index: {e}")
        st.stop()

    # -----------------------------
    # Helper functions
    # -----------------------------
    def paper_search(keyword, top_k=5):
        retriever = faiss_index.as_retriever(search_kwargs={"k": top_k})
        results = retriever.get_relevant_documents(keyword)
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
            return papers_dict[paper_id_or_url].get("summary", "No summary available.")
        return "No summary found."

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
        query = f"Based on the paper titled '{title}', answer this question: {question}"
        return QA_chain.run(query)

    # -----------------------------
    # Run the search
    # -----------------------------
    results = paper_search(selected_interest, top_k=5)

    if not results:
        st.warning("No matching papers found for this topic.")
    else:
        st.success(f"Found {len(results)} papers related to '{selected_interest}' 📚")

        for i, paper in enumerate(results):
            st.markdown(f"### 📄 {i+1}. {paper['title']}")
            st.write(f"🔗 [Open Paper Source]({paper['source']})")
            st.caption(paper['preview'])

            col1, col2, col3 = st.columns([1, 2, 1])

            # Show summary
            with col1:
                if st.button(f"📝 Summary #{i+1}"):
                    summary = show_summary(paper["source"], papers)
                    st.text_area("Summary", summary, height=250)

            # Ask question
            with col2:
                user_q = st.text_input(f"❓ Ask about paper #{i+1}")
                if user_q:
                    with st.spinner("Answering..."):
                        answer = answer_question(paper["source"], user_q, papers)
                        st.success(answer)

            # Show graph
            with col3:
                if st.button(f"📊 Graph #{i+1}"):
                    graph_path = show_graph(paper["source"], papers)
                    if graph_path and os.path.exists(graph_path):
                        st.im
