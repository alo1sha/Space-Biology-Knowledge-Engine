# ========================================
# 🪐 Streamlit App: Space Biology Knowledge Engine (Final)
# ========================================

import streamlit as st
import pandas as pd
import pickle
import os
import re
import zipfile
from collections import Counter

# -----------------------------
# 0. Extract graph images (if zipped)
# -----------------------------
if os.path.exists("graph_images_zip.zip"):
    with zipfile.ZipFile("graph_images_zip.zip", 'r') as zip_ref:
        zip_ref.extractall("graph_images")

# -----------------------------
# 1. Load data
# -----------------------------
DATA_PATH = "Search Engine Data.csv"      # main CSV
PAPERS_PATH = "papers.pkl"                # summaries + graphs

# Load CSV
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.error("❌ CSV file not found. Please ensure `Search Engine Data.csv` is uploaded.")
    st.stop()

# Load papers dictionary (summaries + graphs)
if os.path.exists(PAPERS_PATH):
    with open(PAPERS_PATH, "rb") as f:
        papers = pickle.load(f)
else:
    papers = {}
    st.warning("⚠️ papers.pkl not found — summaries and graphs will be unavailable.")

# -----------------------------
# 2. Generate Smart Interests
# -----------------------------
titles = " ".join(df["Title"].dropna().tolist()).lower()
words = re.findall(r"\b[a-z]{5,}\b", titles)
stopwords = {"study", "effect", "based", "using", "analysis", "results", "data", "system", "method", "approach", "paper"}
filtered = [w for w in words if w not in stopwords]
common = Counter(filtered).most_common(15)
auto_interests = [w.capitalize() for w, _ in common]

# -----------------------------
# 3. Page setup
# -----------------------------
st.set_page_config(page_title="🪐 Space Biology Knowledge Engine", layout="wide")
st.title("🪐 Space Biology Knowledge Engine")
st.markdown("Explore research papers by your **interest**, read summaries, ask questions, and visualize entity graphs.")

# -----------------------------
# 4. Interest selection
# -----------------------------
selected_interest = st.selectbox("🎯 Choose your research interest:", auto_interests)

# -----------------------------
# 5. Search Papers
# -----------------------------
if st.button("🚀 Search Papers"):
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    OUTPUT_DIR = "vector_store_output"
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_index = FAISS.load_local(OUTPUT_DIR, embedding_model, allow_dangerous_deserialization=True)

    # -----------------------------
    # Helper Functions
    # -----------------------------
    def paper_search(keyword, top_k=5):
        results = faiss_index.as_retriever(search_kwargs={"k": top_k * 3}).get_relevant_documents(keyword)
        found = []
        seen_sources = set()
        for doc in results:
            source = doc.metadata.get("source", "Unknown")
            if source in seen_sources:
                continue
            seen_sources.add(source)
            found.append({
                "title": doc.metadata.get("title", "Untitled"),
                "source": source,
                "preview": doc.page_content[:300] + "..."
            })
            if len(found) >= top_k:
                break
        return found

    def show_summary(paper_id_or_url, papers_dict):
        if paper_id_or_url in papers_dict:
            return papers_dict[paper_id_or_url]["summary"]
        return f"No summary available for {paper_id_or_url}"

    def show_graph(paper_id_or_url, papers_dict):
        if paper_id_or_url not in papers_dict:
            return None

        graph_path = papers_dict[paper_id_or_url].get("graph_image")
        if graph_path and not os.path.exists(graph_path):
            filename = os.path.basename(graph_path)
            alt_path = os.path.join("graph_images", filename)
            if os.path.exists(alt_path):
                graph_path = alt_path
            else:
                return None
        return graph_path

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
    # Execute search
    # -----------------------------
    results = paper_search(selected_interest, top_k=5)

    if not results:
        st.warning("No matching papers found for this interest.")
    else:
        st.success(f"✅ Found {len(results)} related papers.")
        for i, paper in enumerate(results):
            st.markdown(f"### 📄 {i+1}. {paper['title']}")
            st.write(f"🔗 [Open Paper Source]({paper['source']})")
            st.caption(paper['preview'])

            # UI columns
            col1, col2, col3 = st.columns(3)

            # 📝 Summary
            with col1:
                if st.button(f"📝 Summary #{i+1}"):
                    summary = show_summary(paper["source"], papers)
                    st.text_area("Summary", summary, height=250)

            # ❓ Question
            with col2:
                user_q = st.text_input(f"❓ Ask about paper #{i+1}", key=f"q{i}")
                if user_q:
                    st.info("Answering... please wait ⏳")
                    answer = answer_question(paper["source"], user_q, papers)
                    st.success(answer)

            # 📊 Graph
            with col3:
                if st.button(f"📊 Graph #{i+1}"):
                    graph_path = show_graph(paper["source"], papers)
                    if graph_path and os.path.exists(graph_path):
                        st.image(graph_path, caption="Entity Graph", use_container_width=True)
                    else:
                        st.warning("No graph available for this paper.")
