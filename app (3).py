# ========================================
# 🪐 Streamlit App: Space Biology Knowledge Engine (Persistent State)
# ========================================

import streamlit as st
import pandas as pd
import pickle
import os
import re
import zipfile
from collections import Counter

# -----------------------------
# 0. Extract graph images if zipped
# -----------------------------
if os.path.exists("graph_images_zip.zip"):
    with zipfile.ZipFile("graph_images_zip.zip", 'r') as zip_ref:
        zip_ref.extractall("graph_images")

# -----------------------------
# 1. Load data
# -----------------------------
DATA_PATH = "Search Engine Data.csv"
PAPERS_PATH = "papers.pkl"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.error("❌ CSV file not found. Please ensure `Search Engine Data.csv` is uploaded.")
    st.stop()

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
# 4. Initialize session state
# -----------------------------
if "search_done" not in st.session_state:
    st.session_state.search_done = False
if "results" not in st.session_state:
    st.session_state.results = []
if "selected_interest" not in st.session_state:
    st.session_state.selected_interest = None

# -----------------------------
# 5. Select Interest
# -----------------------------
selected_interest = st.selectbox("🎯 Choose your research interest:", auto_interests, index=0)

if st.button("🚀 Search Papers"):
    st.session_state.selected_interest = selected_interest

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    OUTPUT_DIR = "vector_store_output"
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_index = FAISS.load_local(OUTPUT_DIR, embedding_model, allow_dangerous_deserialization=True)

    def paper_search(keyword, top_k=5):
        results = faiss_index.as_retriever(search_kwargs={"k": top_k * 3}).get_relevant_documents(keyword)
        found, seen = [], set()
        for doc in results:
            source = doc.metadata.get("source", "Unknown")
            if source in seen:
                continue
            seen.add(source)
            found.append({
                "title": doc.metadata.get("title", "Untitled"),
                "source": source,
                "preview": doc.page_content[:300] + "..."
            })
            if len(found) >= top_k:
                break
        return found

    st.session_state.results = paper_search(selected_interest, top_k=5)
    st.session_state.search_done = True

# -----------------------------
# 6. Display Results (Persistent)
# -----------------------------
if st.session_state.search_done and st.session_state.results:
    results = st.session_state.results
    st.success(f"✅ Found {len(results)} related papers for '{st.session_state.selected_interest}'.")

    for i, paper in enumerate(results):
        st.markdown(f"### 📄 {i+1}. {paper['title']}")
        st.write(f"🔗 [Open Paper Source]({paper['source']})")
        st.caption(paper['preview'])

        col1, col2, col3 = st.columns(3)

        def show_summary(paper_id_or_url):
            if paper_id_or_url in papers:
                return papers[paper_id_or_url]["summary"]
            return f"No summary available for {paper_id_or_url}"

        def show_graph(paper_id_or_url):
            if paper_id_or_url not in papers:
                return None
            graph_path = papers[paper_id_or_url].get("graph_image")
            if graph_path and not os.path.exists(graph_path):
                filename = os.path.basename(graph_path)
                alt_path = os.path.join("graph_images", filename)
                if os.path.exists(alt_path):
                    graph_path = alt_path
                else:
                    return None
            return graph_path

        def answer_question(paper_id_or_url, question):
            if paper_id_or_url not in papers:
                return f"No data found for {paper_id_or_url}"
            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
            llm = HuggingFacePipeline(pipeline=qa_pipeline)
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            from langchain.chains import RetrievalQA
            OUTPUT_DIR = "vector_store_output"
            embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            faiss_index = FAISS.load_local(OUTPUT_DIR, embedding_model, allow_dangerous_deserialization=True)
            QA_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff",
                retriever=faiss_index.as_retriever(search_kwargs={"k": 3})
            )
            title = papers[paper_id_or_url]["title"]
            query = f"Based on the paper titled '{title}', answer: {question}"
            return QA_chain.run(query)

        # 📝 Summary
        with col1:
            if st.button(f"📝 Summary #{i+1}", key=f"sum_{i}"):
                st.text_area("Summary", show_summary(paper["source"]), height=250, key=f"sum_txt_{i}")

        # ❓ Question
        with col2:
            q = st.text_input(f"❓ Ask about paper #{i+1}", key=f"q_{i}")
            if q:
                st.info("Answering... please wait ⏳")
                ans = answer_question(paper["source"], q)
                st.success(ans)

        # 📊 Graph
        with col3:
            if st.button(f"📊 Graph #{i+1}", key=f"graph_{i}"):
                gpath = show_graph(paper["source"])
                if gpath and os.path.exists(gpath):
                    st.image(gpath, caption="Entity Graph", use_container_width=True)
                else:
                    st.warning("No graph available for this paper.")
