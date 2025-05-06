import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF for PDF reading

# --- Caching Functions ---
@st.cache_data
def load_catalog():
    return pd.read_csv("product_catalog.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Indexing and Retrieval ---
def build_index(texts, model):
    embeddings = model.encode(texts, convert_to_tensor=False)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def retrieve_with_scores(query, texts, model, index, embeddings, k=5):
    query_embedding = model.encode([query])
    scores, indices = index.search(np.array(query_embedding).astype('float32'), k)
    return [(texts[i], round(1 - (scores[0][j] / 4), 2)) for j, i in enumerate(indices[0])]

# --- PDF Resume Reader ---
def extract_resume_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# --- Prompt Generator ---
def format_prompt(job_role, job_level, job_description):
    return f"""
You are an expert SHL assessment recommender.

Given the following:
- Job Title: {job_role}
- Job Level: {job_level}
- Job Description: {job_description}

Return the most relevant SHL assessments from the catalog.
"""

# --- UI ---
st.set_page_config(page_title="SHL Assessment Recommender", page_icon="🧠")
st.title("🧠 SHL GenAI Assessment Recommender")
st.markdown("Get personalized SHL assessment suggestions based on your job role or resume.")

# Load model and data
catalog_df = load_catalog()
model = load_model()
product_texts = (catalog_df['Product Name'] + " - " + catalog_df['Description']).tolist()
index, embeddings = build_index(product_texts, model)

# Input Fields
job_role = st.text_input("Job Title", placeholder="e.g., Software Developer")
job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior", "Executive"])
job_description = st.text_area("Job Description", placeholder="Paste job description here...")

uploaded_resume = st.file_uploader("📄 Or upload a resume (PDF)", type=["pdf"])
if uploaded_resume:
    job_description = extract_resume_text(uploaded_resume)
    st.text_area("Extracted Resume Text", job_description, height=150)

# Recommendation Trigger
if (job_description or job_role) and st.button("🔍 Recommend Assessments"):
    query = format_prompt(job_role, job_level, job_description)
    st.info("Searching SHL catalog for relevant assessments...")
    results = retrieve_with_scores(query, product_texts, model, index, embeddings)

    st.subheader("✅ Top SHL Assessment Matches:")
    for res, score in results:
        name, desc = res.split(" - ", 1)
        st.markdown(f"**{name}**  \n*{desc}*  \n_Relevance Score: {score}_\n")
