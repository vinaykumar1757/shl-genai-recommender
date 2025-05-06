import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import fitz


@st.cache_data
def load_catalog():
    return pd.read_csv("product_catalog.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def build_index(texts, model):
    embeddings = model.encode(texts, convert_to_tensor=False)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def retrieve_with_scores(query, texts, model, index, embeddings, k=5):
    query_embedding = model.encode([query])
    scores, indices = index.search(np.array(query_embedding).astype('float32'), k)
    return [(texts[i], round(1 - (scores[0][j] / 4), 2)) for j, i in enumerate(indices[0])] 

def extract_resume_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def format_prompt(job_role, job_level, job_description):
    return f"""
You are an expert assessment advisor at SHL. Recommend the most suitable assessments based on the following:

- Job Title: {job_role}
- Job Level: {job_level}
- Job Description:\n{job_description}

Return a list of relevant SHL assessments with brief justifications.
"""

st.title("🔍 SHL Assessment Recommendation Engine (GenAI Enhanced)")
st.markdown("Get personalized SHL assessment recommendations based on your role or resume.")

catalog_df = load_catalog()
model = load_model()
product_texts = (catalog_df['Product Name'] + " - " + catalog_df['Description']).tolist()
index, embeddings = build_index(product_texts, model)

job_role = st.text_input("Job Title", placeholder="e.g., Data Analyst")
job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior", "Executive"])
job_description = st.text_area("Job Description", placeholder="Paste the job description here...")

uploaded_resume = st.file_uploader("📄 Or upload a resume (PDF)", type=["pdf"])
if uploaded_resume:
    job_description = extract_resume_text(uploaded_resume)
    st.text_area("Extracted Resume Text", job_description, height=150)

if (job_description or job_role) and st.button("Recommend Assessments"):
    query = format_prompt(job_role, job_level, job_description)
    st.info("🔍 Searching SHL catalog for best matches...")
    results = retrieve_with_scores(query, product_texts, model, index, embeddings)

    st.subheader("✅ Recommended SHL Assessments:")
    for res, score in results:
        name, desc = res.split(" - ", 1)
        st.markdown(f"**{name}** (Relevance: {score})\n\n*{desc}*")

