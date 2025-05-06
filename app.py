import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import faiss

# Load the product catalog
@st.cache_data
def load_catalog():
    return pd.read_csv("product_catalog.csv")

# Load the embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Build FAISS index
def build_index(texts, model):
    embeddings = model.encode(texts, convert_to_tensor=False)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index, embeddings

# Retrieve top k matches
def retrieve(query, texts, model, index, embeddings, k=5):
    query_embedding = model.encode([query])
    _, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Streamlit App
st.title("SHL Assessment Recommendation Engine")
st.markdown("Enter your job role or hiring requirement, and we'll recommend the most relevant SHL assessments.")

# Load data
catalog_df = load_catalog()
model = load_model()

product_texts = (catalog_df['Product Name'] + " - " + catalog_df['Description']).tolist()
index, embeddings = build_index(product_texts, model)

# Input box
query = st.text_input("Enter a job role or skill requirement:", placeholder="e.g., Software Engineer with Python")

if query:
    st.write("🔎 Fetching relevant assessments...")
    results = retrieve(query, product_texts, model, index, embeddings)
    
    st.subheader("Recommended SHL Assessments:")
    for res in results:
        st.markdown(f"- {res}")
