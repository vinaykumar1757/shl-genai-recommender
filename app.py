import streamlit as st
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

st.set_page_config(page_title="SHL GenAI Assessment Recommender")
st.title("SHL Assessment Recommendation Engine")

@st.cache_resource
def load_recommender():
    df = pd.read_csv("product_catalog.csv")
    df["combined"] = df["Name"] + ". " + df["Description"] + " Use case: " + df["Use Case"]

    loader = DataFrameLoader(df[["combined"]], page_content_column="combined")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = OpenAI(temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

qa_chain = load_recommender()

skill = st.text_input("Enter the skills or problem statement you're hiring for:")
if st.button("Get Recommendations") and skill:
    with st.spinner("Finding best-fit assessments..."):
        result = qa_chain.run(f"Given this skill or need: '{skill}', which SHL assessments should I use? Return only the most relevant ones with names and 1-line descriptions.")
        st.success("Recommendation:")
        st.write(result)
