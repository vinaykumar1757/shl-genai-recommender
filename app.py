# app.py
import streamlit as st
import pandas as pd
import os
import openai
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader

st.set_page_config(page_title="SHL GenAI Assessment Recommender")
st.title("🔍 SHL GenAI Assessment Recommender")

openai.api_key = os.getenv("59f8653cb6cb4b708a176c0b9bcf0ecd")

# Load product catalog
loader = CSVLoader(file_path="product_catalog.csv")
docs = loader.load()

# Split text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(docs)

# Create embeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# User input
user_query = st.text_input("Enter the skill or requirement (e.g., leadership, coding, communication):")

if user_query:
    with st.spinner("Generating recommendation..."):
        result = qa.run(user_query)
        st.success("Recommended Assessment:")
        st.write(result)

