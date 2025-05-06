import streamlit as st
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DataFrameLoader
import os

st.set_page_config(page_title="🔍 SHL GenAI Assessment Recommender", layout="wide")

st.title("🔍 SHL GenAI Assessment Recommender")
st.markdown("This app uses Generative AI to recommend SHL assessments based on the skills you provide.")

# Load the product catalog
@st.cache_data
def load_data():
    df = pd.read_csv("product_catalog.csv")
    df = df[["Assessment Name", "Skills"]]
    df = df.dropna()
    return df

df = load_data()

# Initialize embeddings and vector store
@st.cache_resource
def create_qa_chain(df):
    # Check and fetch OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your environment or Streamlit secrets.")
        st.stop()

    # Convert DataFrame into documents
    loader = DataFrameLoader(df, page_content_column="Skills")
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    return qa

qa = create_qa_chain(df)

# User input
skills = st.text_input("💡 Enter a list of skills (comma-separated):")

if st.button("Get Recommendation"):
    if skills.strip() == "":
        st.warning("Please enter some skills to get recommendations.")
    else:
        with st.spinner("Generating recommendations..."):
            query = f"Suggest best SHL assessments from the catalog based on the following skills: {skills}"
            result = qa.run(query)
            st.subheader("🔎 Recommended SHL Assessments:")
            st.write(result)


