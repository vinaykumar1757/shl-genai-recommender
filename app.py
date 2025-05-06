import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate

catalog_path = "product_catalog.csv"
df = pd.read_csv(catalog_path)

df["combined"] = df["Assessment Name"] + ". " + df["Description"] + ". " + df["Use Case"].fillna("")

temp_catalog = "temp_catalog.csv"
df[["combined"]].to_csv(temp_catalog, index=False, header=False)

loader = CSVLoader(file_path=temp_catalog)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

db = FAISS.from_documents(
    docs,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

db.save_local("faiss_index")

db = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever = db.as_retriever(search_kwargs={"k": 5})

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant at SHL. Your job is to recommend relevant assessments based on the given query.

    Context:
    {context}

    User Query:
    {question}

    Return up to 3 assessments in JSON format with fields:
    - name
    - pdf_link
    - description
    - reason_for_recommendation
    """
)

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.3),
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

query = "We need to assess a technical lead in Python and leadership skills"
response = qa_chain.run(query)

print(response)
