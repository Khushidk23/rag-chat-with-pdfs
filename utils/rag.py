from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import tempfile
import os

def create_vectorstore_and_retriever(documents):
    embeddings = OpenAIEmbeddings()
    persist_directory = os.path.join(os.getcwd(), "db")  # a folder named 'db' will be used
    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_directory)
    return vectordb.as_retriever()