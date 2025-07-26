import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from utils.document import load_and_split_pdf
from utils.rag import create_vectorstore_and_retriever

load_dotenv()

st.set_page_config(page_title="Chat with Your PDFs", layout="wide")
st.title("📄 Chat with Your Documents")

uploaded_files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = load_and_split_pdf(temp_path)
        all_docs.extend(docs)

        # Optional cleanup
        os.remove(temp_path)

    # Create vector store retriever
    retriever = create_vectorstore_and_retriever(all_docs)

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # User asks a question
    query = st.text_input("💬 Ask a question based on the uploaded PDF(s):")

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain(query)

        st.subheader("Answer")
        st.write(result['result'])

        st.subheader("Sources")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Page {i+1}:** {doc.metadata.get('page', 'Unknown')}")
            st.markdown(doc.page_content[:300] + "...")
