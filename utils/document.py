import re
import hashlib
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Normalize and hash page content
def normalize_and_hash(text):
    # Remove excessive whitespace, lowercase everything
    text = ' '.join(text.lower().split())
    # Remove all non-alphanumeric characters (optional but useful)
    text = re.sub(r'\W+', '', text)
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# Better deduplication
def deduplicate_docs(documents):
    seen = set()
    unique_docs = []
    for doc in documents:
        norm_hash = normalize_and_hash(doc.page_content)
        if norm_hash not in seen:
            seen.add(norm_hash)
            unique_docs.append(doc)
    return unique_docs

def load_and_split_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # Normalize content before deduplication
    for doc in documents:
        doc.page_content = ' '.join(doc.page_content.split())

    documents = deduplicate_docs(documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)
