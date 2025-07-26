import shutil, os

# Delete Chroma directory if it exists
if os.path.exists(".chroma"):
    shutil.rmtree(".chroma")

# Delete FAISS index file if it exists
if os.path.exists("faiss_index.pkl"):
    os.remove("faiss_index.pkl")

print("Deleted .chroma directory and faiss_index.pkl file (if they existed).")
