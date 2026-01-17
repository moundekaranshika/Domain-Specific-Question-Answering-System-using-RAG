# paste ingest.py code here
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils import load_documents, split_documents

DATA_DIR = "data"
VECTOR_DB_PATH = "faiss_index"


def main():
    print("Loading documents...")
    documents = load_documents(DATA_DIR)

    print("Splitting documents...")
    docs = split_documents(documents)

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)

    print("Ingestion complete. Vector store saved.")


if __name__ == "__main__":
    main()
