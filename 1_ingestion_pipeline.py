
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory."""

    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. "
            "Please create it and add your company files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}. "
            "Please add your company documents."
        )

    print(f"Total documents loaded: {len(documents)}")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks with overlap."""

    print("\nSplitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Content:\n{chunk.page_content}")
        print("-" * 50)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_embedding_model():
    """Initialize HuggingFace embedding model locally."""

    print("\nLoading HuggingFace embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print("HuggingFace embedding model loaded successfully.")

    return embedding_model


def create_vector_store(
    chunks,
    embedding_model,
    persist_directory="db/chroma_db"
):
    """Generate embeddings and store them in ChromaDB."""

    print("\nCreating embeddings and storing in ChromaDB...")

    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("\nVector store created successfully!")
    print(f"Persisted at: {persist_directory}")
    print(f"Total stored chunks: {vectorstore._collection.count()}")

    return vectorstore


def main():
    """Main document ingestion pipeline."""

    print("=== RAG Document Ingestion Pipeline ===\n")

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    # Initialize embedding model
    embedding_model = create_embedding_model()

    # Check whether vector store already exists
    if os.path.exists(persistent_directory):

        print("Vector store directory already exists.")

        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model
        )

        document_count = vectorstore._collection.count()

        if document_count > 0:
            print(
                f"Existing vector store loaded successfully. "
                f"Total chunks: {document_count}"
            )
            return vectorstore

        print("Existing vector store is empty. Starting ingestion...")

    else:
        print("Vector store does not exist. Initializing...\n")

    # Step 1: Load documents
    documents = load_documents(docs_path)

    # Step 2: Split documents into chunks
    chunks = split_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200
    )

    # Step 3: Generate embeddings and store in ChromaDB
    vectorstore = create_vector_store(
        chunks,
        embedding_model,
        persistent_directory
    )

    print("\nIngestion complete!")
    print("Your documents are now ready for RAG queries.")

    return vectorstore


if __name__ == "__main__":
    main()
