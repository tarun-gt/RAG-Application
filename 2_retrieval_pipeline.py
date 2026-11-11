
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

persistent_directory = "db/chroma_db"


def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )


def load_vector_store():
    embedding_model = create_embedding_model()

    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model
    )

    return db


def retrieve_documents(query, k=5):
    db = load_vector_store()

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    relevant_docs = retriever.invoke(query)

    print(f"\nUser Query: {query}")
    print("\n--- Retrieved Context ---")

    for i, doc in enumerate(relevant_docs, 1):
        print(f"\nDocument {i}:")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")

    return relevant_docs


if __name__ == "__main__":
    query = "How much did Microsoft pay to acquire GitHub?"
    retrieve_documents(query)
