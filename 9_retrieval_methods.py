
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Setup
persistent_directory = "db/chroma_db"

# Initialize local embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# Connect to existing ChromaDB vector store
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

# Query to test
query = "How much did Microsoft pay to acquire GitHub?"
# query = "How do you plant tomatoes in a garden?"

print(f"Query: {query}\n")


def print_documents(docs):
    """Display retrieved documents."""
    print(f"Retrieved {len(docs)} documents:\n")

    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:")
        print(f"{doc.page_content}\n")

    print("-" * 60)


# ──────────────────────────────────────────────────────────────────
# METHOD 1: Basic Similarity Search
# Returns the top k most similar documents
# ──────────────────────────────────────────────────────────────────

print("=== METHOD 1: Similarity Search (k=3) ===")

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

docs = retriever.invoke(query)
print_documents(docs)


# ──────────────────────────────────────────────────────────────────
# METHOD 2: Similarity with Score Threshold
# Returns documents that meet the similarity threshold
# ──────────────────────────────────────────────────────────────────

# print("\n=== METHOD 2: Similarity with Score Threshold ===")

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 3,
#         "score_threshold": 0.3
#     }
# )

# docs = retriever.invoke(query)
# print_documents(docs)


# ──────────────────────────────────────────────────────────────────
# METHOD 3: Maximum Marginal Relevance (MMR)
# Balances relevance and diversity to avoid redundant results
# ──────────────────────────────────────────────────────────────────

# print("\n=== METHOD 3: Maximum Marginal Relevance (MMR) ===")

# retriever = db.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 3,            # Final number of documents
#         "fetch_k": 10,     # Initial candidate pool
#         "lambda_mult": 0.5 # 0 = maximum diversity, 1 = maximum relevance
#     }
# )

# docs = retriever.invoke(query)
# print_documents(docs)


print("=" * 60)
print("Done! Try different queries or parameters to compare retrieval methods.")
