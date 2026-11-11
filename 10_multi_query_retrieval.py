
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List

# ──────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────

persistent_directory = "db/chroma_db"

# Local embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# Local LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Connect to existing ChromaDB
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)


# ──────────────────────────────────────────────────────────────────
# PYDANTIC MODEL FOR STRUCTURED OUTPUT
# ──────────────────────────────────────────────────────────────────

class QueryVariations(BaseModel):
    queries: List[str] = Field(
        description="Three alternative versions of the original search query"
    )


# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Tesla make money?"

print(f"Original Query: {original_query}\n")


# ──────────────────────────────────────────────────────────────────
# STEP 1: GENERATE MULTIPLE QUERY VARIATIONS
# ──────────────────────────────────────────────────────────────────

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""
Generate exactly 3 different variations of this query
that would help retrieve relevant documents.

Original query: {original_query}

Return 3 alternative queries that rephrase or approach
the same question from different angles.
"""

response = llm_with_tools.invoke(prompt)

query_variations = response.queries

print("Generated Query Variations:")

for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "=" * 60)


# ──────────────────────────────────────────────────────────────────
# STEP 2: SEARCH WITH EACH QUERY VARIATION
# ──────────────────────────────────────────────────────────────────

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

all_retrieval_results = []

for i, query in enumerate(query_variations, 1):

    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")

    docs = retriever.invoke(query)

    # Store results for future RRF calculation
    all_retrieval_results.append(docs)

    print(f"Retrieved {len(docs)} documents:\n")

    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")

    print("-" * 50)


print("\n" + "=" * 60)
print("Multi-Query Retrieval Complete!")


# ──────────────────────────────────────────────────────────────────
# RETRIEVAL RESULTS STRUCTURE
# ──────────────────────────────────────────────────────────────────

# all_retrieval_results = [
#     [Doc1, Doc2, Doc3, Doc4, Doc5],  # Query 1 results
#     [Doc2, Doc1, Doc6, Doc7, Doc3],  # Query 2 results
#     [Doc8, Doc2, Doc9, Doc10, Doc11] # Query 3 results
# ]
