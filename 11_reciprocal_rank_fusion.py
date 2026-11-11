
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List
from collections import defaultdict


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
        description="Three alternative search queries"
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

llm_with_tools = llm.with_structured_output(
    QueryVariations,
    method="json_schema"
)

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

    # Store results for RRF calculation
    all_retrieval_results.append(docs)

    print(f"Retrieved {len(docs)} documents:\n")

    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")

    print("-" * 50)


print("\n" + "=" * 60)
print("Multi-Query Retrieval Complete!")
print("Notice how different query variations retrieved different documents.")


# ──────────────────────────────────────────────────────────────────
# STEP 3: APPLY RECIPROCAL RANK FUSION (RRF)
# ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):
    """
    Combine multiple ranked document lists using Reciprocal Rank Fusion.

    RRF Score = Sum of 1 / (k + rank)

    Documents appearing in multiple result lists receive higher scores.
    """

    if verbose:
        print("\n" + "=" * 60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("=" * 60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")

    rrf_scores = defaultdict(float)
    all_unique_chunks = {}

    chunk_id_map = {}
    chunk_counter = 1

    # Process each query's retrieval results
    for query_idx, chunks in enumerate(chunk_lists, 1):

        if verbose:
            print(f"Processing Query {query_idx} results:")

        for position, chunk in enumerate(chunks, 1):

            chunk_content = chunk.page_content

            # Assign unique chunk ID
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1

            chunk_id = chunk_id_map[chunk_content]

            # Store actual document object
            all_unique_chunks[chunk_content] = chunk

            # Calculate reciprocal rank score
            position_score = 1 / (k + position)

            # Accumulate RRF score
            rrf_scores[chunk_content] += position_score

            if verbose:
                print(
                    f"  Position {position}: {chunk_id} "
                    f"+{position_score:.4f} "
                    f"(running total: {rrf_scores[chunk_content]:.4f})"
                )

                print(f"    Preview: {chunk_content[:80]}...")

        if verbose:
            print()

    # Sort documents by RRF score
    sorted_chunks = sorted(
        [
            (all_unique_chunks[content], score)
            for content, score in rrf_scores.items()
        ],
        key=lambda x: x[1],
        reverse=True
    )

    if verbose:
        print(
            f"RRF Complete! Processed {len(sorted_chunks)} "
            f"unique chunks from {len(chunk_lists)} queries."
        )

    return sorted_chunks


# Apply RRF
fused_results = reciprocal_rank_fusion(
    all_retrieval_results,
    k=60,
    verbose=True
)


# ──────────────────────────────────────────────────────────────────
# STEP 4: DISPLAY FINAL FUSED RESULTS
# ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FINAL RRF RANKING")
print("=" * 60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):

    print(f"RANK {rank} (RRF Score: {rrf_score:.4f})")

    print(f"{doc.page_content[:200]}...")

    print("-" * 50)


print(
    f"\nRRF Complete! Fused {len(fused_results)} unique documents "
    f"from {len(query_variations)} query variations."
)

print("\nKey benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score")
print("   • Balanced fusion using k=60 for gentle position penalties")


# ──────────────────────────────────────────────────────────────────
# STEP 5: USAGE EXAMPLES
# ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("USAGE EXAMPLES")
print("=" * 60)

print("Original Query:", original_query)
print("Generated Queries:", len(query_variations))
print("Retrieved Document Lists:", len(all_retrieval_results))
print("Unique Documents After RRF:", len(fused_results))
