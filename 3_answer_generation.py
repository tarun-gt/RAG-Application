
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

persistent_directory = "db/chroma_db"

# Load the same embedding model used during ingestion
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# Load ChromaDB
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

# User query
query = "How much did Microsoft pay to acquire GitHub?"

# Retrieve relevant documents
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

# Display retrieved context
print("\n--- Retrieved Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:\n{doc.page_content}")

# Check whether documents were retrieved
if not relevant_docs:
    print("\nNo relevant documents found.")
    exit()

# Combine query and retrieved documents
context = "\n\n".join(
    [doc.page_content for doc in relevant_docs]
)

combined_input = f"""
Based on the following documents, answer this question:

Question:
{query}

Documents:
{context}

Instructions:
- Answer using only the information in the documents.
- Do not use external knowledge or make assumptions.
- If the documents do not contain enough information,
  say exactly:
  "I don't have enough information to answer that question
  based on the provided documents."
- Give a clear and concise answer.
"""

# Initialize local LLM using Ollama
model = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Define messages
messages = [
    SystemMessage(
        content="You are a helpful assistant. Answer strictly based on the provided context."
    ),
    HumanMessage(content=combined_input)
]

# Generate response
result = model.invoke(messages)

# Display generated answer
print("\n--- Generated Response ---")
print(result.content)
