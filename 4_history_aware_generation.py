
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# Connect to the existing ChromaDB
persistent_directory = "db/chroma_db"

# Use the same embedding model used during ingestion
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)


# Initialize local Ollama LLM
model = ChatOllama(
    model="llama3.2",
    temperature=0
)


# Store conversation history
chat_history = []


def rewrite_question(user_question):
    """
    Rewrite the user's follow-up question into a standalone
    question using previous conversation history.
    """

    if not chat_history:
        return user_question

    messages = [
        SystemMessage(
            content=(
                "You are a question rewriting assistant. "
                "Given the conversation history and the new question, "
                "rewrite the new question as a standalone question "
                "that can be understood without previous context. "
                "Do not answer the question. Return only the rewritten question."
            )
        )
    ] + chat_history + [
        HumanMessage(content=f"New question: {user_question}")
    ]

    result = model.invoke(messages)

    search_question = result.content.strip()

    print(f"Searching for: {search_question}")

    return search_question


def retrieve_documents(search_question, k=3):
    """
    Retrieve relevant document chunks from ChromaDB.
    """

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(search_question)

    print(f"\nFound {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)

        print(f"\nDocument {i}:")
        print(f"{preview}...")

    return docs


def generate_answer(user_question, docs):
    """
    Generate an answer using retrieved documents
    and conversation history.
    """

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    combined_input = f"""
Answer the following question using the provided documents.

Question:
{user_question}

Retrieved Documents:
{context}

Instructions:
1. Use only the information available in the retrieved documents.
2. Do not use external knowledge or make assumptions.
3. Consider conversation history only to understand the user's intent.
4. If the retrieved documents do not contain enough information,
   respond exactly with:
   "I don't have enough information to answer that question based on the provided documents."
5. Provide a clear and concise answer.
"""

    messages = [
        SystemMessage(
            content=(
                "You are a helpful RAG assistant. "
                "Answer questions strictly based on the retrieved documents. "
                "Never invent information."
            )
        )
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)

    return result.content


def ask_question(user_question):
    """
    Complete conversational RAG workflow.
    """

    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Rewrite follow-up question
    search_question = rewrite_question(user_question)

    # Step 2: Retrieve relevant documents
    docs = retrieve_documents(search_question, k=3)

    # Step 3: Generate answer
    if not docs:
        answer = (
            "I don't have enough information to answer that "
            "question based on the provided documents."
        )
    else:
        answer = generate_answer(user_question, docs)

    # Step 4: Update conversation history
    chat_history.append(
        HumanMessage(content=user_question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )

    # Step 5: Display answer
    print("\n--- Generated Response ---")
    print(answer)

    return answer


def start_chat():
    print("=" * 60)
    print("Conversational RAG System")
    print("Powered by HuggingFace + ChromaDB + Ollama")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "quit":
            print("\nGoodbye!")
            break

        if not question:
            continue

        try:
            ask_question(question)

        except Exception as e:
            print(f"\nError processing your question: {e}")


if __name__ == "__main__":
    start_chat()
