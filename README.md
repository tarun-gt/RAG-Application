# Local RAG Application

A Retrieval-Augmented Generation (RAG) application built using **LangChain, Hugging Face, ChromaDB, and Ollama**. The project explores how to build a complete RAG pipeline using locally hosted models, without depending on paid LLM or embedding APIs.

The application currently focuses on information from Wikipedia pages covering **Google, Microsoft, NVIDIA, SpaceX, and Tesla**.

The goal is to retrieve relevant information from ingested documents, improve retrieval quality through hybrid search and reranking, and generate context-aware answers using a locally running LLM.

## Key Features

- **Document Ingestion:** Load and process text-based documents and explore PDF ingestion.
- **Document Parsing:** Extract text and structured content from supported document formats.
- **Text Chunking:** Split documents into smaller, manageable chunks using recursive and semantic chunking strategies.
- **Local Embeddings:** Generate document embeddings using Hugging Face Sentence Transformers.
- **Vector Storage:** Store and retrieve embeddings using ChromaDB.
- **Semantic Search:** Retrieve documents based on vector similarity.
- **Keyword Search:** Use BM25 to retrieve documents based on keyword relevance.
- **Hybrid Search:** Combine semantic and keyword-based retrieval.
- **Multi-Query Retrieval:** Generate alternative search queries to improve retrieval coverage.
- **Reciprocal Rank Fusion (RRF):** Combine ranked results from multiple retrieval strategies.
- **Cross-Encoder Reranking:** Improve relevance by reranking retrieved documents using a locally hosted cross-encoder.
- **Local Answer Generation:** Generate answers using Ollama and Llama 3.2.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Package Manager | uv |
| RAG Framework | LangChain |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Rank Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Local LLM Runtime | Ollama |
| Generation Model | `llama3.2` |
| Environment Variables | python-dotenv |

## Architecture

The application follows a multi-stage RAG architecture.

```text
                  DOCUMENT INGESTION
                          |
                          v
                 Document Processing
                          |
                          v
                    Text Chunking
                          |
                          v
                 Hugging Face Embeddings
                          |
                          v
                     ChromaDB
                  (Vector Storage)
                          |
                          |
USER QUERY ---------------+
     |
     v
 Query Processing / Multi-Query
     |
     v
 +-----------------------------+
 |                             |
 v                             v
Vector Search                BM25 Search
 |                             |
 +-------------+---------------+
               |
               v
       Reciprocal Rank Fusion
               |
               v
       Retrieved Documents
               |
               v
       Cross-Encoder Reranking
               |
               v
       Top Relevant Documents
               |
               v
        Ollama (Llama 3.2)
               |
               v
         Generated Answer
```

## How It Works

### 1. Document Ingestion

The first stage loads source documents into the application.

The current knowledge base focuses on Wikipedia content about:

- Google
- Microsoft
- NVIDIA
- SpaceX
- Tesla

The ingestion layer prepares source content for downstream processing.

### 2. Document Chunking

Large documents are divided into smaller chunks to improve retrieval accuracy and provide relevant context to the LLM.

Chunking strategies explored in this project include:

- Character-based chunking
- Recursive character text splitting
- Semantic chunking
- Agentic chunking (experimental)

Example using recursive character splitting:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)
```

### 3. Local Embeddings

The application uses Hugging Face Sentence Transformers to convert text chunks into numerical vector representations.

Model:

`sentence-transformers/all-MiniLM-L6-v2`

Example:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
```

The model produces 384-dimensional embeddings.

Using local embeddings eliminates the need for paid embedding API calls.

### 4. Vector Storage with ChromaDB

Generated embeddings are stored in ChromaDB for efficient similarity-based retrieval.

Example:

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="db/chroma_db",
    collection_metadata={"hnsw:space": "cosine"}
)
```

The vector database can subsequently be loaded using the same embedding model.

### 5. Hybrid Search

A single retrieval strategy may not retrieve all relevant information.

This project explores hybrid retrieval by combining:

**Vector Search**
- Retrieves semantically similar documents.
- Helps identify relevant information even when exact keywords differ.

**BM25 Search**
- Retrieves documents based on keyword matching.
- Helps identify documents containing important names, terms, and phrases.

Combining both approaches improves retrieval coverage.

### 6. Multi-Query Retrieval

A user query may have multiple possible interpretations or ways of expressing the same information.

Multi-query retrieval uses an LLM to generate alternative search queries.

For example:

Original query:

`How much did Microsoft pay to acquire GitHub?`

Possible alternative queries:

- What was the acquisition price of GitHub?
- How much did Microsoft's GitHub acquisition cost?
- What was the deal value when Microsoft acquired GitHub?

These queries can be used to retrieve additional relevant documents.

### 7. Reciprocal Rank Fusion (RRF)

When multiple retrieval strategies return ranked document lists, Reciprocal Rank Fusion combines their rankings.

The RRF formula is:

\[
RRF(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}
\]

Where:

- `d` represents a document.
- `R` represents the collection of ranked result lists.
- `rank_r(d)` represents the document's position in a result list.
- `k` is a ranking constant, commonly set to 60.

RRF helps combine results without requiring the scores from different retrieval methods to be directly comparable.

### 8. Cross-Encoder Reranking

The initial retrieval stage focuses on finding candidate documents. However, the retrieved documents may contain both relevant and irrelevant results.

A cross-encoder evaluates each query-document pair together and assigns a relevance score.

Model:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

Example:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

pairs = [
    (query, doc.page_content)
    for doc in retrieved_docs
]

scores = reranker.predict(pairs)

ranked_results = sorted(
    zip(retrieved_docs, scores),
    key=lambda item: item[1],
    reverse=True
)

reranked_docs = [
    doc for doc, score in ranked_results[:10]
]
```

This improves the ordering of retrieved documents before they are passed to the LLM.

### 9. Answer Generation with Ollama

The final stage uses a locally hosted LLM to generate answers based on the reranked documents.

Model:

`llama3.2`

Example:

```python
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.2",
    temperature=0
)

result = model.invoke(combined_input)

print(result.content)
```

The model receives the original user query and the top relevant documents as context.

The generation prompt instructs the model to:

- Answer using the supplied documents.
- Avoid introducing unsupported external information.
- Acknowledge when the retrieved context is insufficient.

## Prerequisites

Install the following before running the application:

1. Python 3.13
2. uv package manager
3. Ollama
4. Git

### Install uv

Follow the official installation instructions:

https://docs.astral.sh/uv/getting-started/installation/

### Install Ollama

Download Ollama for Windows, macOS, or Linux:

https://ollama.com/download

Download the Llama 3.2 model:

```bash
ollama pull llama3.2
```

Verify the installation:

```bash
ollama list
```

Make sure the Ollama service is running before executing the application.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd RAG
```

### 2. Create the virtual environment

```bash
uv sync
```

This installs dependencies defined in `pyproject.toml` and synchronizes the environment using `uv.lock`.

### 3. Install required dependencies

If you are setting up the project from scratch, the following packages cover the core pipeline described above:

```bash
uv add langchain
uv add langchain-core
uv add langchain-community
uv add langchain-classic
uv add langchain-text-splitters
uv add langchain-huggingface
uv add langchain-chroma
uv add langchain-ollama
uv add sentence-transformers
uv add rank-bm25
uv add python-dotenv
```

Then synchronize:

```bash
uv sync
```

Note: Add optional packages such as `langchain-experimental` or document-specific parsing dependencies only if you use those features.

### 4. Download the local models

The Hugging Face embedding and cross-encoder models are downloaded automatically when first initialized, provided the machine has internet access.

Download the LLM:

```bash
ollama pull llama3.2
```

### 5. Run the application

For a Python script:

```bash
uv run python main.py
```

For a Jupyter Notebook, select the project's `.venv` Python interpreter as the notebook kernel.

## Configuration

The application uses locally hosted models, so OpenAI and Cohere API keys are not required for the described core pipeline.

| Component | Configuration |
|---|---|
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Ollama Model | `llama3.2` |
| Ollama Base URL | `http://localhost:11434` |
| ChromaDB Directory | `db/chroma_db` |
| Vector Similarity | Cosine |

## Project Structure

An example structure for organizing the application:

```text
RAG/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── db/
│   └── chroma_db/
│
├── notebooks/
│   └── rag_experiments.ipynb
│
├── src/
│   ├── ingestion/
│   │   └── document_loader.py
│   │
│   ├── chunking/
│   │   └── text_splitter.py
│   │
│   ├── embeddings/
│   │   └── embedding_model.py
│   │
│   ├── retrieval/
│   │   ├── vector_search.py
│   │   ├── hybrid_search.py
│   │   ├── multi_query.py
│   │   └── reranker.py
│   │
│   ├── generation/
│   │   └── answer_generator.py
│   │
│   └── main.py
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

This is a suggested organization; adjust the filenames and folders to match the actual repository.

## Important Considerations

- Use the same embedding model and configuration for both indexing and querying.
- If you change the embedding model, recreate the ChromaDB collection because vector dimensions and embedding spaces may differ.
- The first model download requires internet access; subsequent inference can run locally.
- Model inference speed depends on available CPU, RAM, and GPU resources.
- Cross-encoder reranking is more computationally expensive than basic vector similarity search.
- `llama3.2` is a text model; image understanding requires a vision-capable model and an appropriate multimodal pipeline.
- The quality of generated answers depends heavily on document quality, chunking, retrieval, and reranking.

## Roadmap

- [x] Local Hugging Face embeddings
- [x] ChromaDB vector storage
- [x] BM25 keyword retrieval exploration
- [x] Hybrid retrieval exploration
- [x] Multi-query retrieval exploration
- [x] Reciprocal Rank Fusion implementation
- [x] Local cross-encoder reranking
- [x] Local LLM answer generation using Ollama
- [ ] Improve document ingestion and metadata handling
- [ ] Add source attribution and citations to generated answers
- [ ] Evaluate retrieval quality using Recall@K, Precision@K, and MRR
- [ ] Evaluate answer quality and faithfulness
- [ ] Explore advanced reranking and retrieval strategies
- [ ] Build an API layer for the RAG application
- [ ] Add a user interface for interacting with the knowledge base

*The checklist reflects the techniques and code discussed during development; verify each item against the current repository before publishing.*

## Learning Objectives

This project is also a hands-on exploration of the core concepts behind production-oriented RAG systems:

- Understanding the complete RAG lifecycle.
- Comparing different document chunking strategies.
- Understanding embedding models and vector similarity.
- Implementing semantic and keyword-based retrieval.
- Improving retrieval through query expansion and rank fusion.
- Understanding the difference between bi-encoder retrieval and cross-encoder reranking.
- Running LLMs locally without paid inference APIs.
- Evaluating the quality of retrieved context and generated answers.

## License
This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [ChromaDB](https://www.trychroma.com/)
- [Ollama](https://ollama.com/)
- [Sentence Transformers](https://www.sbert.net/)

---

**Built as a hands-on exploration of local Retrieval-Augmented Generation, information retrieval, and open-source LLM technologies.**