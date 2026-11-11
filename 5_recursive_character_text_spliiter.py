
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)


tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""


# ============================================================
# Example 1: CharacterTextSplitter
# ============================================================

print("\n" + "=" * 60)
print("1. CHARACTER TEXT SPLITTER")
print("=" * 60)

splitter1 = CharacterTextSplitter(
    separator=" ",       # Split using spaces
    chunk_size=100,
    chunk_overlap=0
)

chunks1 = splitter1.split_text(tesla_text)

print(f"Total chunks: {len(chunks1)}")

for i, chunk in enumerate(chunks1, 1):
    print(f"\nChunk {i}: ({len(chunk)} characters)")
    print(f'"{chunk}"')


# ============================================================
# Example 2: RecursiveCharacterTextSplitter
# ============================================================

print("\n" + "=" * 60)
print("2. RECURSIVE CHARACTER TEXT SPLITTER")
print("=" * 60)

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=100,
    chunk_overlap=0
)

chunks2 = recursive_splitter.split_text(tesla_text)

print(f"Total chunks: {len(chunks2)}")

for i, chunk in enumerate(chunks2, 1):
    print(f"\nChunk {i}: ({len(chunk)} characters)")
    print(f'"{chunk}"')


# ============================================================
# Example 3: RecursiveCharacterTextSplitter with overlap
# ============================================================

print("\n" + "=" * 60)
print("3. RECURSIVE SPLITTER WITH CHUNK OVERLAP")
print("=" * 60)

recursive_splitter_overlap = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=100,
    chunk_overlap=20
)

chunks3 = recursive_splitter_overlap.split_text(tesla_text)

print(f"Total chunks: {len(chunks3)}")

for i, chunk in enumerate(chunks3, 1):
    print(f"\nChunk {i}: ({len(chunk)} characters)")
    print(f'"{chunk}"')
