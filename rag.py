
import asyncio
from pathlib import Path

import aiohttp
import chromadb


DOCUMENTS_DIR = Path("documents")
DATABASE_DIR = Path("chroma_db")

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "embeddinggemma"

COLLECTION_NAME = "python_documentation"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


client = chromadb.PersistentClient(path=str(DATABASE_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


def split_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


async def create_embeddings(texts):
    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            OLLAMA_URL,
            json={
                "model": EMBEDDING_MODEL,
                "input": texts
            }
        ) as response:
            response.raise_for_status()
            data = await response.json()

    return data["embeddings"]


async def index_documents():
    global collection
    DOCUMENTS_DIR.mkdir(exist_ok=True)

    documents = []
    metadatas = []
    ids = []

    for file_path in DOCUMENTS_DIR.iterdir():
        if file_path.suffix.lower() not in {".txt", ".md"}:
            continue

        text = file_path.read_text(encoding="utf-8")
        chunks = split_text(text)

        for index, chunk in enumerate(chunks):
            documents.append(chunk)

            metadatas.append({
                "source": file_path.name,
                "chunk": index + 1
            })

            ids.append(f"{file_path.name}_{index}")

    if not documents:
        print("В папке documents нет документов для индексации.")
        return

    embeddings = []

    batch_size = 32

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        batch_embeddings = await create_embeddings(batch)
        embeddings.extend(batch_embeddings)

    client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Документов обработано: {len(list(DOCUMENTS_DIR.iterdir()))}")
    print(f"Фрагментов добавлено в базу: {len(documents)}")


async def search_documents(query, limit=4):
    if collection.count() == 0:
        return []

    query_embedding = await create_embeddings([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(limit, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    found_documents = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        found_documents.append({
            "text": document,
            "source": metadata["source"],
            "chunk": metadata["chunk"],
            "distance": distance
        })

    return found_documents


if __name__ == "__main__":
    asyncio.run(index_documents())
