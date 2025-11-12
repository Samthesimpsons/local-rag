import logging
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import Config
from src.services.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages ChromaDB vector store operations."""

    def __init__(self) -> None:
        """Initialize the vector store with ChromaDB and embedding model."""
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(f"Initializing ChromaDB at {Config.CHROMA_DB_PATH}")
        self.client = chromadb.PersistentClient(
            path=str(Config.CHROMA_DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )

        self.logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(
            Config.EMBEDDING_MODEL_NAME,
            cache_folder=str(Config.MODELS_PATH),
            token=Config.HUGGINGFACE_TOKEN,
        )

        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self.logger.info(f"Vector store initialized with collection: {Config.COLLECTION_NAME}")

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects to add
        """
        if not chunks:
            self.logger.warning("No chunks to add to vector store")
            return

        self.logger.info(f"Adding {len(chunks)} chunks to vector store...")

        documents = [chunk.content for chunk in chunks]
        metadatas: list[dict[str, str | int | float | bool]] = [
            dict(chunk.metadata.items()) for chunk in chunks
        ]
        ids = [f"{chunk.metadata['source']}_page_{chunk.metadata['page']}" for chunk in chunks]

        self.logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            documents, show_progress_bar=True, convert_to_numpy=True
        )

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self.logger.info(f"Adding batch {i // batch_size + 1} ({i}-{batch_end})")

            self.collection.add(
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                metadatas=metadatas[i:batch_end],  # type: ignore
                ids=ids[i:batch_end],
            )

        self.logger.info(f"Successfully added {len(chunks)} chunks to vector store")

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Search for similar documents in the vector store.

        Args:
            query: Search query text
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing content and metadata
        """
        self.logger.info(f"Searching for: '{query}' (top_k={top_k})")

        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        formatted_results: list[dict[str, Any]] = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = {}
                if results["metadatas"] and results["metadatas"][0]:
                    metadata = results["metadatas"][0][i]
                distance = None
                if results.get("distances") and results["distances"] and results["distances"][0]:
                    distance = results["distances"][0][i]
                formatted_results.append(
                    {
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance,
                    }
                )

        self.logger.info(f"Found {len(formatted_results)} relevant chunks")

        return formatted_results

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        count = self.collection.count()
        self.logger.info(f"Collection contains {count} documents")
        return count

    def reset_collection(self) -> None:
        """Delete and recreate the collection (careful!)."""
        self.logger.warning("Resetting collection...")
        self.client.delete_collection(name=Config.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info("Collection reset complete")
