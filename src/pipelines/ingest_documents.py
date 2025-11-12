import logging

from src.config import Config
from src.services.document_processor import DocumentProcessor
from src.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def main() -> None:
    """Ingest documents into ChromaDB."""
    logger.info("Starting document ingestion pipeline...")

    try:
        processor = DocumentProcessor()

        logger.info(f"Processing documents from: {Config.DOCUMENTS_PATH}")
        chunks = processor.process_directory(Config.DOCUMENTS_PATH)

        if not chunks:
            logger.error(
                "No document chunks extracted. Please add PDF files to the documents directory."
            )
            return

        vector_store = VectorStore()

        existing_count = vector_store.get_collection_count()
        if existing_count > 0:
            logger.warning(
                f"Collection already contains {existing_count} documents. "
                "This will add new documents to the existing collection."
            )
            response = input("Do you want to reset the collection first? (yes/no): ")
            if response.lower() in ["yes", "y"]:
                vector_store.reset_collection()
                logger.info("Collection reset complete.")

        vector_store.add_documents(chunks)

        final_count = vector_store.get_collection_count()
        logger.info(f"Document ingestion complete! Total documents in collection: {final_count}")

    except Exception as error:
        logger.error(f"Document ingestion pipeline failed: {error}")
        raise


if __name__ == "__main__":
    main()
