import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""

    content: str
    metadata: dict[str, str | int]


class DocumentProcessor:
    """Process PDF documents and extract text chunks at page level."""

    def __init__(self) -> None:
        """Initialize the document processor."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_pdf(self, pdf_path: Path) -> list[DocumentChunk]:
        """
        Process a single PDF file and extract text at page level.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of DocumentChunk objects, one per page
        """
        self.logger.info(f"Processing PDF: {pdf_path.name}")
        chunks: list[DocumentChunk] = []

        try:
            document = pymupdf.open(pdf_path)

            for page_number in range(len(document)):
                page = document[page_number]
                text = str(page.get_text())

                if not text.strip():
                    self.logger.debug(f"Skipping empty page {page_number + 1} in {pdf_path.name}")
                    continue

                chunk = DocumentChunk(
                    content=text.strip(),
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number + 1,
                        "total_pages": len(document),
                        "file_path": str(pdf_path.absolute()),
                    },
                )
                chunks.append(chunk)

            document.close()
            self.logger.info(f"Extracted {len(chunks)} pages from {pdf_path.name}")

        except Exception as error:
            self.logger.error(f"Failed to process {pdf_path.name}: {error}")
            raise

        return chunks

    def process_directory(self, directory_path: Path) -> list[DocumentChunk]:
        """
        Process all PDF files in a directory.

        Args:
            directory_path: Path to directory containing PDF files

        Returns:
            List of all DocumentChunk objects from all PDFs
        """
        self.logger.info(f"Processing PDFs in directory: {directory_path}")
        all_chunks: list[DocumentChunk] = []

        pdf_files = list(directory_path.glob("*.pdf"))

        if not pdf_files:
            self.logger.warning(f"No PDF files found in {directory_path}")
            return all_chunks

        self.logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_path in pdf_files:
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as error:
                self.logger.error(f"Skipping {pdf_path.name} due to error: {error}")
                continue

        self.logger.info(f"Total chunks extracted from {len(pdf_files)} PDFs: {len(all_chunks)}")

        return all_chunks
