"""Document chunking utility for GeoPol Bot knowledge pipeline.

Splits documents into contextual passages while rigorously preserving
source URLs, publication dates, and institutional attribution metadata.
"""

from typing import List
from backend.app.knowledge.schemas import DocumentSource, DocumentChunk


class DocumentChunker:
    """Chunks documents into semantic passages while preserving all provenance metadata."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: DocumentSource) -> List[DocumentChunk]:
        """Split a single DocumentSource into a list of DocumentChunks."""
        text = doc.content.strip()
        if not text:
            return []

        chunks: List[DocumentChunk] = []
        start = 0
        text_length = len(text)
        chunk_index = 0

        while start < text_length:
            end = start + self.chunk_size

            # If we are not at the end of the text, try to find a natural break (sentence or space)
            if end < text_length:
                # Look for sentence boundary (.!?) near end
                break_point = -1
                for punct in [". ", "? ", "! ", "\n"]:
                    pos = text.rfind(punct, start, end)
                    if pos != -1 and pos > start + (self.chunk_size // 2):
                        break_point = pos + 1
                        break

                if break_point != -1:
                    end = break_point
                else:
                    # Fallback to last space
                    space_pos = text.rfind(" ", start, end)
                    if space_pos != -1 and space_pos > start + (self.chunk_size // 2):
                        end = space_pos

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"{doc.id}#chunk-{chunk_index}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc.id,
                        chunk_index=chunk_index,
                        title=doc.title,
                        source_name=doc.source_name,
                        source_url=doc.source_url,
                        publication_date=doc.publication_date,
                        topic=doc.topic,
                        content=chunk_text,
                    )
                )
                chunk_index += 1

            if end >= text_length:
                break

            # Slide window forward with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def chunk_corpus(self, corpus: List[DocumentSource]) -> List[DocumentChunk]:
        """Chunk an entire collection of documents."""
        all_chunks: List[DocumentChunk] = []
        for doc in corpus:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
