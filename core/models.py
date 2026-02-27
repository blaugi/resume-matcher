import uuid
from dataclasses import dataclass

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


from pydantic import BaseModel, Field


class SuggestedEdit(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_text: str = Field(
        description="The exact original text from the resume to be replaced."
    )
    new_text: str = Field(
        description="The suggested new text to replace the original text."
    )
    reason: str = Field(
        description="A short explanation of why this edit is suggested."
    )
    status: str = Field(
        default="pending",
        description="Status of the edit: pending, accepted, rejected.",
    )
    projected_similarity: float = Field(default=0.0)
    similarity_delta: float = Field(default=0.0)


@dataclass
class TextChunk:
    document: Document
    vector: list[int | float]

    def update_text(self, new_text: str, embedding_model: HuggingFaceEmbeddings):
        self.document.page_content = new_text
        self.vector = embedding_model.embed_query(new_text)


@dataclass
class LoadedDocument:
    chunks: list[TextChunk]
    vector: list[int | float]
    current_text: str = ""

    @classmethod
    def create_from_path(
        cls,
        path: str,
        chunker,
        embedding_model: HuggingFaceEmbeddings,
        fallback_chunker=None,
    ):
        match path.split(".")[1]:
            case "pdf":
                document = DoclingLoader(path).load()
            case "txt" | "typ":
                document = TextLoader(path).load()
            case _:
                raise Exception("Unsupported file type.")

        if document_text := document[0].page_content:
            full_vector = embedding_model.embed_query(document_text)

        chunked_document = chunker.split_documents(document)
        if len(chunked_document) <= 1 and fallback_chunker:
            chunked_document = fallback_chunker.split_documents(document)

        chunked_document_texts = [doc.page_content for doc in chunked_document]
        chunk_embeddings = embedding_model.embed_documents(chunked_document_texts)

        chunks = []
        for i, chunk in enumerate(chunked_document):
            chunks.append(TextChunk(chunk, chunk_embeddings[i]))

        return cls(chunks, full_vector, current_text=document_text or "")

    @classmethod
    def create_from_text(
        cls,
        text: str,
        chunker,
        embedding_model: HuggingFaceEmbeddings,
        fallback_chunker=None,
    ):
        document = Document(
            page_content=text,
            metadata={"source": "in-memory string", "date": "2026-02-26"},
        )

        if document_text := document.page_content:
            full_vector = embedding_model.embed_query(document_text)

        chunked_document = chunker.split_documents([document])
        if len(chunked_document) <= 1 and fallback_chunker:
            chunked_document = fallback_chunker.split_documents([document])

        chunked_document_texts = [doc.page_content for doc in chunked_document]
        chunk_embeddings = embedding_model.embed_documents(chunked_document_texts)

        chunks = []
        for i, chunk in enumerate(chunked_document):
            chunks.append(TextChunk(chunk, chunk_embeddings[i]))

        return cls(chunks, full_vector, current_text=document_text or "")

    # This is assuming the order of the chunks wont change
    def get_full_text(self) -> str:
        return self.current_text

    def _vectorize_self(self, embedding_model: HuggingFaceEmbeddings):
        self.vector = embedding_model.embed_query(self.get_full_text())
