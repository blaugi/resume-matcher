import uuid
from dataclasses import dataclass, field

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


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

    @classmethod
    def create_from_path(
        cls, path: str, chunker: SemanticChunker, embedding_model: HuggingFaceEmbeddings
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
        chunked_document_texts = [doc.page_content for doc in chunked_document]
        chunk_embeddings = embedding_model.embed_documents(chunked_document_texts)

        chunks = []
        for i, chunk in enumerate(chunked_document):
            chunks.append(TextChunk(chunk, chunk_embeddings[i]))

        return cls(chunks, full_vector)

    @classmethod
    def create_from_text(
        cls, text: str, chunker: SemanticChunker, embedding_model: HuggingFaceEmbeddings
    ):
        document = Document(
            page_content=text,
            metadata={"source": "in-memory string", "date": "2026-02-26"}
        )

        if document_text := document.page_content:
            full_vector = embedding_model.embed_query(document_text)

        chunked_document = chunker.split_documents([document])
        chunked_document_texts = [doc.page_content for doc in chunked_document]
        chunk_embeddings = embedding_model.embed_documents(chunked_document_texts)

        chunks = []
        for i, chunk in enumerate(chunked_document):
            chunks.append(TextChunk(chunk, chunk_embeddings[i]))

        return cls(chunks, full_vector)
    

    # This is assuming the order of the chunks wont change
    def get_full_text(self) -> str:
        full_text = ""
        for chunk in self.chunks:
            full_text += chunk.document.page_content
        return full_text

    def _vectorize_self(self, embedding_model: HuggingFaceEmbeddings):
        self.vector = embedding_model.embed_query(self.get_full_text())

@dataclass
class ChunkMatch:
    resume_chunk: TextChunk
    job_chunk: TextChunk
    similarity: float
    status: str
    original_similarity: float = 0.0
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.original_similarity == 0.0:
            self.original_similarity = self.similarity

    def get_job_text(self) -> str:
        return self.job_chunk.document.page_content

    def get_resume_text(self) -> str:
        return self.resume_chunk.document.page_content
