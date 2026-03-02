import uuid
from dataclasses import dataclass

from langchain_community.document_loaders import TextLoader
from langchain_docling.loader import DoclingLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field


class SuggestedEdit(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_text: str = Field(
        description="The exact original text from the resume to be replaced."
    )
    new_text: str = Field(
        description="The suggested new text to replace the original text. You may add 'variables', that is to say, values/information you do not know but would be interesting to add. Add them with this **format**."
    )
    reason: str = Field(
        description="A short explanation of why this edit is suggested. Include the motivating section from the job offer (truncated if needed)."
    )
    status: str = Field(
        default="pending",
        description="Status of the edit: pending, accepted, rejected.",
    )
    projected_similarity: float = Field(default=0.0)
    similarity_delta: float = Field(default=0.0)
    is_keyword:bool = Field(
        description="Whether or not the current edit is keyword addition or rephrasing."
    )


@dataclass
class LoadedDocument:
    vector: list[int | float]
    current_text: str = ""

    @classmethod
    def create_from_path(
        cls,
        path: str,
        embedding_model: HuggingFaceEmbeddings,
    ):
        match path.split(".")[1]:
            case "pdf":
                document = DoclingLoader(path).load()
            case "txt" | "typ":
                document = TextLoader(path, encoding="utf-8").load()
            case _:
                raise Exception("Unsupported file type.")

        document_text = document[0].page_content or ""
        full_vector = embedding_model.embed_query(document_text)

        return cls(full_vector, current_text=document_text)

    @classmethod
    def create_from_text(
        cls,
        text: str,
        embedding_model: HuggingFaceEmbeddings,
    ):
        document_text = text
        full_vector = embedding_model.embed_query(document_text)

        return cls(full_vector, current_text=document_text)

    def get_full_text(self) -> str:
        return self.current_text

    def _vectorize_self(self, embedding_model: HuggingFaceEmbeddings):
        self.vector = embedding_model.embed_query(self.get_full_text())
