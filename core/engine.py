from langchain.chat_models import init_chat_model

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from core.config import Config
from core.models import LoadedDocument, SuggestedEdit
from pydantic import BaseModel, Field


class SuggestedEditsList(BaseModel):
    edits: list[SuggestedEdit] = Field(
        description="List of suggested edits to the resume."
    )


class ResumeEngine:
    def __init__(self):

        self.model = init_chat_model(Config.CHAT_MODEL)
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

        self.current_resume: LoadedDocument | None = None
        self.current_job: LoadedDocument | None = None
        self.edit_list: list[SuggestedEdit] | None = None
        self.original_overall_similarity: float | None = None

    def load_resume(self, path: str):
        self.current_resume = LoadedDocument.create_from_path(
            path, self.embeddings
        )

    def load_job(self, input_str: str, path: bool = False):
        if path:
            self.current_job = LoadedDocument.create_from_path(
                input_str,
                self.embeddings,
            )
        else:
            self.current_job = LoadedDocument.create_from_text(
                input_str,
                self.embeddings,
            )

    def get_overall_similarity(self) -> tuple[float, float]:
        """Returns (current_similarity, original_similarity)"""
        if not self.current_job or not self.current_resume:
            return 0.0, 0.0

        # Calculate cosine similarity between the full document vectors
        similarity = cosine_similarity(
            [self.current_job.vector], [self.current_resume.vector]
        )[0][0]

        orig = (
            self.original_overall_similarity
            if self.original_overall_similarity is not None
            else float(similarity)
        )
        return float(similarity), orig

    def calculate_document_similarity(self, text1: str, text2: str) -> float:
        vec1 = self.embeddings.embed_query(text1)
        vec2 = self.embeddings.embed_query(text2)
        return float(cosine_similarity([vec1], [vec2])[0][0])

    def generate_edits(self) -> list[SuggestedEdit]:
        if not self.current_job or not self.current_resume:
            print("Need to have a resume and job offer loaded")
            return []

        # Capture initial overall similarity
        if self.original_overall_similarity is None:
            score, _ = self.get_overall_similarity()
            self.original_overall_similarity = score

        resume_text = self.current_resume.current_text
        job_text = self.current_job.current_text

        prompt = Config.Prompts.generate_edits_prompt.format(
            resume_text=resume_text, job_text=job_text
        )

        structured_model = self.model.with_structured_output(SuggestedEditsList)
        result = structured_model.invoke(prompt)

        edits = result.edits
        baseline_similarity = self.calculate_document_similarity(resume_text, job_text)

        for edit in edits:
            if edit.original_text in resume_text:
                temp_resume = resume_text.replace(edit.original_text, edit.new_text, 1)
                projected_sim = self.calculate_document_similarity(
                    temp_resume, job_text
                )
                edit.projected_similarity = projected_sim
                edit.similarity_delta = projected_sim - baseline_similarity
            else:
                edit.status = "invalid"  # Original text not found

        self.edit_list = edits
        return edits

    def get_edit_from_id(self, edit_id: str) -> SuggestedEdit | None:
        if not self.edit_list:
            return None
        for edit in self.edit_list:
            if edit.id == edit_id:
                return edit
        return None

    def apply_edit(self, edit_id: str, custom_new_text: str | None = None) -> bool:
        edit = self.get_edit_from_id(edit_id)
        if not edit or not self.current_resume:
            return False

        new_text = custom_new_text if custom_new_text is not None else edit.new_text

        if edit.original_text in self.current_resume.current_text:
            self.current_resume.current_text = self.current_resume.current_text.replace(
                edit.original_text, new_text, 1
            )
            edit.status = "accepted"
            # Update overall vector
            self.current_resume._vectorize_self(self.embeddings)
            return True
        return False

    def _call_llm(self, sys_prompt: str, query: str) -> str:
        messages = [
            (
                "system",
                sys_prompt,
            ),
            ("human", query),
        ]
        return self.model.invoke(messages).text

    def format_resume_text(self, text: str, format_type: str) -> str:
        if format_type.lower() == "markdown":
            prompt = Config.Prompts.markdown_format_prompt.format(text=text)
        elif format_type.lower() == "typst":
            prompt = Config.Prompts.typst_format_prompt.format(text=text)
        else:
            return text

        return self._call_llm(sys_prompt="", query=prompt)
