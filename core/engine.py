import numpy as np
from langchain.chat_models import init_chat_model

# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from core.config import Config
from core.models import ChunkMatch, LoadedDocument, TextChunk


class ResumeEngine:
    def __init__(self):

        self.model = init_chat_model(Config.CHAT_MODEL)
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.semantic_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=50,
        )
        self.semantic_splitter_low = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80,
        )

        self.current_resume: LoadedDocument | None = None
        self.current_job: LoadedDocument | None = None
        self.match_list: list[ChunkMatch] | None = None
        self.original_overall_similarity: float | None = None

    def load_resume(self, path: str):
        self.current_resume = LoadedDocument.create_from_path(
            path, self.semantic_splitter, self.embeddings
        )

    def load_job(self, input_str: str, path: bool = False):
        if path:
            self.current_job = LoadedDocument.create_from_path(
                input_str, self.semantic_splitter_low, self.embeddings
            )
        else:
            self.current_job = LoadedDocument.create_from_text(
                input_str, self.semantic_splitter_low, self.embeddings
            )

    def update_resume_chunk(self, chunk_id: str, new_text: str):
        """Update text, re-vectorize local chunk, update full doc vector, and recalculate similarity."""
        match = self.get_match_from_uuid(chunk_id)
        if self.current_resume and match:
            # Update local chunk text and vector
            match.resume_chunk.update_text(new_text, self.embeddings)

            # Recalculate local similarity for this chunk match
            new_similarity = float(
                cosine_similarity(
                    [match.job_chunk.vector], [match.resume_chunk.vector]
                )[0][0]
            )
            match.similarity = new_similarity
            match.status = self._get_status_from_score(new_similarity)

            # Update overall document vector
            self.current_resume._vectorize_self(self.embeddings)

    def _get_status_from_score(self, score: float) -> str:
        if score > 0.85:
            return "MET"
        elif score > 0.65:
            return "WEAK MATCH"
        else:
            return "MISSING"

    def get_resume_chunks(self) -> list[TextChunk]:
        return self.current_resume.chunks if self.current_resume else []

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

    def get_matches(self, rerun: bool = False) -> list[ChunkMatch] | None:
        if not self.current_job or not self.current_resume:
            # TODO see how to send this to user layer
            print("Need to have a resume and job offer loaded")
            return

        if self.match_list and not rerun:
            return self.match_list

        # Capture initial overall similarity
        if self.original_overall_similarity is None:
            score, _ = self.get_overall_similarity()
            self.original_overall_similarity = score

        resume_vecs = [chunk.vector for chunk in self.current_resume.chunks]
        job_vecs = [chunk.vector for chunk in self.current_job.chunks]
        similarity_matrix = cosine_similarity(job_vecs, resume_vecs)

        matches = []
        for i, job_chunk in enumerate(self.current_job.chunks):
            scores = similarity_matrix[i]

            best_match_idx = np.argmax(scores)
            best_score = scores[best_match_idx]
            resume_match = self.current_resume.chunks[best_match_idx]

            match = ChunkMatch(
                resume_chunk=resume_match,
                job_chunk=job_chunk,
                similarity=float(best_score),
                status=self._get_status_from_score(best_score),
                original_similarity=float(best_score),
            )
            matches.append(match)

        self.match_list = matches
        return matches

    def get_match_from_uuid(self, uuid: str) -> ChunkMatch | None:
        if not self.match_list:
            return None

        for match in self.match_list:
            if match.chunk_id == uuid:
                found_match = match
                break
        else:
            found_match = None
        return found_match

    def get_match_texts(self, uuid: str) -> tuple[str, str]:
        """Returns job_text, resume_text as strings"""
        chunk_match = self.get_match_from_uuid(uuid)
        if chunk_match:
            return chunk_match.get_job_text(), chunk_match.get_resume_text()
        else:
            return "Error", "Error"

    def _call_llm(self, sys_prompt: str, query: str) -> str:
        messages = [
            (
                "system",
                sys_prompt,
            ),
            ("human", query),
        ]
        return self.model.invoke(messages).text

    def reformat_chunk(self, chunk: ChunkMatch):
        prompt = Config.Prompts.reformat_chunk.format(
            job_chunk=chunk.get_job_text(), resume_chunk=chunk.get_resume_text()
        )

        return self._call_llm(sys_prompt="", query=prompt)
