from core.models import TextChunk, LoadedDocument, ChunkMatch

# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_docling.loader import DoclingLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


class ResumeEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.semantic_splitter = SemanticChunker(
            self.embeddings, breakpoint_threshold_type="percentile"
        )
        self.semantic_splitter_low = SemanticChunker(
            self.embeddings, breakpoint_threshold_type="percentile"
        )

        self.current_resume: LoadedDocument | None = None
        self.current_job: LoadedDocument | None = None
    
    def load_resume(self, path: str):
        self.current_resume = LoadedDocument.create_from_path(
            path, self.semantic_splitter, self.embeddings
        )
    def load_job(self, path: str):
        self.current_job= LoadedDocument.create_from_path(
            path, self.semantic_splitter_low, self.embeddings
        )

    def update_resume_chunk(self, chunk_index: int, new_text: str):
        """Update text and re-vectorize (Write)"""
        if self.current_resume:
            chunk = self.current_resume.chunks[chunk_index]
            
            chunk.update_text(new_text, self.embeddings) 
            self.current_resume._vectorize_self(self.embeddings)

    def get_resume_chunks(self) -> list[TextChunk]:
        return self.current_resume.chunks if self.current_resume else []

    def get_matches(self, rerun: bool = False) -> list[ChunkMatch] | None:
        if not self.current_job or not self.current_resume:
            #TODO see how to send this to user layer
            print("Need to have a resume and job offer loaded")
            return 

        if self.match_list and not rerun:
            return self.match_list

        resume_vecs = [chunk.vector for chunk in self.current_resume.chunks]
        job_vecs = [chunk.vector for chunk in self.current_job.chunks]
        similarity_matrix = cosine_similarity(job_vecs, resume_vecs)

        matches = []
        for i, job_chunk in enumerate(self.current_job.chunks):
            scores = similarity_matrix[i]

            best_match_idx = np.argmax(scores)
            best_score = scores[best_match_idx]
            resume_match = self.current_resume.chunks[best_match_idx]

            if best_score > 0.85:
                status = "MET"
            elif best_score > 0.65:
                status = "WEAK MATCH"  
            else:
                status = "MISSING"

            match = ChunkMatch(
                resume_chunk=resume_match,
                job_chunk=job_chunk,
                similarity=best_score,
                status=status,
            )
            matches.append(match)

        self.match_list = matches
        return matches
     