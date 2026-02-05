# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_docling.loader import DoclingLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

FILE_PATH = "data/test.pdf"

docling = DoclingLoader(file_path=FILE_PATH)
txt_loader = TextLoader("data/job_offer.txt", encoding="utf-8")

job_offer_text = txt_loader.load()
resume_text = docling.load()

# # This can work decently. But Semantic splits very well (essentially is splitting by headers)
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_overlap=20,
#     length_function=len,
# )

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=50)
resume_chunks = semantic_splitter.split_documents(resume_text)

semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=50)
job_chunks = semantic_splitter.split_documents(job_offer_text)

job_chunks_texts = [doc.page_content for doc in job_chunks]
resume_chunks_texts = [doc.page_content for doc in resume_chunks]


print(f"lenght of chunks: JOB: {len(job_chunks)} RESUME: {len(resume_chunks)}")

resume_chunks_vecs = embeddings.embed_documents(resume_chunks_texts)
job_chunks_vecs = embeddings.embed_documents(job_chunks_texts)

similarity_matrix = cosine_similarity(job_chunks_vecs, resume_chunks_vecs)

similarities= []
for i, job_req in enumerate(job_chunks):
    # Get the row of scores for this specific job requirement
    scores = similarity_matrix[i]
    
    # Find the SINGLE BEST match in the resume
    best_match_idx = np.argmax(scores)
    best_score = scores[best_match_idx]
    best_match_text = resume_chunks[best_match_idx]
    
    # 5. Classify the Match
    if best_score > 0.85:
        status = "MET"
    elif best_score > 0.65:
        status = "WEAK MATCH" # Good candidate for rewriting!
    else:
        status = "MISSING" # Needs to be added from scratch
        
    similarities.append({
        "requirement": job_req,
        "match_status": status,
        "best_match_in_resume": best_match_text, # The specific bullet point to edit
        "score": best_score
    })


# for i, result in enumerate(results):
#     print(f" {i}. {result["score"]} \nRequirement:{result["requirement"]} \nMatch:{result["best_match_in_resume"]} \n{50*'-'}")
