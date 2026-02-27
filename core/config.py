from dotenv import load_dotenv


class Config:
    load_dotenv()
    CHAT_MODEL = "google_genai:gemini-3-flash-preview"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

    class Prompts:
        reformat_chunk = """You are an expert resume writer. Your task is to rewrite a specific chunk of a resume to better align with a given job requirement, while remaining truthful to the original experience.

Job Requirement:
{job_chunk}

Original Resume Chunk:
{resume_chunk}

Rewrite the resume chunk to highlight the skills and experiences that match the job requirement. Use strong action verbs and professional language. Do not invent new experiences. Return ONLY the rewritten resume chunk text."""

        extract_keywords = """You are an expert technical recruiter. Extract the most important keywords, skills, and technologies from the following job description chunk.

Job Description Chunk:
{job_chunk}

Return the keywords as a comma-separated list."""

        markdown_format_prompt = """You are an expert document formatter. Your task is to format the provided resume text into Markdown.
Do not change the core content, only apply Markdown formatting (headings, bullet points, bold text, etc.) to make it look professional.
Return ONLY the formatted Markdown text.

Resume Text:
{text}"""

        typst_format_prompt = """You are an expert document formatter. Your task is to format the provided resume text into Typst.
Do not change the core content, only apply Typst formatting to make it look professional.
Return ONLY the formatted Typst text.

Resume Text:
{text}"""
