from dotenv import load_dotenv


class Config:
    load_dotenv()
    CHAT_MODEL = "google_genai:gemini-3-flash-preview"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

    class Prompts:
        generate_edits_prompt = """You are an expert resume writer. Your task is to analyze the provided resume and job offer, and suggest specific text replacements in the resume to better match the job offer.
Provide exact string replacements. The 'original_text' must be an exact substring of the resume.

Resume:
{resume_text}

Job Offer:
{job_text}"""

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
