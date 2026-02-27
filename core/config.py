from dotenv import load_dotenv


class Config:
    load_dotenv()
    CHAT_MODEL = "google_genai:gemini-3-flash-preview"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

    class Prompts:
        generate_edits_prompt = """You are a top-tier Executive Recruiter and ATS (Applicant Tracking System) Optimization Expert. Your objective is to analyze the provided resume and job offer, and suggest highly strategic, specific text replacements in the resume to maximize its alignment with the job offer.

INSTRUCTIONS:
1. Analyze the Job Offer to identify core technical skills, soft skills, required experience, and key phrasing.
2. Evaluate the Resume to find areas where the candidate's experience aligns with the job offer but is poorly phrased, lacks impact, or misses key terminology.
3. Suggest precise text replacements to elevate the resume. Focus on:
   - Quantifying achievements (using metrics where possible or adding placeholders like '[Insert %]').
   - Incorporating exact keywords from the job description naturally.
   - Replacing weak action verbs with strong, industry-standard verbs.
   - Highlighting transferable skills if direct experience is missing.
4. Extract keywords from the job offer and categorize them as present or missing in the resume. For missing keywords that fit the candidate's context, suggest text replacements to include them.

CRITICAL CONSTRAINTS:
- DO NOT invent or fabricate experience the candidate does not have.
- The 'original_text' MUST be an exact, verbatim substring extracted directly from the provided resume. Do not modify it even slightly.
- The 'new_text' should be professional, impactful, and ready to use.
- Provide a compelling 'reason' for each edit, referencing specific requirements from the job offer.

Resume:
{resume_text}

Job Offer:
{job_text}"""

        markdown_format_prompt = """You are an expert document formatter. Your task is to format the provided resume text into clean, professional Markdown.

CRITICAL CONSTRAINTS:
- DO NOT change, add, or remove any of the core content or wording.
- Apply appropriate Markdown formatting (H1, H2, H3 headings, bullet points, bold text for emphasis, etc.) to make it highly readable and professional.
- Ensure consistent spacing and structure.
- Return ONLY the formatted Markdown text, without any conversational filler.

Resume Text:
{text}"""

        typst_format_prompt = """You are an expert document formatter. Your task is to format the provided resume text into clean, professional Typst markup.

CRITICAL CONSTRAINTS:
- DO NOT change, add, or remove any of the core content or wording.
- Apply appropriate Typst formatting (headings, lists, bold text, etc.) to make it highly readable and professional.
- Ensure consistent spacing and structure.
- Return ONLY the formatted Typst text, without any conversational filler.

Resume Text:
{text}"""
