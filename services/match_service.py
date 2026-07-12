"""
match_service.py

Core matching logic for the AI Resume Matcher.
Handles PDF text extraction, LLM-based skill extraction (resume + JD),
skill normalization, match scoring, and suggestion generation.

Architecture, FastAPI routes, and Gradio UI are unchanged -- this file
only improves reliability, consistency, and accuracy of the matching logic.
"""

import os
import json
import time
import pdfplumber
import requests

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable is not set. "
        "Add it as a secret in your HF Space settings."
    )

API_URL = "https://router.huggingface.co/v1/chat/completions"
# Most default HF providers (Together, Fireworks, Sambanova, ...) only host the
# bigger Qwen2.5 variants (7B+). Featherless AI is confirmed to host the 3B one,
# so we pin it explicitly with the ":provider" suffix. Model is unchanged.
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct:featherless-ai"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# Minimum characters of extracted PDF text before we consider it "readable".
MIN_RESUME_TEXT_LENGTH = 30

# Empty skeleton returned whenever extraction/parsing fails, so downstream
# code (calculate_score, etc.) never has to special-case a missing key.
EMPTY_EXTRACTION = {"skills": [], "experience": [], "education": []}

# Common aliases -> standard skill name. Lookup is case-insensitive and
# whitespace-normalized (extra internal spaces collapsed) before matching.
SKILL_ALIASES = {
    "fast api": "FastAPI",
    "fastapi": "FastAPI",
    "py torch": "PyTorch",
    "pytorch": "PyTorch",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "llms": "LLM",
    "llm": "LLM",
    "large language model": "LLM",
    "large language models": "LLM",
    "cv": "Computer Vision",
    "computer vision": "Computer Vision",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "react js": "React",
    "reactjs": "React",
    "react": "React",
    "node js": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "mongo db": "MongoDB",
    "mongodb": "MongoDB",
    "tensor flow": "TensorFlow",
    "tensorflow": "TensorFlow",
    "postgre sql": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "sql server": "SQL Server",
    "hugging face": "Hugging Face",
    "huggingface": "Hugging Face",
    "hugging-face": "Hugging Face",
    "aws": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
}


class MatchService:
    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------
    def ask_llm(self, prompt, _retry=True):
        """
        Calls the Hugging Face Router chat-completions endpoint.
        temperature/top_p are set low to make skill extraction more
        consistent across repeated calls on the same resume.
        """
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.1,
            "top_p": 0.9,
        }

        start_time = time.time()
        try:
            # Serverless providers often need to "cold start" a model that
            # hasn't been requested recently -- this can take well over a
            # minute the first time, so we give it a generous timeout.
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        except requests.exceptions.Timeout:
            if _retry:
                # One retry: the model is very likely warm now after the
                # first (timed-out) request triggered its cold start.
                return self.ask_llm(prompt, _retry=False)
            raise RuntimeError(
                "Hugging Face API timed out twice in a row. The model may be "
                "cold-starting on the provider's side -- wait a minute and try again."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Could not reach Hugging Face API: {e}")

        elapsed = time.time() - start_time
        print(f"[DEBUG] HF API response time: {elapsed:.2f}s")

        if response.status_code >= 400:
            # Surface HF's actual error body instead of a generic "Bad Request".
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(
                f"Hugging Face API error {response.status_code} for model "
                f"'{MODEL_NAME}': {detail}"
            )

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Unexpected LLM response format: {e}")

    # ------------------------------------------------------------------
    # JSON parsing (never raises -- always returns a usable dict)
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_markdown_fences(raw_text):
        """Strip ```json ... ``` or plain ``` ... ``` fences if present."""
        text = raw_text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.lower().startswith("json"):
                    text = text[4:]
        return text.strip()

    @staticmethod
    def _safe_json_parse(raw_text):
        """
        Robustly extracts a {"skills": [...], "experience": [...], "education": [...]}
        object from an LLM response. Handles:
          - markdown code fences
          - leading/trailing explanation text
          - extra text after the JSON ("Extra data" errors)
          - multiple JSON blocks in one response (picks the first valid one
            that actually looks like our expected shape)
          - completely invalid/unparsable output

        Never raises -- falls back to EMPTY_EXTRACTION on any failure, so a
        single bad LLM response can't crash the whole match_resume pipeline.
        """
        if not raw_text:
            return dict(EMPTY_EXTRACTION)

        text = MatchService._strip_markdown_fences(raw_text)
        decoder = json.JSONDecoder()
        idx = 0

        while idx < len(text):
            start = text.find("{", idx)
            if start == -1:
                break
            try:
                obj, end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                idx = start + 1
                continue

            if isinstance(obj, dict) and "skills" in obj:
                obj.setdefault("skills", [])
                obj.setdefault("experience", [])
                obj.setdefault("education", [])
                # Guard against null values in case the model returns
                # "skills": null instead of an empty list.
                for key in ("skills", "experience", "education"):
                    if obj[key] is None:
                        obj[key] = []
                return obj

            idx = start + max(end, 1)

        print(f"[WARN] Could not parse expected JSON from LLM response. Preview: {raw_text[:200]!r}")
        return dict(EMPTY_EXTRACTION)

    # ------------------------------------------------------------------
    # PDF text extraction
    # ------------------------------------------------------------------
    def extract_text(self, pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        print(f"[DEBUG] Resume length: {len(text)} chars. Preview: {text[:200]!r}")

        if len(text.strip()) < MIN_RESUME_TEXT_LENGTH:
            raise RuntimeError(
                "Could not extract readable text from this PDF. It's likely a "
                "scanned/image-based resume rather than one with selectable "
                "text -- try exporting it from Word/Google Docs as PDF instead, "
                "or use an OCR tool first."
            )
        return text

    # ------------------------------------------------------------------
    # LLM-based extraction
    # ------------------------------------------------------------------
    def extract_resume(self, resume_text):
        prompt = f"""You are an ATS (Applicant Tracking System) AI.

Analyze the resume below and extract information into ONE JSON object.

Rules:
- Return ONLY valid JSON. No markdown. No code blocks. No explanations.
- Extract ALL technical skills mentioned, across these categories:
  Programming Languages, Frameworks, Libraries, Databases, Machine Learning,
  Deep Learning, Computer Vision, NLP, LLMs, Cloud Technologies,
  Operating Systems, Development Tools.
- Never miss a skill that is explicitly mentioned in the text.
- Use standard, commonly recognized skill names (e.g. "FastAPI" not "Fast Api").
- Remove duplicate skills.
- Return experience as a list of short strings (role/project + duration if present).
- Return education as a list of short strings (degree + institution if present).

Return EXACTLY this JSON shape and nothing else:
{{
  "skills": [],
  "experience": [],
  "education": []
}}

Resume:
{resume_text}
"""
        response = self.ask_llm(prompt)
        return self._safe_json_parse(response)

    def extract_job_description(self, jd):
        prompt = f"""You are an ATS (Applicant Tracking System) AI.

Analyze the job description below and extract information into ONE JSON object.

Rules:
- Return ONLY valid JSON. No markdown. No code blocks. No explanations.
- Extract ALL required/preferred technical skills, across these categories:
  Programming Languages, Frameworks, Libraries, Databases, Machine Learning,
  Deep Learning, Computer Vision, NLP, LLMs, Cloud Technologies,
  Operating Systems, Development Tools.
- Never miss a skill that is explicitly mentioned in the text.
- Use standard, commonly recognized skill names (e.g. "FastAPI" not "Fast Api").
- Remove duplicate skills.
- Return experience as a list of short strings describing required experience.
- Return education as a list of short strings describing required education.

Return EXACTLY this JSON shape and nothing else:
{{
  "skills": [],
  "experience": [],
  "education": []
}}

Job Description:
{jd}
"""
        response = self.ask_llm(prompt)
        return self._safe_json_parse(response)

    # ------------------------------------------------------------------
    # Skill normalization
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_skills(skills):
        """
        Cleans and standardizes a list of raw skill strings:
          - trims whitespace, collapses internal multi-spaces
          - ignores case for comparison
          - maps common aliases to a standard name (see SKILL_ALIASES)
          - removes duplicates while preserving first-seen order
        """
        if not skills:
            return []

        normalized = []
        seen = set()
        for raw in skills:
            if not raw or not isinstance(raw, str):
                continue
            cleaned = " ".join(raw.strip().split())
            if not cleaned:
                continue
            standard = SKILL_ALIASES.get(cleaned.lower(), cleaned)
            dedup_key = standard.lower()
            if dedup_key not in seen:
                seen.add(dedup_key)
                normalized.append(standard)
        return normalized

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def calculate_score(self, resume_skills, jd_skills):
        """
        Normalizes both skill lists, then compares case/space-insensitively.
        Returns (score, matched_skills, missing_skills, resume_norm, jd_norm).
        """
        resume_norm = self.normalize_skills(resume_skills)
        jd_norm = self.normalize_skills(jd_skills)

        resume_lookup = {s.lower(): s for s in resume_norm}
        jd_lookup = {s.lower(): s for s in jd_norm}

        matched_keys = set(resume_lookup) & set(jd_lookup)
        missing_keys = set(jd_lookup) - set(resume_lookup)

        matched = sorted(jd_lookup[k] for k in matched_keys)
        missing = sorted(jd_lookup[k] for k in missing_keys)

        score = (len(matched) / max(len(jd_lookup), 1)) * 100
        return round(score, 2), matched, missing, resume_norm, jd_norm

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------
    def generate_suggestion(self, resume_text, jd_text, missing_skills=None):
        """
        Plain-text response on purpose -- this is NOT JSON, do not parse it
        with _safe_json_parse / json.loads.
        """
        missing_str = ", ".join(missing_skills) if missing_skills else "none identified"

        prompt = f"""Compare this resume and job description as an ATS expert.

Missing skills identified: {missing_str}

Give EXACTLY 5 bullet points of concrete, actionable suggestions to improve
the resume for this specific job. Cover a mix of:
- How to address the missing skills above (add a project, course, or reword existing experience)
- Resume structure/content improvements
- ATS optimization (keyword placement, formatting)
- Professional keywords to add

Respond in plain text bullet points only. Do NOT return JSON. Do NOT use markdown headers.

Resume:
{resume_text}

Job Description:
{jd_text}
"""
        return self.ask_llm(prompt)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def match_resume(self, pdf_path, jd):
        resume_text = self.extract_text(pdf_path)

        resume_json = self.extract_resume(resume_text)
        jd_json = self.extract_job_description(jd)

        print(f"[DEBUG] Resume JSON: {resume_json}")
        print(f"[DEBUG] JD JSON: {jd_json}")
        print(f"[DEBUG] Resume skills (raw): {resume_json.get('skills', [])}")
        print(f"[DEBUG] JD skills (raw): {jd_json.get('skills', [])}")

        score, matched, missing, resume_norm, jd_norm = self.calculate_score(
            resume_json.get("skills", []), jd_json.get("skills", [])
        )

        print(f"[DEBUG] Normalized resume skills: {resume_norm}")
        print(f"[DEBUG] Normalized JD skills: {jd_norm}")
        print(f"[DEBUG] Matched skills: {matched}")
        print(f"[DEBUG] Missing skills: {missing}")
        print(f"[DEBUG] Final match score: {score}")

        suggestion = self.generate_suggestion(resume_text, jd, missing_skills=missing)

        return {
            "match_score": score,
            "resume_skills": resume_norm,
            "jd_skills": jd_norm,
            "matched_skills": matched,
            "missing_skills": missing,
            "suggestion": suggestion,
        }
