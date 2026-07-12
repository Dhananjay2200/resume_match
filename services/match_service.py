

import json
import time
import difflib
import pdfplumber
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Auto-detect: use GPU if one is actually present, otherwise fall back to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"[INFO] Loading {MODEL_NAME} | device={DEVICE} | dtype={DTYPE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
print("[INFO] Model loaded.")

if DEVICE == "cpu":
    print(
        "[WARN] Running on CPU -- generation will be noticeably slower than "
        "on GPU. This is expected on a machine without a CUDA GPU."
    )

MIN_RESUME_TEXT_LENGTH = 30
EMPTY_EXTRACTION = {"skills": [], "experience": [], "education": []}

SKILL_ALIASES = {
    "fast api": "FastAPI", "fastapi": "FastAPI",
    "py torch": "PyTorch", "pytorch": "PyTorch",
    "ml": "Machine Learning", "machine learning": "Machine Learning",
    "dl": "Deep Learning", "deep learning": "Deep Learning",
    "llms": "LLM", "llm": "LLM",
    "large language model": "LLM", "large language models": "LLM",
    "cv": "Computer Vision", "computer vision": "Computer Vision",
    "nlp": "NLP", "natural language processing": "NLP",
    "react js": "React", "reactjs": "React", "react": "React",
    "node js": "Node.js", "nodejs": "Node.js", "node.js": "Node.js",
    "mongo db": "MongoDB", "mongodb": "MongoDB",
    "tensor flow": "TensorFlow", "tensorflow": "TensorFlow",
    "postgre sql": "PostgreSQL", "postgresql": "PostgreSQL",
    "sql server": "SQL Server",
    "hugging face": "Hugging Face", "huggingface": "Hugging Face",
    "hugging-face": "Hugging Face",
    "aws": "AWS", "gcp": "GCP", "google cloud": "GCP",
    "ci/cd": "CI/CD", "cicd": "CI/CD",
}


def _run_generation(prompt, max_new_tokens=800):
    """
    Runs a single generation call on whichever device was auto-detected
    at startup (DEVICE). No Hugging Face Spaces / ZeroGPU machinery --
    just a plain local model.generate() call.
    """
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time
    print(f"[DEBUG] Generation time ({DEVICE}): {elapsed:.2f}s")

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


class MatchService:
    
    # LLM call
    
    def ask_llm(self, prompt, max_new_tokens=800):
        try:
            return _run_generation(prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            raise RuntimeError(f"Local model generation failed: {e}")

    
    # JSON parsing (never raises -- always returns a usable dict)
    
    @staticmethod
    def _strip_markdown_fences(raw_text):
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
        object from an LLM response. Never raises -- falls back to
        EMPTY_EXTRACTION on any failure.
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
                for key in ("skills", "experience", "education"):
                    if obj[key] is None:
                        obj[key] = []
                return obj

            idx = start + max(end, 1)

        print(f"[WARN] Could not parse expected JSON from LLM response. Preview: {raw_text[:200]!r}")
        return dict(EMPTY_EXTRACTION)

    
    # PDF text extraction
    
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

    
    # LLM-based extraction
    
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
    @staticmethod
    def _skills_are_related(skill_a, skill_b):
        """
        Looser match for skills that mean the same thing but are worded
        differently (e.g. resume says 'Deep Q-Learning', JD says
        'Reinforcement Learning'). Two skills are considered related if:
          - one is a substring of the other (e.g. 'SQL' in 'SQL Server'), or
          - they share a significant word overlap, or
          - they're textually similar (typo/phrasing tolerance)
        """
        a, b = skill_a.lower(), skill_b.lower()
        if a == b:
            return True
        if a in b or b in a:
            return True

        a_words = set(a.replace("-", " ").replace("/", " ").split())
        b_words = set(b.replace("-", " ").replace("/", " ").split())
        # Ignore filler words so "Machine Learning" vs "Learning Models"
        # doesn't false-match on "Learning" alone.
        filler = {"and", "or", "with", "the", "of", "for", "a"}
        a_words -= filler
        b_words -= filler
        if a_words and b_words and (a_words & b_words):
            overlap_ratio = len(a_words & b_words) / min(len(a_words), len(b_words))
            if overlap_ratio >= 0.5:
                return True

        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85

    def calculate_score(self, resume_skills, jd_skills):
        resume_norm = self.normalize_skills(resume_skills)
        jd_norm = self.normalize_skills(jd_skills)

        resume_lookup = {s.lower(): s for s in resume_norm}
        jd_lookup = {s.lower(): s for s in jd_norm}

        # Pass 1: exact match (after normalization).
        exact_matched_keys = set(resume_lookup) & set(jd_lookup)
        remaining_jd_keys = set(jd_lookup) - exact_matched_keys
        remaining_resume_keys = set(resume_lookup) - exact_matched_keys

        matched = {jd_lookup[k] for k in exact_matched_keys}

        # Pass 2: fuzzy/substring match for whatever didn't match exactly,
        # so near-equivalent skills worded differently still count.
        still_missing_keys = set()
        for jd_key in remaining_jd_keys:
            found = False
            for resume_key in remaining_resume_keys:
                if self._skills_are_related(jd_key, resume_key):
                    matched.add(jd_lookup[jd_key])
                    found = True
                    break
            if not found:
                still_missing_keys.add(jd_key)

        missing = sorted(jd_lookup[k] for k in still_missing_keys)
        matched = sorted(matched)

        score = (len(matched) / max(len(jd_lookup), 1)) * 100
        return round(score, 2), matched, missing, resume_norm, jd_norm

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------
    def generate_suggestion(self, resume_text, jd_text, missing_skills=None):
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
        return self.ask_llm(prompt, max_new_tokens=500)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def match_resume(self, pdf_path, jd):
        resume_text = self.extract_text(pdf_path)

        resume_json = self.extract_resume(resume_text)
        jd_json = self.extract_job_description(jd)

        print(f"[DEBUG] Resume JSON: {resume_json}")
        print(f"[DEBUG] JD JSON: {jd_json}")

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
