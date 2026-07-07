import os
import json
import pdfplumber
import requests

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable is not set. "
        "Add it as a secret in your HF Space settings."
    )

API_URL = "https://router.huggingface.co/v1/chat/completions"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct:featherless-ai"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}


class MatchService:
    def ask_llm(self, prompt, _retry=True):
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
        }
        try:
           
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        except requests.exceptions.Timeout:
            if _retry:
                
                return self.ask_llm(prompt, _retry=False)
            raise RuntimeError(
                "Hugging Face API timed out twice in a row. The model may be "
                "cold-starting on the provider's side -- wait a minute and try again."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Could not reach Hugging Face API: {e}")

        if response.status_code >= 400:
            
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(
                f"Hugging Face API error {response.status_code} for model "
                f"'{MODEL_NAME}': {detail}"
            )

        return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _clean_json(raw_text):
        
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.lower().startswith("json"):
                text = text[4:]
        return text.strip()

    @staticmethod
    def _extract_json(raw_text):
        
        text = MatchService._clean_json(raw_text)
        decoder = json.JSONDecoder()
        
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in model response: {raw_text!r}")
        obj, _ = decoder.raw_decode(text[start:])
        return obj

    def extract_text(self, pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def extract_resume(self, resume_text):
        prompt = f"""
You are an ATS AI.

Analyze the following resume and extract the information below.
Respond with ONLY valid JSON, no markdown formatting, no explanation, in exactly this shape:
{{
  "skills": [],
  "experience": [],
  "education": []
}}

Resume:
{resume_text}
"""
        response = self.ask_llm(prompt)
        return self._extract_json(response)

    def extract_job_description(self, jd):
        prompt = f"""
You are an ATS AI.

Analyze the following job description and extract the information below.
Respond with ONLY valid JSON, no markdown formatting, no explanation, in exactly this shape:
{{
  "skills": [],
  "experience": [],
  "education": []
}}

Job Description:
{jd}
"""
        response = self.ask_llm(prompt)
        return self._extract_json(response)

    def calculate_score(self, resume_skills, jd_skills):
        resume_set = {s.strip().lower() for s in resume_skills}
        jd_set = {s.strip().lower() for s in jd_skills}

        matched = list(resume_set & jd_set)
        missing = list(jd_set - resume_set)

        score = (len(matched) / max(len(jd_set), 1)) * 100
        return score, matched, missing

    def generate_suggestion(self, resume_text, jd_text):
    
        prompt = f"""
Compare this resume and job description, then give 3-5 concrete, actionable
suggestions to improve the resume so it better matches the job description.
Respond in plain text (short bullet points), not JSON.

Resume:
{resume_text}

Job Description:
{jd_text}
"""
        return self.ask_llm(prompt)

    def match_resume(self, pdf_path, jd):
        resume_text = self.extract_text(pdf_path)
        resume = self.extract_resume(resume_text)
        jd_data = self.extract_job_description(jd)

        score, matched, missing = self.calculate_score(
            resume.get("skills", []), jd_data.get("skills", [])
        )
        suggestion = self.generate_suggestion(resume_text, jd)

        return {
            "match_score": round(score, 2),
            "matched_skills": matched,
            "missing_skills": missing,
            "suggestion": suggestion,
        }