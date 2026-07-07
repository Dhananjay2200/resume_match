---
title: Resume JD Matcher
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 📄 Resume ↔ Job Description Matcher

An ATS-style tool that compares a resume (PDF) against a job description and
returns a match score, matched/missing skills, and AI-generated suggestions
to improve the resume. Built with **Gradio** + **Hugging Face Inference
Providers** (Qwen2.5-3B-Instruct via Featherless AI).

## ✨ Features

- Upload a resume PDF and paste a job description
- Extracts skills, experience, and education from both using an LLM
- Calculates a percentage match score based on skill overlap
- Lists matched and missing skills
- Generates concrete suggestions to improve the resume for that specific role

## 🗂️ Project Structure

```
.
├── app.py                     # Gradio UI
├── services/
│   └── match_service.py       # PDF parsing + LLM calls + scoring logic
├── requirements.txt
└── README.md
```

## 🔧 Setup

### 1. Get a Hugging Face token

Create a fine-grained token at https://huggingface.co/settings/tokens with
**"Make calls to Inference Providers"** permission enabled.

### 2. Run locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt

export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"   # Windows (PowerShell): $env:HF_TOKEN="hf_..."
python app.py
```

The app will launch at `http://localhost:7860`.

## 🤗 Hugging Face Space

Live demo: [Hugging Face Space](#) https://huggingface.co/spaces/Dk22000/Resume_Matche

## 🧪 Sample Job Description (for quick testing)

Paste this into the "Job Description" box along with any resume PDF to try
the app out:

```
Software Engineer — Backend (Python)

We are looking for a Backend Software Engineer to join our growing
engineering team. You will design, build, and maintain scalable APIs and
services that power our core product.

Responsibilities:
- Design and implement RESTful APIs using FastAPI or Django
- Write clean, well-tested Python code
- Work with relational databases (PostgreSQL, SQL Server) and ORMs
- Optimize application performance and troubleshoot production issues
- Collaborate with frontend, DevOps, and product teams
- Participate in code reviews and technical design discussions

Requirements:
- 1-3 years of experience with Python
- Solid understanding of REST API design
- Experience with SQL and relational databases
- Familiarity with Git and CI/CD pipelines
- Knowledge of Docker and containerized deployments is a plus
- Experience with cloud platforms (AWS, Azure, or GCP) is a plus
- Understanding of asynchronous programming (asyncio) is a plus
- Strong problem-solving skills and attention to detail

Nice to have:
- Experience with FastAPI, SQLAlchemy, or Pydantic
- Exposure to machine learning or AI-powered applications
- Familiarity with testing frameworks like pytest
```

## ⚠️ Notes & Limitations

- Uses Hugging Face's free serverless inference — the first request after a
  period of inactivity may take up to ~1-2 minutes while the model
  "cold-starts" on the provider's side. Subsequent requests are much faster.
- Match score is based purely on skill-name overlap extracted by the LLM; it
  is not a substitute for human review of a resume.
- PDF text extraction relies on `pdfplumber`; scanned/image-only PDFs (no
  selectable text) will not extract correctly.

## 📄 License

Add your preferred license here (e.g. MIT).
