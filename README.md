# 📄 Resume ↔ Job Description Matcher

An ATS-style tool that compares a resume (PDF) against a job description and
returns a match score, matched/missing skills, and AI-generated suggestions
to improve the resume. Built with **Gradio** + **Qwen2.5-3B-Instruct**,
running fully locally (no external API calls, no billing, no rate limits).

## ✨ Features

- Upload a resume PDF and paste a job description
- Extracts skills, experience, and education from both using a local LLM
- Normalizes skill names (e.g. "Fast API" / "FastAPI" / "ML" / "Machine
  Learning" are recognized as the same skill)
- Calculates a percentage match score, with fuzzy matching for
  differently-worded but related skills (e.g. "Deep Q-Learning" vs
  "Reinforcement Learning")
- Lists matched and missing skills
- Generates 5 concrete, actionable suggestions to improve the resume for
  that specific role
- **Automatically uses GPU if available, falls back to CPU otherwise** --
  no configuration needed

## 🗂️ Project Structure

```
.
├── app.py                     # Gradio UI
├── services/
│   ├── __init__.py
│   └── match_service.py       # PDF parsing + local LLM inference + scoring
├── requirements.txt
└── README.md
```

## 🔧 Setup

### 1. Install dependencies

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

### 2. (Optional but recommended) Install GPU-enabled PyTorch

`requirements.txt` installs a default PyTorch build, which on many systems
is CPU-only even if a GPU is present. To actually use your GPU, install the
CUDA-matched build **before** or **instead of** the default one:

```bash
# Example for CUDA 12.1 -- check https://pytorch.org/get-started/locally/
# for the exact command matching your GPU driver / CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Without this step, the app still works -- it just runs on CPU (slower).

### 3. Run

```bash
python app.py
```

The app launches at `http://localhost:7860`.

## ⚙️ How device selection works

No flags or environment variables needed. On startup, `match_service.py`
checks `torch.cuda.is_available()` once:

- **GPU found** → model loads on GPU in `bfloat16` (faster).
- **No GPU found** → model loads on CPU in `float32` (slower, but fully
  functional).

You'll see which mode it picked in the console/logs on startup:
```
[INFO] Loading Qwen/Qwen2.5-3B-Instruct | device=cuda | dtype=bfloat16
```
or
```
[INFO] Loading Qwen/Qwen2.5-3B-Instruct | device=cpu | dtype=float32
[WARN] Running on CPU -- generation will be noticeably slower than on GPU.
```

This makes the same code portable across a personal laptop, a company
server with a GPU, or a plain CPU-only machine, with zero code changes.

## 🧪 Sample Job Description (for quick testing)

Paste this into the "Job Description" box along with any resume PDF to try
the app out:

```
Machine Learning Engineer (AI/ML)

We are seeking a Machine Learning Engineer to design, train, and deploy
deep learning models for real-world applications. You will work across the
full ML lifecycle — from data preprocessing to model deployment and
monitoring in production.

Responsibilities:
- Design, train, and fine-tune deep learning models using PyTorch
- Build and experiment with reinforcement learning algorithms for
  decision-making systems
- Develop and deploy ML models as APIs using FastAPI
- Work with large language models (LLMs) and inference APIs (Hugging Face,
  OpenAI, etc.)
- Build data pipelines for model training and evaluation
- Optimize model performance, latency, and resource usage for production
- Collaborate with backend engineers to integrate ML models into existing
  systems
- Write clean, testable Python code and maintain reproducible experiments

Requirements:
- Strong hands-on experience with PyTorch and deep learning fundamentals
- Understanding of reinforcement learning concepts (Q-learning, policy
  gradients, reward shaping)
- Proficiency in Python and standard ML libraries (NumPy, Pandas,
  scikit-learn)
- Experience deploying ML models via APIs (FastAPI, Flask, or similar)
- Familiarity with Hugging Face models, transformers, and inference
  providers
- Experience with SQL/relational databases for data storage
- Understanding of model evaluation metrics and experiment tracking
- Bachelor's degree in Computer Science, IT, or related field (or
  equivalent practical project experience)

Nice to have:
- Experience deploying apps on Hugging Face Spaces or similar platforms
- Exposure to Gradio or Streamlit for building ML demo UIs
- Familiarity with financial systems, reconciliation engines, or
  transaction-based data
- Contributions to open-source ML projects or a public GitHub portfolio
```

## ⚠️ Notes & Limitations

- First run downloads Qwen2.5-3B-Instruct (~6GB) from Hugging Face Hub --
  this can take a few minutes depending on your connection. Subsequent
  runs load from the local cache and start much faster.
- CPU inference is functional but noticeably slower than GPU (expect
  seconds-to-tens-of-seconds per LLM call, vs. sub-second to a couple
  seconds on GPU).
- Match score is based on skill-name overlap (with normalization + fuzzy
  matching) extracted by the LLM; it's a helpful signal, not a substitute
  for human review of a resume.
- PDF text extraction relies on `pdfplumber`; scanned/image-only PDFs (no
  selectable text) will not extract correctly -- the app will raise a
  clear error in that case rather than silently failing.

## 📄 License

Add your preferred license here (e.g. MIT).
