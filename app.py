import gradio as gr
from services.match_service import MatchService

matcher = MatchService()


def match_resume(resume_file, job_desc):
    if resume_file is None:
        return "⚠️ Please upload a resume PDF.", "", "", ""
    if not job_desc or not job_desc.strip():
        return "⚠️ Please paste a job description.", "", "", ""

    try:
        result = matcher.match_resume(resume_file.name, job_desc)
    except Exception as e:
        return f"❌ Error: {e}", "", "", ""

    score = f"{result['match_score']}%"
    matched = ", ".join(result["matched_skills"]) or "None"
    missing = ", ".join(result["missing_skills"]) or "None"
    suggestion = result["suggestion"]

    return score, matched, missing, suggestion


with gr.Blocks(title="Resume ↔ JD Matcher") as demo:
    gr.Markdown(
        "# 📄 Resume ↔ Job Description Matcher\n"
        "Upload a resume (PDF) and paste a job description to get an "
        "ATS-style match score, skill gaps, and improvement suggestions.\n\n"
        "_Powered by Qwen2.5-3B-Instruct via Hugging Face_"
    )

    with gr.Row():
        resume_input = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
        jd_input = gr.Textbox(
            label="Job Description",
            lines=10,
            placeholder="Paste the job description here...",
        )

    submit_btn = gr.Button("Match Resume", variant="primary")

    score_output = gr.Textbox(label="Match Score")
    matched_output = gr.Textbox(label="Matched Skills")
    missing_output = gr.Textbox(label="Missing Skills")
    suggestion_output = gr.Textbox(label="Suggestions", lines=8)

    submit_btn.click(
        fn=match_resume,
        inputs=[resume_input, jd_input],
        outputs=[score_output, matched_output, missing_output, suggestion_output],
    )

if __name__ == "__main__":
    demo.launch()
