import re
import html
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Resume Feedback Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- CUSTOM UI ----------------
st.markdown(
    """
    <style>
    :root {
        --accent: #7c8cff;
        --accent2: #35c6b2;
        --surface: #171d2b;
        --surface2: #20283a;
        --muted: #a8b2c7;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(84, 105, 255, .14), transparent 30%),
            radial-gradient(circle at 95% 5%, rgba(53, 198, 178, .09), transparent 26%),
            #0d1220;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(255,255,255,.08);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.55rem 1.8rem;
        border: 1px solid rgba(143, 157, 255, .25);
        border-radius: 20px;
        background: linear-gradient(115deg, rgba(69, 88, 190, .28), rgba(28, 166, 151, .12));
        margin-bottom: 1.2rem;
    }

    .hero-kicker {
        color: #9eafff;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }

    .hero h1 {
        color: #f7f8ff;
        font-size: 2.25rem;
        line-height: 1.15;
        margin: 0;
        font-weight: 800;
    }

    .hero p {
        color: #c0c9dc;
        margin: .65rem 0 0 0;
        font-size: 1rem;
    }

    .privacy {
        border: 1px solid rgba(53,198,178,.28);
        background: rgba(24, 126, 111, .13);
        color: #b9f4e8;
        padding: .85rem 1rem;
        border-radius: 12px;
        margin: .8rem 0 1.5rem 0;
        font-size: .9rem;
    }

    .section-label {
        color: #9eafff;
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .25rem;
    }

    .panel {
        background: linear-gradient(145deg, rgba(31,40,59,.96), rgba(22,29,44,.96));
        border: 1px solid rgba(160,174,205,.14);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        margin-bottom: .8rem;
    }

    .panel h3 {
        color: #f2f5ff;
        margin: 0 0 .35rem 0;
        font-size: 1.05rem;
    }

    .panel p {
        color: #b6c0d3;
        margin: 0;
        font-size: .9rem;
    }

    .skill-chip {
        display: inline-block;
        padding: .38rem .7rem;
        margin: .25rem .28rem .25rem 0;
        border-radius: 999px;
        font-size: .84rem;
        font-weight: 650;
    }

    .skill-good {
        color: #aaf0d9;
        background: rgba(28, 167, 126, .16);
        border: 1px solid rgba(28, 167, 126, .34);
    }

    .skill-missing {
        color: #ffd3a8;
        background: rgba(218, 130, 55, .13);
        border: 1px solid rgba(218, 130, 55, .34);
    }

    .feedback-card {
        padding: 1rem 1.15rem;
        border-radius: 13px;
        background: #1b2435;
        border: 1px solid rgba(148,163,184,.15);
        border-left: 4px solid #7c8cff;
        margin-bottom: .7rem;
    }

    .feedback-card h4 {
        color: #f4f6ff;
        margin: 0 0 .35rem 0;
        font-size: 1rem;
    }

    .feedback-status {
        color: #9eafff;
        font-size: .82rem;
        font-weight: 700;
        margin-bottom: .25rem;
    }

    .feedback-card p {
        color: #c0c9d9;
        margin: 0;
        font-size: .9rem;
    }

    .feedback-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .8rem;
        flex-wrap: wrap;
        margin-bottom: .55rem;
    }
    .feedback-badge {
        display: inline-block;
        padding: .25rem .65rem;
        border-radius: 999px;
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .02em;
    }
    .badge-good {
        color: #aaf0d9;
        background: rgba(28,167,126,.16);
        border: 1px solid rgba(28,167,126,.38);
    }
    .badge-warn {
        color: #ffe0a6;
        background: rgba(218,160,55,.14);
        border: 1px solid rgba(218,160,55,.38);
    }
    .badge-bad {
        color: #ffc2c2;
        background: rgba(220,75,75,.13);
        border: 1px solid rgba(220,75,75,.38);
    }
    .feedback-label {
        color: #9eafff;
        font-size: .75rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin: .55rem 0 .2rem 0;
    }
    .plan-item {
        background: #1b2435;
        border: 1px solid rgba(148,163,184,.15);
        border-left: 4px solid #7c8cff;
        border-radius: 12px;
        padding: .8rem 1rem;
        margin-bottom: .6rem;
    }
    .plan-item strong { color: #f4f6ff; }
    .plan-item p { color: #c0c9d9; margin: .2rem 0 0 0; font-size: .9rem; }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1d273a, #151e2e);
        border: 1px solid rgba(148,163,184,.15);
        padding: 1rem 1.1rem;
        border-radius: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #aebbd2;
    }

    div[data-testid="stMetricValue"] {
        color: #f7f8ff;
    }

    .small-note {
        color: #9ca9bf;
        font-size: .82rem;
    }

    .stButton > button[kind="primary"] {
        border: 0;
        border-radius: 10px;
        padding: .65rem 1.35rem;
        font-weight: 750;
        background: linear-gradient(100deg, #687cff, #4c9eea);
    }

    .stButton > button[kind="primary"]:hover {
        border: 0;
        filter: brightness(1.08);
    }

    hr {
        border-color: rgba(148,163,184,.16);
    }

    /* Final UI polish: clearer controls, cards and responsive spacing */
    .stApp { color: #edf2ff; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    [data-testid="stFileUploader"] section {
        border: 1px dashed rgba(143,157,255,.55);
        border-radius: 14px;
        background: rgba(91,108,190,.08);
    }
    [data-testid="stTextArea"] textarea {
        background: #111a2a;
        color: #edf2ff;
        border: 1px solid rgba(148,163,184,.24);
        border-radius: 12px;
        line-height: 1.55;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #8494ff;
        box-shadow: 0 0 0 1px #8494ff;
    }
    [data-testid="stRadio"] label { color: #d7def0; }
    [data-testid="stMetric"] {
        min-height: 118px;
        transition: transform .18s ease, border-color .18s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(124,140,255,.48);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(148,163,184,.18) !important;
        border-radius: 16px !important;
        background: rgba(22,30,46,.35);
    }
    .stDownloadButton > button {
        border: 1px solid rgba(53,198,178,.45);
        border-radius: 11px;
        font-weight: 750;
        padding: .7rem 1.1rem;
        background: linear-gradient(100deg, rgba(30,145,133,.25), rgba(70,104,190,.25));
        color: #eafffb;
    }
    .stDownloadButton > button:hover { border-color: #35c6b2; color: white; }
    [data-testid="stCheckbox"] label { color: #d4dced; }
    [data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg,#687cff,#35c6b2); }
    @media (max-width: 760px) {
        .block-container { padding: 1rem .8rem 2rem .8rem; }
        .hero { padding: 1.2rem; }
        .hero h1 { font-size: 1.65rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    st.markdown("### 📄 Resume Feedback Assistant")
    st.caption("INTRODUCTION TO MACHINE LEARNING")
    st.markdown("<span style='background:rgba(53,198,178,.14);color:#9cebdc;padding:5px 10px;border-radius:20px;font-size:12px;font-weight:700;'>● ANALYSIS WORKSPACE</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**Analysis modules**")
    st.markdown("🧩 Resume structure")
    st.markdown("🛠️ Skills matching")
    st.markdown("🤖 TF-IDF similarity")
    st.divider()
    st.caption(
        "This application provides preliminary resume feedback. "
        "It does not make hiring decisions or guarantee interview outcomes."
    )


# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">AI-assisted career toolkit · ML project</div>
        <h1>📄 Resume Feedback Assistant</h1>
        <p>Understand your resume better. Compare relevant content with a target role and discover practical areas to improve.</p>
    </div>
    <div class="privacy">
        🔒 <b>Privacy note:</b> Resume text is processed in this app session for feedback.
        Avoid entering sensitive personal information that is not needed for analysis.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- INPUT AREA ----------------
st.markdown('<div class="section-label">Step 01 · Your documents</div>', unsafe_allow_html=True)
st.header("Submit your resume")

input_col, job_col = st.columns([1, 1], gap="large")

resume_text = ""

with input_col:
    with st.container(border=True):
        st.subheader("📝 Resume")
        input_method = st.radio(
            "Choose resume input method",
            ["Paste resume text", "Upload PDF"],
            horizontal=True,
            key="resume_input_method",
        )

        if input_method == "Paste resume text":
            resume_text = st.text_area(
                "Resume content",
                height=260,
                placeholder=(
                    "Education: B.Tech in Computer Science\n"
                    "Skills: Python, SQL, Machine Learning\n"
                    "Projects: Resume Feedback Assistant\n"
                    "Experience: Internship ..."
                ),
                key="resume_text_input",
            )
        else:
            uploaded_file = st.file_uploader(
                "Choose a PDF resume",
                type=["pdf"],
                help="Upload a text-based PDF of up to 20 pages.",
                key="resume_pdf_upload",
            )
            if uploaded_file is not None:
                try:
                    pdf_reader = PdfReader(BytesIO(uploaded_file.getvalue()))
                    if len(pdf_reader.pages) > 20:
                        st.error("Please upload a PDF containing no more than 20 pages.")
                    else:
                        extracted_pages = [
                            page.extract_text() or "" for page in pdf_reader.pages
                        ]
                        resume_text = "\n".join(extracted_pages)
                        if resume_text.strip():
                            st.success("PDF text extracted successfully.")
                        else:
                            st.warning(
                                "No readable text was found. This may be a scanned/image-only PDF."
                            )
                except Exception:
                    st.error("Unable to read this PDF. Please try another valid PDF.")

with job_col:
    with st.container(border=True):
        st.subheader("💼 Target job description")
        job_description = st.text_area(
            "Paste the job description",
            height=260,
            placeholder=(
                "We are looking for a Machine Learning Intern.\n"
                "Required skills: Python, Pandas, Scikit-learn,\n"
                "Machine Learning and Data Preprocessing."
            ),
            key="job_description_input",
        )


# ---------------- ANALYSIS FUNCTIONS ----------------
def analyze_resume(text):
    text_lower = text.lower()
    feedback = []

    sections = {
        "Education": ["education", "academic background", "qualification"],
        "Skills": ["skills", "technical skills", "core competencies"],
        "Projects": ["projects", "project experience", "academic projects"],
        "Experience": ["experience", "work experience", "internship"],
    }

    for section, keywords in sections.items():
        found = any(
            re.search(r"\b" + re.escape(keyword) + r"\b", text_lower)
            for keyword in keywords
        )
        if found:
            feedback.append({
                "category": section,
                "status": "Section detected",
                "suggestion": {
                    "Education": "Your education heading was found. Make sure the degree, institution and graduation year are easy to identify.",
                    "Skills": "Your skills heading was found. Keep relevant technical skills in a clear, easy-to-scan list.",
                    "Projects": "Your projects heading was found. For each project, explain what you built, which tools you used and what you contributed.",
                    "Experience": "Your experience heading was found. Include the role, organisation, dates and your main responsibilities or achievements."
                }[section],
            })
        else:
            feedback.append({
                "category": section,
                "status": "Section not detected",
                "suggestion": {
                    "Education": "No familiar education heading was found. Add an Education section with your degree, institution and year.",
                    "Skills": "No familiar skills heading was found. Add a Skills section and list the tools or abilities you genuinely have.",
                    "Projects": "No familiar projects heading was found. Add relevant academic or personal projects and describe your contribution.",
                    "Experience": "No familiar experience heading was found. If you have internships, work or relevant volunteering, add an Experience section. If not, do not invent experience."
                }[section],
            })

    word_count = len(text.split())
    if word_count < 100:
        feedback.append({
            "category": "Content length",
            "status": "Very short input",
            "suggestion": (
                "Only a small amount of text was provided. Check that you submitted the full resume and add relevant details about education, projects and experience."
            ),
        })
    elif word_count > 1000:
        feedback.append({
            "category": "Content length",
            "status": "Long input",
            "suggestion": (
                "The resume text is lengthy. Keep the most relevant information and remove repeated or unrelated details."
            ),
        })
    else:
        feedback.append({
            "category": "Content length",
            "status": "Basic length check passed",
            "suggestion": (
                "The text length check looks reasonable. Still review whether each detail is relevant to the target role."
            ),
        })

    if any(marker in text for marker in ["•", "●", "▪", "- "]):
        feedback.append({
            "category": "Presentation",
            "status": "Bullet points detected",
            "suggestion": (
                "Bullet markers were found. Keep the formatting consistent and begin experience or project points with clear action verbs where appropriate."
            ),
        })
    else:
        feedback.append({
            "category": "Presentation",
            "status": "No common bullet markers detected",
            "suggestion": (
                "No common bullet markers were found. Consider using short bullet points to make projects and responsibilities easier to scan."
            ),
        })

    return feedback, word_count


def compare_resume_with_job(resume, job_description):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(similarity) * 100, 2)


def compare_skills(resume, job_description):
    skills = [
        "python", "sql", "machine learning", "scikit-learn", "pandas", "numpy",
        "data preprocessing", "model evaluation", "deep learning", "tensorflow",
        "pytorch", "excel", "power bi", "tableau", "flask", "django", "statistics",
    ]

    resume_lower = resume.lower()
    job_lower = job_description.lower()

    required_skills = [skill for skill in skills if skill in job_lower]
    detected_skills = [skill for skill in required_skills if skill in resume_lower]
    not_detected_skills = [
        skill for skill in required_skills if skill not in resume_lower
    ]
    return detected_skills, not_detected_skills


# ---------------- ANALYZE ACTION ----------------
st.markdown('<div class="section-label">Step 02 · Run analysis</div>', unsafe_allow_html=True)

analyze_clicked = st.button("🔍 Analyze Resume", type="primary", use_container_width=False)

if analyze_clicked:
    # Clear previous results first so an invalid/empty submission cannot
    # leave an old dashboard visible below the validation message.
    for result_key in (
        "detected_skills",
        "not_detected_skills",
        "resume_feedback",
        "resume_word_count",
        "similarity_percentage",
    ):
        st.session_state.pop(result_key, None)
    st.session_state["analysis_complete"] = False

    if not resume_text.strip():
        st.error("Please paste resume text or upload a readable PDF.")
    elif len(resume_text.strip()) < 30:
        st.warning("Please provide more resume text for analysis.")
    elif not job_description.strip():
        st.warning("Please paste a job description to use the ML comparison.")
    else:
        try:
            feedback, word_count = analyze_resume(resume_text)
            similarity = compare_resume_with_job(resume_text, job_description)
            detected_skills, not_detected_skills = compare_skills(
                resume_text, job_description
            )

            st.session_state["detected_skills"] = detected_skills
            st.session_state["not_detected_skills"] = not_detected_skills
            st.session_state["resume_feedback"] = feedback
            st.session_state["resume_word_count"] = word_count
            st.session_state["similarity_percentage"] = similarity
            st.session_state["analysis_complete"] = True
        except ValueError:
            st.error(
                "The text could not be compared. Please provide a longer resume "
                "and job description with readable words."
            )


def build_analysis_pdf(resume_text, job_description, similarity, detected_skills, not_detected_skills, feedback, word_count):
    """Create a downloadable PDF summary of the current resume analysis."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Resume Feedback Analysis Report",
        author="Resume Feedback Assistant",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontSize=20,
        leading=25, textColor=colors.HexColor("#263b75"), alignment=TA_CENTER,
        spaceAfter=5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", parent=styles["Heading2"], fontSize=12,
        leading=15, textColor=colors.HexColor("#263b75"),
        spaceBefore=4 * mm, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="SmallBody", parent=styles["BodyText"], fontSize=9,
        leading=13, textColor=colors.HexColor("#333333"),
    ))

    story = [
        Paragraph("Resume Feedback Analysis Report", styles["ReportTitle"]),
        Paragraph("Generated by Resume Feedback Assistant", styles["SmallBody"]),
        Spacer(1, 5 * mm),
    ]

    summary_data = [
        ["Resume–Job Similarity", f"{similarity:.2f}%"],
        ["Words Analyzed", str(word_count)],
        ["Skills Detected", str(len(detected_skills))],
        ["Skills Not Detected", str(len(not_detected_skills))],
        ["Sections Detected", str(sum(1 for item in feedback if item["status"] == "Section detected"))],
        ["Checks Performed", str(len(feedback))],
    ]
    summary_table = Table(summary_data, colWidths=[78 * mm, 78 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f4fb")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#263b75")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5dbea")),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [Paragraph("Analysis Summary", styles["SectionHeading"]), summary_table]

    def safe_para(value):
        return html.escape(str(value)).replace("\n", "<br/>")

    story.append(Paragraph("Skills Detected", styles["SectionHeading"]))
    story.append(Paragraph(safe_para(", ".join(skill.title() for skill in detected_skills) or "None detected"), styles["SmallBody"]))
    story.append(Paragraph("Skills Not Detected", styles["SectionHeading"]))
    story.append(Paragraph(safe_para(", ".join(skill.title() for skill in not_detected_skills) or "None"), styles["SmallBody"]))

    story.append(Paragraph("Detailed Feedback", styles["SectionHeading"]))
    for item in feedback:
        story.append(Paragraph(f"<b>{safe_para(item['category'])}</b> — {safe_para(item['status'])}", styles["SmallBody"]))
        story.append(Paragraph(safe_para(item["suggestion"]), styles["SmallBody"]))
        story.append(Spacer(1, 2 * mm))

    story += [
        HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd3e1")),
        Spacer(1, 3 * mm),
        Paragraph(
            "Note: The similarity percentage represents TF-IDF text similarity only. "
            "It is not a hiring probability, validated resume-quality score, or proof of skill. "
            "Skills marked as not detected should only be added if they accurately reflect your abilities.",
            styles["SmallBody"],
        ),
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------- RESULTS DASHBOARD ----------------
if st.session_state.get("analysis_complete", False):
    detected_skills = st.session_state.get("detected_skills", [])
    not_detected_skills = st.session_state.get("not_detected_skills", [])
    feedback = st.session_state.get("resume_feedback", [])
    word_count = st.session_state.get("resume_word_count", 0)
    similarity = st.session_state.get("similarity_percentage", 0.0)

    st.divider()
    st.markdown('<div class="section-label">Step 03 · Your results</div>', unsafe_allow_html=True)
    st.header("📊 Resume Feedback Dashboard")

    # Similarity score panel
    score_col, score_info_col = st.columns([1, 2], gap="large")
    with score_col:
        st.metric("Resume–Job Description Similarity", f"{similarity:.2f}%")
        st.progress(max(0.0, min(float(similarity) / 100, 1.0)))
    with score_info_col:
        st.markdown(
            """
            <div class="panel">
                <h3>🤖 ML-based text comparison</h3>
                <p>
                    The similarity value is calculated using TF-IDF vectorization
                    and cosine similarity. It represents text similarity only;
                    it is not a hiring probability, a validated resume-quality
                    score, or proof of skill.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Summary metric cards
    st.subheader("📌 Resume statistics")
    metric_cols = st.columns(5)
    metrics = [
        ("Words analyzed", word_count),
        ("Skills detected", len(detected_skills)),
        ("Skills missing", len(not_detected_skills)),
        ("Checks performed", len(feedback)),
        (
            "Sections detected",
            sum(1 for item in feedback if item["status"] == "Section detected"),
        ),
    ]
    for col, (label, value) in zip(metric_cols, metrics):
        with col:
            st.metric(label, value)

    # Skills analysis
    st.divider()
    st.subheader("🛠️ Relevant skills analysis")
    skills_col1, skills_col2 = st.columns(2, gap="large")

    with skills_col1:
        with st.container(border=True):
            st.markdown("### ✅ Skills detected")
            if detected_skills:
                chips = "".join(
                    f'<span class="skill-chip skill-good">{html.escape(skill.title())}</span>'
                    for skill in detected_skills
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.info("No matching skills from the predefined list were detected.")

    with skills_col2:
        with st.container(border=True):
            st.markdown("### 💡 Skills not detected")
            if not_detected_skills:
                chips = "".join(
                    f'<span class="skill-chip skill-missing">{html.escape(skill.title())}</span>'
                    for skill in not_detected_skills
                )
                st.markdown(chips, unsafe_allow_html=True)
                st.caption(
                    "These skills were not detected in the resume text. "
                    "Add them only if they accurately reflect your abilities."
                )
            else:
                st.success("All identified job-description skills were detected in the resume.")

    # Simple chart
    st.subheader("📈 Skills match overview")
    chart_data = {
        "Detected": len(detected_skills),
        "Not detected": len(not_detected_skills),
    }
    st.bar_chart(chart_data, horizontal=True)

    # Detailed feedback - plain-language status and action
    st.divider()
    st.subheader("🧾 Detailed resume feedback")
    st.caption("This section explains what the app found in your resume and what you can improve.")

    status_display = {
        "Section detected": ("Found", "badge-good"),
        "Section not detected": ("Not found", "badge-bad"),
        "Very short input": ("Needs attention", "badge-bad"),
        "Long input": ("Review length", "badge-warn"),
        "Basic length check passed": ("Looks reasonable", "badge-good"),
        "Bullet points detected": ("Found", "badge-good"),
        "No common bullet markers detected": ("Could be clearer", "badge-warn"),
    }

    for item in feedback:
        category = html.escape(item["category"])
        suggestion = html.escape(item["suggestion"])
        label, badge_class = status_display.get(
            item["status"], (item["status"], "badge-warn")
        )
        st.markdown(
            f"""
            <div class="feedback-card">
                <div class="feedback-top">
                    <h4>{category}</h4>
                    <span class="feedback-badge {badge_class}">{html.escape(label)}</span>
                </div>
                <div class="feedback-label">What this means</div>
                <p>{suggestion}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Dynamic improvement plan based on this analysis
    st.divider()
    st.subheader("🧭 Your resume improvement plan")
    st.caption("These next steps are generated from the checks above. Only add information that is true about your experience and abilities.")

    improvement_items = []
    missing_sections = [
        item["category"] for item in feedback
        if item["status"] == "Section not detected"
    ]
    if missing_sections:
        improvement_items.append((
            "High priority",
            "Add or clarify these sections",
            "The app could not find: " + ", ".join(missing_sections) +
            ". Add a section only if it is relevant to your background.",
            "badge-bad",
        ))

    if not_detected_skills:
        improvement_items.append((
            "High priority",
            "Review skills not detected",
            "The job description mentions: " +
            ", ".join(skill.title() for skill in not_detected_skills) +
            ". Include any of these only if you genuinely have that skill; use a recognised spelling in your resume.",
            "badge-bad",
        ))

    length_item = next((item for item in feedback if item["category"] == "Content length"), None)
    if length_item and length_item["status"] == "Very short input":
        improvement_items.append((
            "High priority",
            "Provide more resume detail",
            "The submitted text is very short. Check that the full resume was pasted or that the PDF text was extracted completely.",
            "badge-bad",
        ))
    elif length_item and length_item["status"] == "Long input":
        improvement_items.append((
            "Medium priority",
            "Tighten the resume",
            "Remove repeated or unrelated details and keep information relevant to the target role.",
            "badge-warn",
        ))

    presentation_item = next((item for item in feedback if item["category"] == "Presentation"), None)
    if presentation_item and presentation_item["status"] == "No common bullet markers detected":
        improvement_items.append((
            "Medium priority",
            "Make content easier to scan",
            "Use consistent bullet points for project work and responsibilities where appropriate.",
            "badge-warn",
        ))

    if not improvement_items:
        improvement_items.append((
            "Good progress",
            "Keep the resume targeted",
            "The checks did not flag any immediate structure, length or listed-skill issues. Review the wording for accuracy and tailor examples to the target role.",
            "badge-good",
        ))

    for index, (priority, title, detail, badge_class) in enumerate(improvement_items, start=1):
        st.markdown(
            f"""
            <div class="plan-item">
                <div class="feedback-top">
                    <strong>{index}. {html.escape(title)}</strong>
                    <span class="feedback-badge {badge_class}">{html.escape(priority)}</span>
                </div>
                <p>{html.escape(detail)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.success("Analysis complete. Review the suggestions before updating your resume.")

    report_pdf = build_analysis_pdf(
        resume_text=st.session_state.get("resume_text_input", "") if st.session_state.get("resume_input_method") == "Paste resume text" else resume_text,
        job_description=st.session_state.get("job_description_input", ""),
        similarity=similarity,
        detected_skills=detected_skills,
        not_detected_skills=not_detected_skills,
        feedback=feedback,
        word_count=word_count,
    )
    st.download_button(
        label="📄 Download Analysis Report (PDF)",
        data=report_pdf,
        file_name="resume_feedback_report.pdf",
        mime="application/pdf",
        type="primary",
    )

else:
    st.info("Add your resume and a target job description, then select **Analyze Resume** to view the dashboard.")


# ---------------- FOOTER ----------------
st.divider()
st.caption("Resume Feedback Assistant  ·  Introduction to Machine Learning Project  ·  TF-IDF + Cosine Similarity")
