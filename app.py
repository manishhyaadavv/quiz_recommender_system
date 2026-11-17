
import streamlit as st
import re
from pypdf import PdfReader
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Resume Feedback Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 750;
        color: #2563EB;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748B;
        margin-bottom: 25px;
    }

    .feedback-card {
        padding: 18px;
        border-radius: 12px;
        background-color: #F1F5F9;
        margin-bottom: 12px;
        border-left: 5px solid #2563EB;
        color: #1E293B;
    }

    .privacy-note {
        padding: 12px;
        border-radius: 8px;
        background-color: #ECFDF5;
        color: #065F46;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">📄 Resume Feedback Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Improve your resume with structured, actionable feedback.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="privacy-note">
🔒 <b>Privacy:</b> Your resume is processed for feedback and is not
intentionally saved to permanent application storage. Avoid submitting
sensitive personal information that is not needed for the review.
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("⚙️ Settings")

    st.write("### About this project")

    st.write(
        "Resume Feedback Assistant reviews resume structure, "
        "clarity, skills presentation and project descriptions."
    )

    st.divider()

    st.write("### Analysis categories")

    st.write("✅ Resume structure")
    st.write("✅ Skills presentation")
    st.write("✅ Project descriptions")
    st.write("✅ Content clarity")

    st.caption(
        "This tool provides feedback only. "
        "It does not make hiring decisions."
    )

# ---------------- RESUME INPUT ----------------

st.subheader("📝 Submit Your Resume")

input_method = st.radio(
    "Choose how to provide your resume:",
    ["Paste Resume Text", "Upload PDF"],
    horizontal=True
)

resume_text = ""

if input_method == "Paste Resume Text":

    resume_text = st.text_area(
        "Paste your resume below",
        height=250,
        placeholder=(
            "Example:\n"
            "Education: B.Tech in Computer Science\n"
            "Skills: Python, SQL, Machine Learning\n"
            "Projects: Resume Feedback Assistant"
        )
    )

else:

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"]
    )

    if uploaded_file is not None:

        try:
            pdf_reader = PdfReader(BytesIO(uploaded_file.getvalue()))

            if len(pdf_reader.pages) > 20:
                st.error("Please upload a PDF containing no more than 20 pages.")
            else:
                extracted_pages = [
                    page.extract_text() or ""
                    for page in pdf_reader.pages
                ]

                resume_text = "\n".join(extracted_pages)

                if not resume_text.strip():
                    st.warning(
                        "No readable text was found. "
                        "The PDF may be scanned or image-based."
                    )
                else:
                    st.success("Resume text extracted successfully!")

        except Exception:
            st.error(
                "Unable to read this PDF. Please try another valid PDF."
            )

# ---------------- ANALYSIS FUNCTION ----------------

def analyze_resume(text):

    text_lower = text.lower()

    feedback = []

    # Check for common resume sections
    sections = {
        "Education": [
            "education", "academic background", "qualification"
        ],
        "Skills": [
            "skills", "technical skills", "core competencies"
        ],
        "Projects": [
            "projects", "project experience", "academic projects"
        ],
        "Experience": [
            "experience", "work experience", "internship"
        ]
    }

    for section, keywords in sections.items():

        found = any(
            re.search(
                r"\b" + re.escape(keyword) + r"\b",
                text_lower
            )
            for keyword in keywords
        )

        if found:
            feedback.append({
                "category": section,
                "status": "Section detected",
                "suggestion": (
                    f"The {section} section appears to be present. "
                    "Check that its content is relevant and clearly written."
                )
            })

        else:
            feedback.append({
                "category": section,
                "status": "Section not detected",
                "suggestion": (
                    f"No familiar {section} heading was detected. "
                    "If relevant, consider adding this section or using "
                    "a clearer heading."
                )
            })

    # Check resume length
    word_count = len(text.split())

    if word_count < 100:
        feedback.append({
            "category": "Content length",
            "status": "Very short input",
            "suggestion": (
                "This text is quite short for a complete resume. "
                "Check that the full resume was provided."
            )
        })

    elif word_count > 1000:
        feedback.append({
            "category": "Content length",
            "status": "Long input",
            "suggestion": (
                "Review whether all information is relevant and "
                "unnecessary repetition can be removed."
            )
        })

    else:
        feedback.append({
            "category": "Content length",
            "status": "Basic length check passed",
            "suggestion": (
                "Review whether the resume includes enough relevant "
                "information without unnecessary repetition."
            )
        })

    # Check for bullet points
    if any(marker in text for marker in ["•", "●", "▪", "- "]):
        feedback.append({
            "category": "Presentation",
            "status": "Bullet points detected",
            "suggestion": (
                "Use consistent bullet formatting and start achievement "
                "statements with clear action verbs where appropriate."
            )
        })

    else:
        feedback.append({
            "category": "Presentation",
            "status": "No common bullet markers detected",
            "suggestion": (
                "Consider using concise bullet points for skills, "
                "projects and relevant experience where appropriate."
            )
        })

    return feedback, word_count

# ---------------- JOB DESCRIPTION INPUT ----------------

st.divider()

st.subheader("💼 Target Job Description")

job_description = st.text_area(
    "Paste the job description here",
    height=180,
    placeholder=(
        "Example:\n"
        "We are looking for a Machine Learning Intern.\n"
        "Required skills: Python, Pandas, Scikit-learn,\n"
        "Machine Learning and Data Preprocessing."
    ),
    key="job_description_input"
)

# ---------------- ML MATCHING FUNCTION ----------------

def compare_resume_with_job(resume, job_description):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    documents = [resume, job_description]

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    similarity_percentage = round(
        similarity * 100, 2
    )

    return similarity_percentage

# ---------------- SKILL COMPARISON ----------------

def compare_skills(resume, job_description):

    skills = [
        "python",
        "sql",
        "machine learning",
        "scikit-learn",
        "pandas",
        "numpy",
        "data preprocessing",
        "model evaluation",
        "deep learning",
        "tensorflow",
        "pytorch",
        "excel",
        "power bi",
        "tableau",
        "flask",
        "django",
        "statistics"
    ]

    resume_lower = resume.lower()
    job_lower = job_description.lower()

    required_skills = [
        skill for skill in skills
        if skill in job_lower
    ]

    detected_skills = [
        skill for skill in required_skills
        if skill in resume_lower
    ]

    not_detected_skills = [
        skill for skill in required_skills
        if skill not in resume_lower
    ]

    return detected_skills, not_detected_skills

# ---------------- ANALYZE BUTTON ----------------

st.divider()

if st.button("🔍 Analyze Resume", type="primary"):

    if not resume_text.strip():

        st.error(
            "Please paste resume text or upload a readable PDF."
        )

    elif len(resume_text.strip()) < 30:

        st.warning(
            "Please provide more resume text for analysis."
        )

    elif not job_description.strip():

        st.warning(
            "Please paste a job description to use the ML comparison."
        )

    else:

        try:

            feedback, word_count = analyze_resume(
                resume_text
            )

            similarity = compare_resume_with_job(
                resume_text,
                job_description
            )
            # Compare resume skills with job description
            detected_skills, not_detected_skills = compare_skills(
             resume_text,
             job_description
)

            # Save results for the dashboard
            st.session_state["detected_skills"] = detected_skills
            st.session_state["not_detected_skills"] = not_detected_skills

            st.session_state["resume_feedback"] = feedback

            st.session_state["resume_word_count"] = word_count

            st.session_state["similarity_percentage"] = similarity

            st.session_state["analysis_complete"] = True

        except ValueError:

            st.error(
                "Unable to compare the text. "
                "Please provide a longer resume and job description."
            )


# ---------------- FEEDBACK DASHBOARD ----------------

if st.session_state.get("analysis_complete", False):

 st.header("📊 Resume Feedback Dashboard")

# ML similarity score
similarity = st.session_state.get("similarity_percentage")

st.subheader("🤖 ML-Based Resume Comparison")

if similarity is not None:
    st.metric(
        "Resume–Job Description Text Similarity",
        f"{similarity}%"
    )

    st.progress(max(0.0, min(float(similarity) / 100, 1.0)))

    st.caption(
        "This score measures text similarity using TF-IDF "
        "and cosine similarity. It is not a hiring probability "
        "or proof of skill."
    )
else:
    st.info("Run resume analysis to calculate text similarity.")

st.divider()

# Skill comparison results
st.subheader("🛠️ Relevant Skills Analysis")

detected_skills = st.session_state.get("detected_skills", [])
not_detected_skills = st.session_state.get(
    "not_detected_skills", []
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ Skills Detected")

    if detected_skills:
        for skill in detected_skills:
            st.write(f"• {skill.title()}")
    else:
        st.info("No matching skills from the predefined list were detected.")

with col2:
    st.markdown("### 💡 Skills Not Detected")

    if not_detected_skills:
        for skill in not_detected_skills:
            st.write(f"• {skill.title()}")

        st.caption(
            "These skills were not detected in the resume text. "
            "Add them only if they accurately reflect your abilities."
        )
    else:
        st.success(
            "All identified job-description skills from the predefined "
            "list were also detected in the resume."
        )

st.divider()

# Resume statistics
st.subheader("📄 Resume Statistics")

word_count = st.session_state.get("resume_word_count", 0)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Words Analyzed", word_count)

with col2:
    st.metric("Skills Detected", len(detected_skills))

with col3:
    st.metric("Skills Not Detected", len(not_detected_skills))

    # Additional Resume Statistics

feedback = st.session_state.get("resume_feedback", [])

col4, col5 = st.columns(2)

with col4:
    st.metric("Checks Performed", len(feedback))

with col5:
    detected = sum(
        1 for item in feedback
        if item.get("status") == "Section detected"
    )

    st.metric("Sections Detected", detected)

    st.info(
        "These are preliminary content checks, not a validated "
        "resume quality score. A detected heading does not guarantee "
        "that the section is complete or well written."
    )

    st.subheader("Detailed Feedback")

    for item in feedback:

        st.markdown(
            f"""
            <div class="feedback-card">
                <h4>{item["category"]}</h4>
                <b>{item["status"]}</b>
                <p>{item["suggestion"]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("📋 Improvement Checklist")

    st.checkbox("Review section headings and organization")
    st.checkbox("Check spelling and grammar")
    st.checkbox("Make project descriptions more specific")
    st.checkbox("Verify that skills accurately reflect my abilities")
    st.checkbox("Remove unnecessary or repetitive information")

    st.success(
        "Initial resume review complete! "
        "Review each suggestion before making changes."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Resume Feedback Assistant | Introduction to Machine Learning Project"
)