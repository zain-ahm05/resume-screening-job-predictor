import re
import joblib
import streamlit as st
from pypdf import PdfReader

# ==========================================
# Resume Screening & Job Role Predictor
# ==========================================

st.set_page_config(
    page_title="Resume Job Role Predictor",
    page_icon="📄",
    layout="centered"
)


# ==========================================
# File paths
# ==========================================

MODEL_PATH = "resume_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


# ==========================================
# Load trained model and TF-IDF vectorizer
# ==========================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer

# ==========================================
# Skills we want to detect
# ==========================================

SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "data science",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "excel",
    "power bi",
    "tableau",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "git",
    "github",
]
def detect_skills(text):

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        if skill.lower() in text:
            found_skills.append(skill)

    return found_skills
# ==========================================
# Clean resume text
# ==========================================

def clean_text(text):
    # Convert text to lowercase
    
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# Extract text from PDF
# ==========================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==========================================
# App Header
# ==========================================

st.title("📄 Resume Job Role Predictor")

st.markdown(
    """
    ### 🤖 AI-Powered Resume Screening

    Upload a resume and the machine learning model will analyze
    the resume and predict the most suitable job category.
    """
)

st.divider()


# ==========================================
# Model information
# ==========================================

st.info(
    "Model: Linear SVM  |  Features: TF-IDF with unigrams + bigrams"
)


# ==========================================
# Upload Resume
# ==========================================

st.subheader("📤 Upload Your Resume")

uploaded_file = st.file_uploader(
    "Choose a PDF or TXT resume",
    type=["pdf", "txt"],
    help="Upload a text-based PDF or TXT resume for analysis."
)


# ==========================================
# Process uploaded resume
# ==========================================

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # --------------------------------------
    # Extract text
    # --------------------------------------

    if uploaded_file.name.lower().endswith(".pdf"):

        raw_text = extract_pdf_text(uploaded_file)

    else:

        raw_text = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )


    # --------------------------------------
    # Check extracted text
    # --------------------------------------

    if not raw_text.strip():

        st.error(
            "Could not extract text from this file. "
            "Please upload a text-based PDF or TXT resume."
        )

    else:

        # ----------------------------------
        # Resume Preview
        # ----------------------------------

        st.subheader("📋 Resume Preview")

        st.text_area(
            "Extracted Resume Text",
            raw_text[:3000],
            height=250
        )


        # ----------------------------------
        # Prediction button
        # ----------------------------------

        if st.button("🔍 Predict Job Role", type="primary"):

            try:

                # Load trained model and vectorizer
                model, vectorizer = load_artifacts()


                # Clean the resume text
                cleaned = clean_text(raw_text)


                # Convert resume text into TF-IDF features
                features = vectorizer.transform([cleaned])


                # Predict the job category
                prediction = model.predict(features)[0]


                # ----------------------------------
                # Display prediction
                # ----------------------------------

                st.divider()

                st.subheader("🎯 Prediction Result")

                st.success(
                    f"Predicted Job Role: **{prediction}**"
                )
                # Detect skills from the resume
                found_skills = detect_skills(raw_text)
                
                st.subheader("🧠 Skills Detected")
                
                if found_skills:
                
                    for skill in found_skills:
                        st.markdown(
                            f"🏷️ **{skill.title()}**"
                        )

                
                else:
                
                    st.info("No known skills were detected.")


                # ----------------------------------
                # Show extracted text
                # ----------------------------------

                with st.expander("🔎 Show processed resume text"):

                    st.text(cleaned[:3000])


            except FileNotFoundError:

                st.error(
                    "Model files were not found.\n\n"
                    "Make sure these two files are in the same "
                    "folder as app.py:\n\n"
                    "• resume_model.pkl\n"
                    "• tfidf_vectorizer.pkl"
                )


            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "Resume Screening & Job Role Predictor | "
    "Built with Python, Streamlit, TF-IDF and Linear SVM"
)