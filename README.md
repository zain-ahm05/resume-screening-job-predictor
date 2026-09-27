# Resume Screening & Job Role Predictor

A machine learning web application that analyzes a resume and predicts the most relevant job-role category. The application also detects common technical skills mentioned in the resume.

## Project Overview

Resume screening is an important part of recruitment, but manually reviewing large numbers of resumes can be time-consuming. This project uses Natural Language Processing (NLP) and Machine Learning to automatically classify resumes into job-role categories.

The project uses **TF-IDF** for text feature extraction and a **Linear SVM** classifier for job-role prediction. A **Streamlit** web application provides an interface where users can upload a resume in PDF or TXT format.

## Features

- Upload a resume in **PDF** or **TXT** format
- Extract text from PDF resumes
- Clean and preprocess resume text
- Convert text into numerical features using **TF-IDF**
- Predict the resume's job-role category using **Linear SVM**
- Detect commonly mentioned technical skills
- Simple web interface built with Streamlit

## Dataset

The original dataset contains **1,149 resumes** across multiple job-role categories.

For the final model, these 10 categories were used:

- ACCOUNTANT
- BANKING
- BUSINESS-DEVELOPMENT
- DESIGNER
- ENGINEERING
- FINANCE
- HEALTHCARE
- INFORMATION-TECHNOLOGY
- SALES
- TEACHER

The data was divided into:

- Training samples: **919**
- Testing samples: **230**
- Test size: **20%**
- Random state: **42**
- Stratified train/test split

## Machine Learning Workflow

```text
Resume Dataset
      ↓
Text Cleaning
      ↓
TF-IDF Vectorization
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Linear SVM
      ↓
Model Evaluation
      ↓
Streamlit Web Application
```

## Text Preprocessing

The resume text is processed before model prediction:

1. Convert text to lowercase
2. Remove unnecessary special characters
3. Normalize whitespace
4. Convert cleaned text into TF-IDF features

Final TF-IDF configuration:

```python
TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)
```

This uses both individual words (unigrams) and two-word combinations (bigrams).

## Model Results

Several approaches were tested during development.

| Model / Approach | Accuracy |
|---|---:|
| Logistic Regression | 76.96% |
| Cleaned Text + Logistic Regression | 77.39% |
| TF-IDF Bigrams + Logistic Regression | 77.83% |
| **Linear SVM** | **84.78%** |

The final Linear SVM achieved **84.78% accuracy on the 230-sample test set**, correctly classifying 195 resumes.

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **TF-IDF**
- **Linear SVM**
- **Joblib**
- **PyPDF**
- **Streamlit**

## Project Structure

```text
resume-screening-job-predictor/
│
├── app.py
├── improved_model.py
├── resume_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
│
├── dataset/
│   └── resume.csv
│
└── README.md
```

Additional Python files used during development may also be present.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/resume-screening-job-predictor.git
cd resume-screening-job-predictor
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.env\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```powershell
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## How to Use

1. Open the Streamlit application.
2. Upload a resume in PDF or TXT format.
3. The application extracts the resume text.
4. The text is cleaned and transformed using the saved TF-IDF vectorizer.
5. The Linear SVM model predicts the job-role category.
6. The predicted role is displayed.
7. Detected technical skills are displayed separately.

## Model Files

The trained model and TF-IDF vectorizer are saved using Joblib:

```text
resume_model.pkl
tfidf_vectorizer.pkl
```

These files allow the application to make predictions without retraining the model every time it starts.

## Limitations

- The model can only predict the job-role categories included in the training data.
- It is not a general-purpose resume evaluation system.
- Skill detection uses a predefined list of skills.
- Resume formatting and image-based/scanned PDFs may affect text extraction.
- Performance depends on the quality and diversity of the training dataset.

## Future Scope

Possible improvements include:

- Add more job-role categories
- Improve skill extraction using NLP techniques
- Add resume ranking
- Add job recommendations
- Extract education and work experience
- Support scanned resumes using OCR
- Experiment with advanced NLP/transformer models
- Deploy the application publicly

## Deployment

The application can be deployed using **Streamlit Community Cloud** by connecting this GitHub repository.

**Deployed Application:**  
Add your Streamlit deployed link here.

## Project Information

**Project:** Resume Screening & Job Role Predictor  
**Type:** Machine Learning Mini Project  
**Technologies:** Python, NLP, TF-IDF, Linear SVM, Streamlit

### Student Details

- **Name:** ____________________
- **Roll No.:** ____________________
- **Department:** Artificial Intelligence & Data Science
- **College:** ____________________
- **Academic Year:** 2026–27

## License

This project was developed for educational and academic purposes.
