#TF: How often does this word appear in this resume?
#IDF: How uncommon is this word across all resumes?
#TF-IDF: How important is this word for identifying this resume?

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load our dataset
df = pd.read_csv("dataset/resume.csv")

# Categories we want to use
selected_categories = [
    "INFORMATION-TECHNOLOGY",
    "BUSINESS-DEVELOPMENT",
    "FINANCE",
    "ENGINEERING",
    "ACCOUNTANT",
    "SALES",
    "HEALTHCARE",
    "BANKING",
    "DESIGNER",
    "TEACHER"
]

# Keep only our selected categories
df = df[df["Category"].isin(selected_categories)]

# Keep only the columns we need
df = df[["Resume_str", "Category"]].dropna()

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

# Convert resume text into numerical features
X = vectorizer.fit_transform(df["Resume_str"])

print("Number of resumes:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nFirst 20 features:")
print(vectorizer.get_feature_names_out()[:20])