import pandas as pd
import re #regular expressions
from nltk.corpus import stopwords

# Load our dataset
df = pd.read_csv("dataset/resume.csv")

# Select the categories we want
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

# Keep only selected categories
df = df[df["Category"].isin(selected_categories)]

# Keep only the columns we need
df = df[["Resume_str", "Category"]]

# Remove missing values
df = df.dropna()


stop_words = set(stopwords.words("english"))
# Function to clean resume text
def clean_text(text):
    text = text.lower()
    #replace unnecessary characters with space
    #Python!!! SQL, C++ @2026
    #Python SQL C
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    #replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]


    return " ".join(words)

# Apply cleaning to every resume
df["Cleaned_Resume"] = df["Resume_str"].apply(clean_text)

# Show original and cleaned resume
print("Original Resume:\n")
# Display the first resume
print(df["Resume_str"].iloc[0])

print("\n\nCleaned Resume:\n")
# Display the cleaned version of the first resume
print(df["Cleaned_Resume"].iloc[0])