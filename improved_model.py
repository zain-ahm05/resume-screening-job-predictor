import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC




# Load dataset
df = pd.read_csv("dataset/resume.csv")

print("Total resumes:", len(df))

# Select the same 10 job categories
selected_categories = [
    "ACCOUNTANT",
    "BANKING",
    "BUSINESS-DEVELOPMENT",
    "DESIGNER",
    "ENGINEERING",
    "FINANCE",
    "HEALTHCARE",
    "INFORMATION-TECHNOLOGY",
    "SALES",
    "TEACHER"
]



df = df[df["Category"].isin(selected_categories)]

print("Resumes after category selection:", len(df))

# Clean resume text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply cleaning to every resume
df["Cleaned_Resume"] = df["Resume_str"].apply(clean_text)

print("\nOriginal Resume:")
print(df["Resume_str"].iloc[0][:500])

print("\nCleaned Resume:")
print(df["Cleaned_Resume"].iloc[0][:500])

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)
# Convert cleaned resumes into numerical features
X = vectorizer.fit_transform(df["Cleaned_Resume"])

# Target labels
y = df["Category"]

print("\nNumber of resumes:", X.shape[0])
print("Number of features:", X.shape[1])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Create the improved Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model using the training data
model.fit(X_train, y_train)

# Create Linear SVM model
svm_model = LinearSVC()

# Train the SVM model
svm_model.fit(X_train, y_train)

print("\nLinear SVM training completed!")

# Make predictions
svm_predictions = svm_model.predict(X_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("Linear SVM Accuracy:", svm_accuracy * 100)

print("\nImproved model training completed!")

# Make predictions on the test data
predictions = model.predict(X_test)

print("\nFirst 10 predictions:")
print(predictions[:10])

print("\nActual categories:")
print(y_test.iloc[:10].values)

# Calculate improved model accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nImproved Model Accuracy:", accuracy)
print("Improved Model Accuracy (%):", accuracy * 100)

from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, svm_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))

import matplotlib.pyplot as plt
import seaborn as sns

# Create confusion matrix
cm = confusion_matrix(y_test, svm_predictions)

# Plot confusion matrix
plt.figure(figsize=(12, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=svm_model.classes_,
    yticklabels=svm_model.classes_
)

plt.title("Resume Job Role Prediction - Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

import joblib

# Save the trained SVM model
joblib.dump(svm_model, "resume_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")