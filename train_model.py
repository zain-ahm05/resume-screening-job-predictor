import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("dataset/resume.csv")


# 2. Select categories
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


# 3. Keep selected categories
df = df[df["Category"].isin(selected_categories)]


# 4. Keep required columns and remove missing values
df = df[["Resume_str", "Category"]].dropna()


# 5. X = resume text
X_text = df["Resume_str"]


# 6. y = correct category
y = df["Category"]


# 7. Convert text into TF-IDF numbers
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(X_text)


# 8. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 9. Display the sizes
print("Total resumes:", len(df))

print("Training samples:", X_train.shape[0])

print("Testing samples:", X_test.shape[0])

print("Training labels:", y_train.shape[0])

print("Testing labels:", y_test.shape[0])

# 10. Create the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# 11. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 12. Make predictions on unseen test data
predictions = model.predict(X_test)

print("\nFirst 10 predictions:")
print(predictions[:10])

print("\nActual categories:")
print(y_test.iloc[:10].values)

# 13. Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(12, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()