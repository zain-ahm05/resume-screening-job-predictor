import pandas as pd

# Load the dataset
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

# Keep only the selected categories
df = df[df["Category"].isin(selected_categories)]

# Keep only the columns we need
df = df[["Resume_str", "Category"]]

# Remove missing values
df = df.dropna()

# Reset row numbers
df = df.reset_index(drop=True)

# Display information
print("Number of resumes:", len(df))

print("\nCategories:")
print(df["Category"].value_counts())

print("\nFirst resume:")
print(df.iloc[0])