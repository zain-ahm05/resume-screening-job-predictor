import pandas as pd

df = pd.read_csv("dataset/resume.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns)
#Print the number of different job categories in my dataset.
print("\nNumber of categories:", df["Category"].nunique())

print("\nCategory counts:")
#Count how many times each value appears.
print(df["Category"].value_counts())