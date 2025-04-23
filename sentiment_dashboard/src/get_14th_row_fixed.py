import pandas as pd

# Load the CSV file containing Reddit posts
df = pd.read_csv("posts.csv")  # Updated path

# Get the 14th row (index 13)
row_14 = df.iloc[13]

# Print the 14th row
print(row_14)
