import pandas as pd

# Load the collected Reddit data
df = pd.read_csv("reddit_data.csv")

# Add a new column for sentiment (you will manually label some rows)
df["sentiment"] = ""

# Save it back
df.to_csv("reddit_data_labeled.csv", index=False)

print("✅ File ready for labeling: reddit_data_labeled.csv")
