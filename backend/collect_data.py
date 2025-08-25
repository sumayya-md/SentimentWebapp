import praw
import pandas as pd

# 🔹 Replace these with your own details
CLIENT_ID = "YQQlnxdKLU_YGZoGqHdT5Q"
CLIENT_SECRET = "b8AU_p3tINMMqsBQ6KHSKGUx5mGUbA"
USERNAME = "Fancy_Translator_731"
PASSWORD = "@Haleema"
USER_AGENT = "SentimentWebApp by u/Fancy_Translator_731"

# 🔹 Authenticate with Reddit
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    username=USERNAME,
    password=PASSWORD,
    user_agent=USER_AGENT,
)

# 🔹 Choose subreddit & fetch posts
subreddit = reddit.subreddit("technology")   # <-- change subreddit if you like
posts = []
for post in subreddit.hot(limit=50):         # get 50 hot posts
    posts.append([post.title, post.selftext, post.score, post.num_comments])

# 🔹 Save to CSV
df = pd.DataFrame(posts, columns=["Title", "Text", "Score", "Comments"])
df.to_csv("reddit_data.csv", index=False, encoding="utf-8")
print("✅ Data saved to reddit_data.csv")
