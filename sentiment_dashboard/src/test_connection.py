import praw

reddit = praw.Reddit("Sentiment_dashboard")
subreddit = reddit.subreddit("technology")
for submission in subreddit.top(limit=5):
    print(f"Title: {submission.title}, Score: {submission.score}")
