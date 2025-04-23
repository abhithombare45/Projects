import praw
reddit = praw.Reddit(
    client_id="jfwAc0EHTv79eqlSdCF1iA",
    client_secret="_hyRWsG0vELRwVfcSTB4cxDugMiN_A",
    user_agent="mac:Sentiment_dashboard:v1.0 (by /u/abhithombare45)"
)
print(reddit.read_only)  # Should print True

