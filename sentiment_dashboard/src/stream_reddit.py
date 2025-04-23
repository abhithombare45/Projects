import praw
import time
import logging
import csv
import os

# Set up logging
logging.basicConfig(
    filename="stream_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Reddit authentication
reddit = praw.Reddit(
    client_id="jfwAc0EHTv79eqlSdCF1iA",
    client_secret="_hyRWsG0vELRwVfcSTB4cxDugMiN_A",
    user_agent="mac:sentiment_dashboard:v1.0 (by /u/Busy-Positive1690)",
)

# Choose subreddit
subreddit = reddit.subreddit("technology")

# Define CSV file path
file_path = "posts.csv"

# Create file with headers if it doesn’t exist
if not os.path.exists(file_path):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Title", "Rate Limit Remaining"])

# Streaming loop with CSV appending
with open(file_path, "a", newline="") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
    try:
        for submission in subreddit.stream.submissions():
            title = submission.title
            if title.count(",") > 1:
                title = f'"{title}"'  # Enclose in double quotes if multiple commas

            row = [time.ctime(), title, reddit.auth.limits["remaining"]]
            writer.writerow(row)
            print(
                f"Time: {time.ctime()}, Title: {title}, Rate limit remaining: {reddit.auth.limits['remaining']}"
            )
            time.sleep(5)
    except Exception as e:
        logging.error(f"Error during streaming: {e}")
        print(f"Error occurred: {e}")
        time.sleep(60)


""" We have 438 (excluding header) rows created with below code. """

# import praw
# import time
# import logging
# import csv
# import os

# # Set up logging
# logging.basicConfig(
#     filename="stream_errors.log",
#     level=logging.ERROR,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# # Reddit authentication
# reddit = praw.Reddit(
#     client_id="jfwAc0EHTv79eqlSdCF1iA",
#     client_secret="_hyRWsG0vELRwVfcSTB4cxDugMiN_A",
#     user_agent="mac:sentiment_dashboard:v1.0 (by /u/Busy-Positive1690)",
# )

# # Choose subreddit
# subreddit = reddit.subreddit("technology")

# # Define CSV file path
# file_path = "posts.csv"

# # Create file with headers if it doesn’t exist
# if not os.path.exists(file_path):
#     with open(file_path, "w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow(["Time", "Title", "Rate Limit Remaining"])

# # Streaming loop with CSV appending
# with open(file_path, "a", newline="") as file:
#     writer = csv.writer(file)
#     try:
#         for submission in subreddit.stream.submissions():
#             row = [time.ctime(), submission.title, reddit.auth.limits["remaining"]]
#             writer.writerow(row)
#             print(
#                 f"Time: {time.ctime()}, Title: {submission.title}, Rate limit remaining: {reddit.auth.limits['remaining']}"
#             )
#             time.sleep(5)
#     except Exception as e:
#         logging.error(f"Error during streaming: {e}")
#         print(f"Error occurred: {e}")
#         time.sleep(60)


""" 1. We are getting 120 posts if run for 10"""

# import praw
# import time

# reddit = praw.Reddit("Sentiment_dashboard")
# subreddit = reddit.subreddit("technology")
# try:
#     for submission in subreddit.stream.submissions():
#         print(f"Time: {time.ctime()}, Title: {submission.title}")
#         print(f"Rate limit remaining: {reddit.auth.limits['remaining']}")
#         time.sleep(5)  # Wait 5 seconds to stay under 100 QPM (12 posts/minute)
# except Exception as e:
#     print(f"Error: {e}")
#     time.sleep(60)  # Wait before retrying


""" 2. We are getting 69,120 posts if run for 10 min. 
    
    But a/c could be banned. """

# import praw
# import time
# import logging

# logging.basicConfig(filename='stream_errors.log', level=logging.ERROR)
# reddit = praw.Reddit("Sentiment_dashboard")
# subreddit = reddit.subreddit("technology")
# try:
#     while True:  # Run in bursts
#         count = 0
#         start_time = time.time()
#         while time.time() - start_time < 30:  # Run for 30 seconds at max speed
#             for submission in subreddit.stream.submissions():
#                 count += 1
#                 print(f"Time: {time.ctime()}, Title: {submission.title}, Count: {count}")
#                 print(f"Rate limit remaining: {reddit.auth.limits['remaining']}")
#         print(f"Burst complete, got {count} posts in 30 seconds")
#         time.sleep(30)  # Wait 30 seconds before next burst
# except Exception as e:
#     logging.error(f"Error occurred: {e}")
#     print(f"Error: {e}")
#     time.sleep(60)
