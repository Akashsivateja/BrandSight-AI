# reddit_client.py

import praw
import config
import time  # Added to prevent hitting Reddit rate limits

def get_reddit_client():
    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT
    )
    return reddit

def search_reddit(brand, limit=config.SEARCH_LIMIT):
    reddit = get_reddit_client()
    posts = []
    for post in reddit.subreddit("all").search(brand, sort="new", limit=limit):
        posts.append(post)
        time.sleep(1)  # ⏱️ Delay to avoid 429 TooManyRequests
    return posts
