import os
import json
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from atproto_client.models import get_or_create
from atproto import CAR, models
from atproto_firehose import FirehoseSubscribeReposClient, parse_subscribe_repos_message

class JSONExtra(json.JSONEncoder):
    def default(self, obj):
        try:
            result = json.JSONEncoder.default(self, obj)
            return result
        except:
            return repr(obj)

client = FirehoseSubscribeReposClient()

# We'll keep the data folder creation logic the same
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
os.makedirs(data_dir, exist_ok=True)

# Track current day and posts for that day
current_day = None
posts_for_current_day = []

def get_day_from_timestamp(timestamp: str) -> str:
    """
    Parse the ISO8601 'createdAt' timestamp string and return YYYY-MM-DD.
    Adjust as needed if the format of your timestamps differs.
    """
    # Remove any trailing Z if present (e.g., 2023-10-03T13:45:00Z)
    # and parse into a datetime object
    timestamp_clean = timestamp.replace('Z', '')
    dt = datetime.fromisoformat(timestamp_clean)
    return dt.strftime('%Y-%m-%d')

def flush_posts_to_parquet(day: str, posts: list):
    """
    Flush the list of posts to a Parquet file named for the day (YYYY-MM-DD).
    """
    if not posts:
        return

    # Convert the posts to a DataFrame
    df = pd.DataFrame(posts)

    # Create a pyarrow table
    table = pa.Table.from_pandas(df)

    # Construct the Parquet file path
    parquet_file_path = os.path.join(data_dir, f"{day}.parquet")

    # Write the table to Parquet
    pq.write_table(table, parquet_file_path)

def on_message_handler(message):
    global current_day, posts_for_current_day

    commit = parse_subscribe_repos_message(message)
    if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
        return

    car = CAR.from_bytes(commit.blocks)

    for op in commit.ops:
        if op.action == "create" and op.cid:
            raw = car.blocks.get(op.cid)
            cooked = get_or_create(raw, strict=False)

            # Only process feed posts
            if cooked.py_type == "app.bsky.feed.post":
                text = raw.get("text")
                langs = raw.get("langs")
                created_at = raw.get("createdAt")

                # Only collect if text is non-empty and 'en' is in langs
                text_is_non_empty = bool(text and text.strip())
                langs_is_english = (
                    langs
                    and isinstance(langs, list)
                    and "en" in langs
                )

                if text_is_non_empty and langs_is_english and created_at:
                    # Determine which day this post belongs to
                    day_str = get_day_from_timestamp(created_at)

                    # If this is the first post or the day changed, flush old buffer
                    if current_day is None:
                        current_day = day_str
                    elif day_str != current_day:
                        flush_posts_to_parquet(current_day, posts_for_current_day)
                        posts_for_current_day.clear()
                        current_day = day_str

                    # Add post to current day's buffer
                    posts_for_current_day.append({
                        "text": text,
                        "createdAt": created_at,
                    })

def main():
    """Main entry point for the script."""
    print("Starting Bluesky Firehose scraper.")
    try:
        client.start(on_message_handler)
    except KeyboardInterrupt:
        print("Flushing remaining posts and exiting...")
    finally:
        # Flush any remaining posts in the buffer
        if posts_for_current_day and current_day:
            flush_posts_to_parquet(current_day, posts_for_current_day)

if __name__ == "__main__":
    main()