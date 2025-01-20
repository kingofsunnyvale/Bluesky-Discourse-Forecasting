import os
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from atproto_client.models import get_or_create
from atproto import CAR, models
from atproto_firehose import FirehoseSubscribeReposClient, parse_subscribe_repos_message

client = FirehoseSubscribeReposClient()

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
os.makedirs(data_dir, exist_ok=True)

current_day = datetime.now().strftime("%Y-%m-%d")
posts_for_current_day = []
FLUST_THRESHOLD = 1000

total_posts_written = 0

def get_current_day():
    return datetime.now().strftime("%Y-%m-%d")

def flush_posts_to_parquet(day: str, posts: list) -> int:
    if not posts:
        return 0

    df = pd.DataFrame(posts)
    table = pa.Table.from_pandas(df)
    parquet_file_path = os.path.join(data_dir, f"{day}.parquet")
    pq.write_table(table, parquet_file_path)

    return len(posts)

def on_message_handler(message):
    global current_day, posts_for_current_day, total_posts_written

    commit = parse_subscribe_repos_message(message)
    if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
        return

    car = CAR.from_bytes(commit.blocks)

    for op in commit.ops:
        if op.action == "create" and op.cid:
            raw = car.blocks.get(op.cid)
            cooked = get_or_create(raw, strict=False)

            # Only process feed posts
            if cooked is not None and cooked.py_type == "app.bsky.feed.post":
                text = raw.get("text")
                langs = raw.get("langs")
                created_at = raw.get("createdAt")

                text_is_non_empty = bool(text and text.strip())
                langs_is_english = (
                    langs
                    and isinstance(langs, list)
                    and "en" in langs
                )

                if text_is_non_empty and langs_is_english and created_at:                        
                    if len(posts_for_current_day) >= FLUST_THRESHOLD:
                        flushed = flush_posts_to_parquet(current_day, posts_for_current_day)
                        total_posts_written += flushed
                        print(f"Total posts written: {total_posts_written}")
                        print(f"Timestamp of last post written: {posts_for_current_day[-1]['createdAt']}")
                        posts_for_current_day.clear()

                    # Add post to current day's buffer
                    posts_for_current_day.append({
                        "text": text,
                        "createdAt": created_at,
                    })
def main():
    print("Starting Bluesky Firehose scraper.")
    try:
        client.start(on_message_handler)
    except KeyboardInterrupt:
        print("Flushing remaining posts and exiting...")
    finally:
        # Flush any remaining posts in the buffer
        if posts_for_current_day and current_day:
            flushed = flush_posts_to_parquet(current_day, posts_for_current_day)
            global total_posts_written
            total_posts_written += flushed
            print(f"Total posts written: {total_posts_written}")


if __name__ == "__main__":
    main()