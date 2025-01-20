import os
from sys import getsizeof
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

# credit: stackoverflow
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

MB = 10**6
GB = 10**3 * MB
FLUST_THRESHOLD = 1000
SIZE_STATS_INTERVAL = FLUST_THRESHOLD * 100
MAX_CACHE_SIZE = 32 * GB

post_dictionary = dict()
posts_for_current_day = []
date_usage = []

total_posts_written = 0

def get_current_day():
    return datetime.now().strftime("%Y-%m-%d")

def flush_posts_to_parquet(filename: str, posts: list) -> int:
    if not posts:
        return 0

    df = pd.DataFrame(posts)
    table = pa.Table.from_pandas(df)
    parquet_file_path = os.path.join(data_dir, f"{filename}.parquet")
    pq.write_table(table, parquet_file_path)

    return len(posts)

def on_message_handler(message):
    global current_day, posts_for_current_day, total_posts_written

    current_day = get_current_day()

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
                    if current_day != get_current_day() or len(posts_for_current_day) >= FLUST_THRESHOLD:
                        if current_day in post_dictionary:
                            flushed_posts = post_dictionary[current_day]
                        else:
                            flushed_posts = []

                        flushed_posts += posts_for_current_day
                        flushed = flush_posts_to_parquet(current_day, flushed_posts)
                        total_posts_written += len(posts_for_current_day)
                        post_dictionary[current_day] = flushed_posts                  
                        if current_day in date_usage:
                            date_usage.remove(current_day)
                        date_usage.append(current_day)
                        print(f"Total posts written: {total_posts_written}")
                        print(f"Timestamp of last post written: {posts_for_current_day[-1]['createdAt']}")
                        posts_for_current_day.clear()
                        current_day = get_current_day()               

                        # memory snap flag
                        if total_posts_written % SIZE_STATS_INTERVAL == 0:
                            print(f"Memory usage stats:")
                            print(f"Total posts written: {total_posts_written}")
                            print(f"Timestamp of last post written: {created_at}")
                            print(f"Current dictionary size is {get_size(post_dictionary)/MB} MB")
                    
                    # if getsizeof(posts_for_current_day) >= MAX_CACHE_SIZE:
                    #     print(f"Dictionary is too large: {getsizeof(posts_for_current_day)}")
                    #     if current_day != date_usage[0]:
                    #         post_dictionary.pop(date_usage[0])
                    #         date_usage.pop(0)
                    #     print(f"New dictionary size is {getsizeof(posts_for_current_day)}")

                    # Add post to current day's buffer
                    posts_for_current_day.append({
                        "text": text,
                        "createdAt": created_at,
                    })
def main():
    global current_day
    print("Starting Bluesky Firehose scraper.")
    try:
        client.start(on_message_handler)
    except KeyboardInterrupt:
        print("Flushing remaining posts and exiting...")
    finally:
        # Flush any remaining posts in the buffer
        if posts_for_current_day and current_day:
            if current_day in post_dictionary:
                flushed_posts = post_dictionary[current_day]
            else:
                flushed_posts = []

            flushed_posts += posts_for_current_day
            flushed = flush_posts_to_parquet(current_day, flushed_posts)
            total_posts_written += flushed
            post_dictionary[current_day] = flushed_posts                  
            if current_day in date_usage:
                date_usage.remove(current_day)
            date_usage.append(current_day)
            print(f"Total posts written: {total_posts_written}")
            print(f"Timestamp of last post written: {posts_for_current_day[-1]['createdAt']}")
            posts_for_current_day.clear()
            current_day = get_current_day()               
                    

if __name__ == "__main__":
    main()