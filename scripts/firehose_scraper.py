import json
import csv
import os

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

# Parameters for batching
BATCH_SIZE = 1000
posts_buffer = []

# Prepare the CSV output path
# This assumes the script is in a folder named "scripts" at the same level as "data".
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
csv_file_path = os.path.join(data_dir, 'posts.csv')

def flush_posts_to_csv():
    """Write the accumulated posts (in posts_buffer) to the CSV file and then clear the buffer."""
    global posts_buffer
    
    # Check if CSV already exists to determine if we need headers
    file_exists = os.path.exists(csv_file_path)
    
    # Ensure the data directory exists, just in case
    os.makedirs(data_dir, exist_ok=True)
    
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["text", "createdAt"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # If file doesn't exist or was just created, write the header
        if not file_exists:
            writer.writeheader()
        
        # Write all rows in our buffer
        writer.writerows(posts_buffer)
    
    # Clear the buffer after flushing
    posts_buffer = []

def on_message_handler(message):
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

                if text_is_non_empty and langs_is_english:
                    filtered_output = {
                        "text": text,
                        "createdAt": created_at,
                    }

                    # Add the post to our buffer
                    posts_buffer.append(filtered_output)

                    # If we hit the batch size, flush to CSV
                    if len(posts_buffer) >= BATCH_SIZE:
                        flush_posts_to_csv()

def main():
    """Main entry point for the script."""
    print("Starting Bluesky Firehose scraper.")
    try:
        client.start(on_message_handler)
    except KeyboardInterrupt:
        print("Flushing remaining posts and exiting...")
    finally:
        if posts_buffer:
            flush_posts_to_csv()

if __name__ == "__main__":
    main()