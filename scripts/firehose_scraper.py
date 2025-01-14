import json
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
                # We'll still look up 'langs' just for checking if 'en' is included
                text = raw.get("text")
                langs = raw.get("langs")
                created_at = raw.get("createdAt")

                # Only print if text is non-empty and 'en' is in langs
                text_is_non_empty = bool(text and text.strip())
                langs_is_english = (
                    langs
                    and isinstance(langs, list)
                    and "en" in langs
                )

                if text_is_non_empty and langs_is_english:
                    # Construct a dict *without* 'langs'
                    filtered_output = {
                        "text": text,
                        "createdAt": created_at,
                    }
                    print(json.dumps(filtered_output, cls=JSONExtra, indent=2))

client.start(on_message_handler)