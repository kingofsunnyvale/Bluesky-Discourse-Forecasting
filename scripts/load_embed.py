#!/usr/bin/env python3
"""
load_embed.py

Hardcoded script to:
  - Load the 'alpindale/two-million-bluesky-posts' dataset from HF
  - Filter posts between a given time range
  - (NEW) Discard any post whose `text` is empty
  - Embed the remaining posts using an EXISTING Hugging Face Inference Endpoint
  - Save partial results (CSV) into ../data/, with the time range in its filename

Command-line arguments:
  --start_time (ISO 8601, e.g. "2024-11-27T07:00:00")
  --end_time   (ISO 8601, e.g. "2024-11-27T08:00:00")
  --checkpoint_interval (save partial results every N batches)

IMPORTANT: 
  1. You need to be logged into Hugging Face with credentials that have access 
     to this private Inference Endpoint. You can do:
         huggingface-cli login
     Or set the HF token as an environment variable (HF_API_TOKEN).
  2. The endpoint URL is hardcoded below.
"""

import os
import sys
import json
import argparse as ap  # <-- Aliasing argparse to avoid conflict with dateutil.parser
import numpy as np
import pandas as pd
from tqdm import tqdm
from dateutil import parser
from datetime import datetime, timezone

# Hugging Face
try:
    from huggingface_hub import InferenceClient
except ImportError:
    print("Please install `huggingface_hub` via `pip install huggingface_hub`.")
    sys.exit(1)

# Datasets library
try:
    from datasets import load_dataset
except ImportError:
    print("Please install `datasets` via `pip install datasets`.")
    sys.exit(1)


###############################################################################
# Hardcoded constants (matching original notebook)
###############################################################################
DATASET_PATH = "alpindale/two-million-bluesky-posts"

# EXISTING endpoint URL
ENDPOINT_URL = "https://oip5t8y3edcaq2fe.us-east-1.aws.endpoints.huggingface.cloud"

# We'll embed in small batches (like the original code)
BATCH_SIZE = 2


def sanitize_datetime(dt_str: str) -> str:
    """
    Replaces ':' with '-' so the datetime can be used safely in filenames.
    """
    return dt_str.replace(":", "-")


def is_in_time_range(example, start_dt, end_dt):
    """
    Returns True if example's 'created_at' is between start_dt (inclusive)
    and end_dt (exclusive).
    """
    # Convert "Z" suffix to '+00:00' for parsing
    aware_str = example["created_at"].replace("Z", "+00:00")
    dt = parser.isoparse(aware_str)
    return start_dt <= dt < end_dt


def safe_text(txt, max_chars=5000):
    """
    Cleans and truncates text to a maximum of `max_chars`.
    """
    txt = txt.strip()
    return txt[:max_chars]


def load_existing_partial_results(csv_path: str) -> pd.DataFrame:
    """
    If there's an existing CSV from a previous partial run,
    load it and parse the 'embedding' column from JSON.
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    print(f"Found existing partial CSV: {csv_path}")
    df_partial = pd.read_csv(csv_path)
    if "embedding" in df_partial.columns:
        df_partial["embedding"] = df_partial["embedding"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    return df_partial


def main(args):
    ###########################################################################
    # 0. Figure out the output CSV path (in ../data/) and create the folder
    ###########################################################################
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Build filename encoding the time range
    start_str_sanitized = sanitize_datetime(args.start_time)
    end_str_sanitized = sanitize_datetime(args.end_time)
    csv_filename = f"subset_{start_str_sanitized}_{end_str_sanitized}_embedded.csv"
    output_csv_path = os.path.join(data_dir, csv_filename)

    print(f"Will save results to: {output_csv_path}")

    ###########################################################################
    # 1. Set up client for the existing endpoint
    ###########################################################################
    # If you've logged in with huggingface-cli, it should use your local credentials.
    # Or you can set the environment variable HF_API_TOKEN=...
    print(f"Using existing endpoint: {ENDPOINT_URL}")
    client = InferenceClient(ENDPOINT_URL)

    ###########################################################################
    # 2. Load dataset and filter by time range, then discard empty text
    ###########################################################################
    print("Loading dataset:", DATASET_PATH)
    ds = load_dataset(DATASET_PATH, split="train")
    print("Total dataset length:", len(ds))

    start_dt = parser.isoparse(args.start_time)
    end_dt = parser.isoparse(args.end_time)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    print(f"Filtering dataset for posts between {start_dt} and {end_dt}...")
    ds_time = ds.filter(lambda x: is_in_time_range(x, start_dt, end_dt))
    print("Found", len(ds_time), "posts in that time range.")

    # NEW: Discard any post whose text is empty (after stripping)
    print("Discarding posts with empty text...")
    ds_time = ds_time.filter(lambda x: len(x["text"].strip()) > 0)
    print("Remaining after removing empty text:", len(ds_time), "\n")

    if len(ds_time) == 0:
        print("No posts found in the specified time range (or all had empty text). Exiting.")
        sys.exit(0)

    df_time = ds_time.to_pandas().reset_index(drop=True)
    df_time["index"] = df_time.index  # We'll track unique row indices

    ###########################################################################
    # 3. Check partial results and skip what's done
    ###########################################################################
    partial_df = load_existing_partial_results(output_csv_path)

    if not partial_df.empty:
        processed_indices = set(partial_df["index"])
        unprocessed_df = df_time[~df_time["index"].isin(processed_indices)]
        print(f"Skipping {len(processed_indices)} already processed posts.")
    else:
        unprocessed_df = df_time

    if unprocessed_df.empty:
        print("All posts in this time range are already embedded.")
        sys.exit(0)

    ###########################################################################
    # 4. Embed in batches, saving partial results
    ###########################################################################
    all_new_rows = []
    indices = list(unprocessed_df.index)

    for start_idx in tqdm(range(0, len(indices), BATCH_SIZE), desc="Embedding Batches"):
        batch_indices = indices[start_idx : start_idx + BATCH_SIZE]
        batch_texts = [safe_text(unprocessed_df.loc[i, "text"]) for i in batch_indices]

        # Inference call - pass "task=feature-extraction" to get embeddings
        response = client.post(
            json={"inputs": batch_texts, "truncate": True},
            task="feature-extraction",
        )
        # The response is raw bytes, so decode -> JSON -> NumPy
        batch_embeddings = np.array(json.loads(response.decode()))

        # Merge embeddings back
        for idx_in_batch, row_emb in zip(batch_indices, batch_embeddings):
            # Get the entire row as a dict
            row_dict = unprocessed_df.loc[idx_in_batch].to_dict()
            # Convert embedding to JSON for storage
            row_dict["embedding"] = json.dumps(row_emb.tolist())
            all_new_rows.append(row_dict)

        # Periodic checkpoint
        if (start_idx // BATCH_SIZE) % args.checkpoint_interval == 0:
            temp_new_df = pd.DataFrame(all_new_rows)
            combined_df = pd.concat([partial_df, temp_new_df], ignore_index=True)
            combined_df.to_csv(output_csv_path, index=False)
            print(f"[Checkpoint] Saved {len(combined_df)} rows to {output_csv_path}")

    # Final save
    final_new_df = pd.DataFrame(all_new_rows)
    final_combined_df = pd.concat([partial_df, final_new_df], ignore_index=True)
    final_combined_df.to_csv(output_csv_path, index=False)

    print(f"\nDone. Saved final CSV with {len(final_combined_df)} rows to {output_csv_path}\n")


if __name__ == "__main__":
    arg_parser = ap.ArgumentParser(
        description="Hardcoded script to embed Bluesky posts within a time range, using an existing HF Endpoint."
    )
    arg_parser.add_argument("--start_time", type=str, required=True,
                           help="ISO datetime for start (e.g. '2024-11-27T07:00:00').")
    arg_parser.add_argument("--end_time", type=str, required=True,
                           help="ISO datetime for end (exclusive).")
    arg_parser.add_argument("--checkpoint_interval", type=int, default=1,
                           help="Save partial CSV after every N batches.")
    args = arg_parser.parse_args()

    main(args)