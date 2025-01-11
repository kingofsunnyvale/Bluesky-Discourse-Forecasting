#!/usr/bin/env python3
"""
find_peak_day.py

Find which day in the "alpindale/two-million-bluesky-posts" dataset
had the highest number of posts (discarding posts with empty text).
"""

import sys
import argparse
import pandas as pd
from datasets import load_dataset
from dateutil.parser import isoparse

def main():
    # Load the dataset
    dataset_name = "alpindale/two-million-bluesky-posts"
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print("Total posts in dataset:", len(ds))

    # Convert to pandas
    df = ds.to_pandas()

    # Discard empty text
    df["text"] = df["text"].astype(str)
    original_len = len(df)
    df = df[df["text"].str.strip() != ""]
    removed = original_len - len(df)
    print(f"Removed {removed} posts with empty text. Remaining: {len(df)}")

    # Convert created_at to date
    def parse_date(dt_str):
        # Some timestamps have trailing "Z" -> convert to +00:00 for parsing
        return isoparse(dt_str.replace("Z", "+00:00")).date()

    df["date"] = df["created_at"].apply(parse_date)

    # Group by date and count
    daily_counts = df.groupby("date")["text"].count().reset_index(name="count")

    # Find the day with the maximum count
    max_day_row = daily_counts.loc[daily_counts["count"].idxmax()]
    peak_day = max_day_row["date"]
    peak_count = max_day_row["count"]

    print(f"\nThe day with the most posts: {peak_day} (with {peak_count} posts).\n")

if __name__ == "__main__":
    main()