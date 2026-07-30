"""
preprocess.py
-------------
Loads and cleans the knowledge base CSV using pandas/numpy:
  - drops empty/duplicate rows
  - normalizes whitespace
  - builds a single "search_text" field combining title + content + tags
    (this is what the retriever indexes against)
"""

import numpy as np
import pandas as pd


def load_knowledge_base(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop rows missing the essentials
    df = df.dropna(subset=["title", "content"])

    # Normalize whitespace / strip
    for col in ["category", "subcategory", "title", "content", "tags"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    # Drop exact duplicate content (keeps the pipeline safe if raw_sources
    # gets appended to more than once)
    df = df.drop_duplicates(subset=["title", "content"])

    # Replace any remaining NaNs (e.g. missing tags) with empty string
    df = df.replace({np.nan: ""})

    # Build the combined text used for retrieval
    df["search_text"] = (
        df["category"] + " | " + df["subcategory"] + " | " +
        df["title"] + ". " + df["content"] + " " + df["tags"]
    )

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    frame = load_knowledge_base("data/knowledge_base.csv")
    print(frame.shape)
    print(frame.head())
