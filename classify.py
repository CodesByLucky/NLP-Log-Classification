"""Hybrid log classification pipeline.

Each log goes to the cheapest tier that can label it:
  1. Regex rules        - fixed templates, no inference
  2. Embeddings + LR    - sources with plenty of labeled data
  3. LLM                - sources with too little labeled data (LegacyCRM)
"""
from pathlib import Path

import pandas as pd

from processor_bert import classify_with_bert_batch
from processor_llm import classify_with_llm
from processor_regex import classify_with_regex

ROOT = Path(__file__).resolve().parent
LLM_SOURCES = {"LegacyCRM"}


def classify(logs):
    """Classify (source, log_message) pairs and return labels in input order.

    Logs that need the embedding model are collected and encoded in one batch.
    """
    labels = [None] * len(logs)
    pending_bert = []

    for i, (source, message) in enumerate(logs):
        if source in LLM_SOURCES:
            labels[i] = classify_with_llm(message)
            continue
        label = classify_with_regex(message)
        if label:
            labels[i] = label
        else:
            pending_bert.append(i)

    if pending_bert:
        bert_labels = classify_with_bert_batch([logs[i][1] for i in pending_bert])
        for i, label in zip(pending_bert, bert_labels):
            labels[i] = label

    return labels


def classify_log(source, log_msg):
    return classify([(source, log_msg)])[0]


def classify_dataframe(df):
    """Return a copy of df with a target_label column added."""
    df = df.copy()
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))
    return df


def classify_csv(input_file, output_file):
    df = pd.read_csv(input_file, skipinitialspace=True)
    classify_dataframe(df).to_csv(output_file, index=False)
    return output_file


if __name__ == "__main__":
    out = classify_csv(ROOT / "resources" / "test.csv", ROOT / "resources" / "output.csv")
    print(f"Saved results to {out}")
