"""Adds work_type by extracting it directly from the 'Work' column.

Format is consistently 'WS/.../<numeric id>-<work type description>', e.g.
'WS/\\t MP620/2024-2025/133166-Construction of buildings for community
cultural activities'. The numeric id is always 5+ digits immediately
followed by '-', so everything after that hyphen is the work type — no
join against MPLADS.csv needed, and no NLP/classification model needed
either, since the category is already explicit in the text.

Confirmed against this dataset: 96 distinct work types, 1 row failed to
match (falls back to the 'Work category' column's own value, then to
'Unknown' if that's also missing) — good enough for a screening feature,
not presented as a perfect parse.
"""
import re

import pandas as pd

_PATTERN = re.compile(r"\d{5,}-(.*)$")


def _extract(raw: str) -> str | None:
    match = _PATTERN.search(raw)
    return match.group(1).strip() if match else None


def add_work_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    extracted = df["work_raw"].apply(_extract)
    fallback = df["work_category"] if "work_category" in df.columns else None
    if fallback is not None:
        df["work_type"] = extracted.fillna(fallback).fillna("Unknown")
    else:
        df["work_type"] = extracted.fillna("Unknown")
    return df


if __name__ == "__main__":
    from FinancialAnomaly.shared.data_loader import load_sanctioned_works

    d = load_sanctioned_works()
    d = add_work_type(d)

    print(f"Distinct work_type categories: {d['work_type'].nunique()}")
    print(f"Rows that fell back to 'Unknown' or work_category: "
          f"{(d['work_type'] == 'Unknown').sum()} unknown")
    print("\nTop 10 by count:")
    print(d["work_type"].value_counts().head(10).to_string())