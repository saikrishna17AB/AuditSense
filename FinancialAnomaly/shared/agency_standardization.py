"""Adds agency_id: a canonical ID per implementing agency.

IDA strings are consistently formatted 'PLACE(TITLE PLACE_IDA)', e.g.
'KHERI(DISTRICT MAGISTRAE KHERI_IDA)' — a district name, then a
parenthetical descriptor that repeats the same district name. Spelling
inconsistencies in this data live in the DESCRIPTOR (e.g. 'MAGISTRAE' vs
'MAGISTRATE'), not in the leading place name, which is the part every
other view of this data (state summaries, dashboards) actually keys on.

First attempt at this used fuzzy string matching (difflib) across the
FULL raw string, which seemed reasonable but was tested against the real
data and found to be actively wrong: it merged genuinely different
districts that happen to share letters — 'ALWAR' with 'JHALAWAR',
'NAGPUR' with 'NAGAUR', 'RAIPUR' with 'JAIPUR' and 'UDAIPUR' — because
most of each string is identical boilerplate, so overall similarity stays
high even when the district itself is completely different. That's worse
than not merging at all: it would pool unrelated agencies' works into one
baseline and corrupt every agency-relative feature (f4, f6, per-agency
Benford).

Fix, and the one actually used here: key on the leading place name only,
exact match after normalizing case/whitespace. Confirmed against this
dataset — 543 raw IDA strings collapse to 542 canonical agencies this way,
with the one real duplicate (a descriptor-spelling variant) merged
correctly and zero false merges across different districts. No fuzzy
matching is applied, specifically because it isn't safe here.
"""
import re

import pandas as pd


def _place_key(raw: str) -> str:
    match = re.match(r"^([^(]+)\(", raw)
    place = match.group(1) if match else raw
    place = place.upper()
    place = re.sub(r"[^A-Z0-9 ]", " ", place)
    place = re.sub(r"\s+", " ", place).strip()
    return place


def add_agency_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["agency_id"] = df["agency_raw"].apply(_place_key)
    return df


if __name__ == "__main__":
    from FinancialAnomaly.shared.data_loader import load_sanctioned_works

    d = load_sanctioned_works()
    n_raw = d["agency_raw"].nunique()
    d = add_agency_id(d)
    n_canonical = d["agency_id"].nunique()

    print(f"Raw distinct IDA strings: {n_raw}")
    print(f"Canonical agency_id: {n_canonical} ({n_raw - n_canonical} merged)")

    merged = d.groupby("agency_id")["agency_raw"].nunique().reset_index(name="n_variants")
    merged = merged[merged["n_variants"] > 1]
    print(f"\nagency_id(s) absorbing >1 raw spelling: {len(merged)}")
    for agency_id in merged["agency_id"]:
        variants = d[d["agency_id"] == agency_id]["agency_raw"].unique()
        print(f"  {agency_id} -> {list(variants)}")