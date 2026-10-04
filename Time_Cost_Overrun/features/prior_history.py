"""Agency PRIOR-PERIOD reliability, sourced from Datasets/MPLADS.csv.

Datasets/MPLADS.csv (60,359 rows, semicolon-delimited) covers
recommendations from Apr 2023 - Mar 2024 - entirely BEFORE
Works_Sanctioned.csv's window starts (~Jul 2024). So an agency's
completion rate in MPLADS.csv is a genuinely leakage-safe prior.

MPLADS.csv's IDA field has no outer district label, unlike
Works_Sanctioned.csv, so FinancialAnomaly's agency_standardization
(which keys off the OUTER label) cannot match them. This module uses
the INNER text instead, verified to recover 460 of 544 Works_Sanctioned
agencies (84.6%). This key is used ONLY to join MPLADS.csv's history
onto this module's data - it does not replace FinancialAnomaly's agency_id.
"""
import re
import pandas as pd

MPLADS_PATH = "Datasets/MPLADS.csv"
ROLE_RULES = [
    (r"PLANNING", "DISTRICT PLANNING OFFICER"),
    (r"MAGI", "DISTRICT MAGISTRATE"),
    (r"COLLECT", "DISTRICT COLLECTOR"),
    (r"COMM?I|\bDC\b", "DEPUTY COMMISSIONER"),
]


def _inner_key(raw) -> str:
    if not isinstance(raw, str):
        return "UNKNOWN"
    name = re.sub(r"\s+", " ", raw.upper().strip())
    m = re.search(r"\((.*)\)", name)
    inner = (m.group(1) if m else name).replace("_IDA", "").strip()
    role = next((canon for pat, canon in ROLE_RULES if re.search(pat, inner)), "UNKNOWN_ROLE")
    return f"{inner} | {role}"


def load_prior_reliability(mplads_path: str = MPLADS_PATH) -> pd.DataFrame:
    mp = pd.read_csv(mplads_path, sep=";")
    mp["inner_key"] = mp["IDA"].apply(_inner_key)

    g = mp.groupby("inner_key").agg(
        prior_n_recommended=("STATUS", "size"),
        prior_n_completed=("STATUS", lambda s: (s == "Completed").sum()),
    )
    g["prior_reliability_rate"] = (g["prior_n_completed"] / g["prior_n_recommended"]).fillna(0)
    return g.reset_index()


def add_prior_reliability(df: pd.DataFrame, raw_col: str = "agency_raw") -> pd.DataFrame:
    df = df.copy()
    df["inner_key"] = df[raw_col].apply(_inner_key)
    prior = load_prior_reliability()[["inner_key", "prior_reliability_rate", "prior_n_recommended"]]
    df = df.merge(prior, on="inner_key", how="left")

    fallback = prior["prior_reliability_rate"].mean()
    n_missing = df["prior_reliability_rate"].isna().sum()
    df["prior_reliability_rate"] = df["prior_reliability_rate"].fillna(fallback)
    df["prior_history_available"] = df["prior_n_recommended"].notna().astype(int)
    df = df.drop(columns=["inner_key", "prior_n_recommended"])

    if n_missing:
        print(f"[prior_history] {n_missing} of {len(df)} rows had no prior-year match "
              f"(filled with dataset-wide fallback {fallback:.3f})")
    return df


if __name__ == "__main__":
    prior = load_prior_reliability()
    print("Agencies with FY23-24 history:", len(prior))
    print(prior["prior_reliability_rate"].describe())
    print()
    print("Match test against Works_Sanctioned.csv:")
    ws = pd.read_csv("Datasets/Works_Sanctioned.csv")
    ws_ids = ws["IDA"].dropna().apply(_inner_key)
    overlap = set(ws_ids) & set(prior["inner_key"])
    print(f"{len(overlap)} of {ws_ids.nunique()} Works_Sanctioned agencies found ({100*len(overlap)/ws_ids.nunique():.1f}%)")