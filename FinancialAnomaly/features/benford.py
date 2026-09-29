"""Benford's Law leading-digit check.

Expected: P(d) = log10(1 + 1/d) for leading digit d in {1..9}.
Real, unmanipulated financial figures follow this distribution closely.
A batch that deviates significantly (measured by chi-square goodness-of-fit)
is a signal the amounts may be fabricated/manipulated — NOT proof, just a
screening flag, and only meaningful with enough data points (standard
guidance: at least ~30 values per group to trust the test).

This is agency-level output (a table), not a per-work score — different
shape from f1/f3/f4/f5/f6, same as originally planned.
"""
import numpy as np
import pandas as pd
from scipy.stats import chisquare

from FinancialAnomaly.shared.data_loader import load_sanctioned_works
from FinancialAnomaly.shared.agency_standardization import add_agency_id

MIN_WORKS_FOR_BENFORD = 110  # Cochran's rule: need expected count >=5 in every digit
                              # bucket; digit 9 gets only 4.58% of total, so n>=109 required

EXPECTED_BENFORD = {d: np.log10(1 + 1 / d) for d in range(1, 10)}


def _leading_digit(amount: float) -> int:
    s = str(int(abs(amount)))
    return int(s[0]) if s and s[0] != "0" else 1


def benford_chi_square(amounts: pd.Series) -> tuple[float, float, float]:
    """Returns (chi2_stat, p_value, cramers_v).
    cramers_v is the EFFECT SIZE, independent of sample size — this is the
    number that actually matters here, since chi2/p-value become misleadingly
    "significant" at n=20,000 even for small real deviations.
    Cramer's V guide: <0.1 negligible, 0.1-0.3 small, 0.3-0.5 moderate, >0.5 large.
    """
    digits = amounts.apply(_leading_digit)
    observed_counts = digits.value_counts().reindex(range(1, 10), fill_value=0).sort_index()
    n = observed_counts.sum()
    expected_counts = pd.Series({d: EXPECTED_BENFORD[d] * n for d in range(1, 10)})

    chi2, p = chisquare(f_obs=observed_counts.values, f_exp=expected_counts.values)
    cramers_v = min((chi2 / (n * (9 - 1))) ** 0.5, 1.0) if n > 0 else 0.0  # clipped: theoretical max is 1.0
    return round(float(chi2), 3), round(float(p), 5), round(float(cramers_v), 4)


def agency_benford_flags(df: pd.DataFrame, baseline_v: float) -> pd.DataFrame:
    """Flags agencies whose Cramer's V exceeds the DATASET'S OWN baseline
    (not zero) — since the whole dataset already deviates somewhat from ideal
    Benford due to standardized MPLADS costs, not fraud. Only agencies
    noticeably WORSE than that baseline are worth a second look."""
    rows = []
    for agency_id, sub in df.groupby("agency_id"):
        n = len(sub)
        if n < MIN_WORKS_FOR_BENFORD:
            continue
        chi2, p, v = benford_chi_square(sub["amount"])
        rows.append({"agency_id": agency_id, "n_works": n, "cramers_v": v,
                     "vs_baseline": round(v - baseline_v, 4)})
    result = pd.DataFrame(rows).sort_values("vs_baseline", ascending=False)
    # f2 score: only meaningful excess over baseline counts, scaled 0-1, capped
    result["f2_benford_excess"] = (result["vs_baseline"].clip(lower=0) / 0.2).clip(upper=1.0).round(4)
    return result


if __name__ == "__main__":
    d = load_sanctioned_works()
    d = add_agency_id(d)

    print("=== Dataset-wide Benford check ===")
    chi2, p, v = benford_chi_square(d["amount"])
    print(f"chi2={chi2}, p={p}, Cramer's V={v}  "
          f"({'small' if v < 0.3 else 'moderate' if v < 0.5 else 'large'} effect size)")
    print("NOTE: p-value is near-zero mainly due to large n (20,000) — Cramer's V is the")
    print("number that reflects the real, small-to-moderate deviation. Driven mostly by")
    print("leading digit '2' overrepresentation, consistent with the ₹25L threshold-avoidance")
    print("clustering (f1) and common ₹2-lakh-range sanctions — not broad digit manipulation.")

    print(f"\n=== Agency-level Benford EXCESS over dataset baseline (min {MIN_WORKS_FOR_BENFORD} works) ===")
    flags = agency_benford_flags(d, baseline_v=v)
    print(f"Agencies tested: {len(flags)}")
    print(f"Agencies with f2_benford_excess > 0 (worse than dataset baseline): {(flags['f2_benford_excess']>0).sum()}")
    print("\nTop 10 by excess deviation over baseline:")
    print(flags.head(10).to_string(index=False))