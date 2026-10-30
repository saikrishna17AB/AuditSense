

---

## Repository layout

```
AuditSense/
├── Datasets/                  Public MPLADS CSV exports (see "Data" below)
├── Time_Cost_Overrun/         Module 1 - time overrun + fund under-utilization
├── FinancialAnomaly/          Module 2 - financial anomaly detection
├── DuplicateWork_Detection/   Module 3 - duplicate work detection
├── SuspiciousAgencyFlag/      Module 4 - vendor/agency risk (planned)
└── RuleBasedFeatures/         Rule-based compliance checks (planned)
```

| Module | Status | Method |
|---|---|---|
| 1. Time / Cost Overrun | Implemented, validated | ANN + SVM, segment-routed |
| 2. Financial Anomaly | Implemented, validated | Isolation Forest + Autoencoder, agreement tiers |
| 3. Duplicate Detection | Implemented (SBERT step runs locally) | Multilingual SBERT + weighted similarity |
| 4. Vendor / Agency Risk | Planned | Gradient Boosting + Louvain graph clustering |

---

## Module 1 - Time & Cost Overrun (`Time_Cost_Overrun/`)

Predicts, at sanction stage, whether a work will run late or leave funds unspent. Trained on **13,119 works** whose sanction record (`Works_Sanctioned.csv`) joins to a completion record (`Works Completed.csv`) on the serial in the `Work` ID.

**Why not cost overrun?** In this data `disbursed_amount` never exceeds `sanctioned_amount` (0 of 13,119 works). The Guidelines (Para 4.16/5.3) require unspent savings to be refunded, so overspending cannot occur. Module 1 predicts the risks that do exist:

| Target | Definition | Base rate |
|---|---|---|
| Time overrun | `execution_days > 365` (Guidelines Para 3.13 norm) | 21.0% |
| Under-utilization | more than 1% of the sanctioned amount never disbursed | 6.2% |

**Features:** sanctioned amount, recommendation-to-sanction gap, season, state, work type, and agency track record. Track record has two parts: `prior_reliability_rate` from FY23-24 data in `MPLADS.csv` (known history, leakage-safe) and each agency's rate across the current dataset (a disclosed simplification, because a strictly leakage-safe version is ~99% cold-start in a ~24-month window).

**Routing.** Work types with at least 150 works go to an ANN; the 77 thinner work types go to a pooled SVM.

**Equations**

Numeric inputs are standardized:

```
x' = (x − μ) / σ
```

ANN forward pass (16 -> 8 -> 1 units):

```
h1 = ReLU(W1·x  + b1)
h2 = ReLU(W2·h1 + b2)
ŷ  = σ(W3·h2 + b3),    σ(z) = 1 / (1 + e^(−z))
```

Class-weighted binary cross-entropy (the minority class is up-weighted by the class-count ratio):

```
L = −(1/n) · Σᵢ  w(yᵢ) · [ yᵢ·log(ŷᵢ) + (1 − yᵢ)·log(1 − ŷᵢ) ]
```

RBF-kernel SVM (fallback), with `class_weight="balanced"` and Platt-scaled probabilities:

```
f(x)     = Σᵢ αᵢ·yᵢ·K(xᵢ, x) + b
K(xᵢ, x) = exp( −γ · ‖xᵢ − x‖² )
```

Evaluation metric:

```
AUC = P( ŷ(x₊) > ŷ(x₋) )
```

**Results (held-out test split)**

| Target | ANN AUC | SVM AUC | Leave-one-out corrected (ANN / SVM) |
|---|---|---|---|
| Time overrun | 0.967 | 0.905 | 0.960 / 0.890 |
| Under-utilization | 0.955 | 0.948 | 0.935 / 0.918 |

The corrected column removes each row's own outcome from its agency-rate feature to rule out self-leakage. Lead with AUC: under-utilization has a 6.2% base rate, so "never at risk" already scores 93.8% accuracy.

```bash
python -m Time_Cost_Overrun.data_loader
python -m Time_Cost_Overrun.features.prior_history
python -m Time_Cost_Overrun.leakage_safe.build_features
python -m Time_Cost_Overrun.rates.model.time_overrun_model
python -m Time_Cost_Overrun.automatically.underutilization
```

---

## Module 2 - Financial Anomaly Detection (`FinancialAnomaly/`)

Flags unusual sanction patterns on `Works_Sanctioned.csv` (20,000 works) without any fraud labels, since none exist for MPLADS.

**Signals (per work unless noted)**

| ID | Signal | Formula |
|---|---|---|
| f1 | Threshold avoidance (just below Rs 25 lakh / Rs 15 lakh) | f1 = max(0, 1 − (T − a) / 50,000) when T − 50,000 < a < T, else 0 (a = amount, T = threshold) |
| f3 | Round-number score | graded by divisibility: 1.0 (multiple of Rs 1 lakh), 0.8 (Rs 50k), 0.6 (Rs 10k), 0.3 (Rs 1k), 0.1 (Rs 100) |
| f4 | Agency cost trend | slope of the agency's monthly mean amount, divided by that agency's amount standard deviation |
| f5 | Fast approval | f5 = 1 − d / 7 when d < 7 days between recommendation and sanction, else 0 |
| f6 | Agency fingerprint | f6 = abs(a − μ_agency) / σ_agency |
| - | Benford's Law (agency level, not per work) | below |

Benford's Law expects leading digit d (1 to 9) with probability P(d) = log10(1 + 1/d). Deviation is measured by chi-square and reported as an effect size (Cramér's V), since p-values become meaningless at large n:

```
χ² = Σ (O_d − E_d)² / E_d        for d = 1..9
V  = sqrt( χ² / (n · (k − 1)) )    with k = 9 digit bins
```

Only agencies with at least 110 works are tested.

**Models.** Isolation Forest scores how quickly a work is isolated by random splits; an autoencoder scores how badly a work is reconstructed after learning "normal" patterns.

```
s(x, n) = 2^( −E(h(x)) / c(n) )
c(n)    = 2·H(n − 1) − 2(n − 1)/n        (H = harmonic number)
```

```
A(x) = ‖x − x̂‖²
```

Both are rescaled to 0-100. Decisions use **agreement tiers**, so disagreement between the models stays visible:

| Tier | Rule |
|---|---|
| High-confidence flag | both scores >= 70 |
| Needs review | one score >= 70 |
| Not flagged | neither |

A fused score is used for ranking only:

```
fused = 0.5 · s_IF + 0.5 · s_AE
```

This is the α = 0.5 case of the general form `100 × [α·N(s_IF) + (1 − α)·N(A)]`. α is heuristic because no labels exist to tune it.

**Validation (synthetic anomaly injection)**

| Test | Result |
|---|---|
| Injected anomalies, AUC | 0.995 |
| Injected anomalies, Recall@300 | 60.3% |
| Real data, works flagged "Needs review" | 101 of 20,000 (0.5%) |

No accuracy against real fraud is claimed. Injected anomalies are built from the same signals the model uses, so this shows the pipeline separates planted anomalies, not that it detects fraud.

```bash
python -m FinancialAnomaly.model.fusion_model
```

---

## Module 3 - Duplicate Work Detection (`DuplicateWork_Detection/`)

Finds works that may be the same asset funded twice, on `MPLADS.csv` (60,359 recommendations; semicolon-delimited). The public data has no GPS coordinates, so location similarity uses village / block / constituency fields.

**Pipeline (`notebooks/01`-`10`):** inspect -> clean -> block analysis -> **candidate pairs** (1,265,292, down from roughly 1.8 billion possible pairs) -> multilingual SBERT embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, 384-D) -> semantic similarity -> feature generation -> labelling set -> scoring -> mismatch inspection.

**Equations**

Semantic similarity between work-description embeddings:

```
S_text = (Eᵢ · Eⱼ) / (‖Eᵢ‖ · ‖Eⱼ‖)
```

Temporal similarity from the recommendation-date gap Δt in days:

```
S_time = exp( −Δt / 180 )
```

Geographic similarity: a weighted average of location-field similarities (exact matches for some fields, a fuzzy text ratio for village names). If GPS coordinates become available, the Haversine form applies:

```
S_geo = exp( −d / 5 )       d in km
d     = 2R · atan2( √a, √(1 − a) ),    R = 6371 km
a     = sin²(Δφ/2) + cos φ₁ · cos φ₂ · sin²(Δλ/2)
```

Structured similarity: mean of exact matches over the comparable categorical fields (S_k = 1 if equal, else 0):

```
S_struct = (1/m) · Σ S_k          m = number of comparable fields
```

Overall duplicate score and classification:

```
S_D = 0.50·S_text + 0.20·S_geo + 0.15·S_time + 0.15·S_struct
```

| Score | Class |
|---|---|
| S_D ≥ 0.75 | Potential Duplicate |
| 0.55 ≤ S_D < 0.75 | Needs Review |
| S_D < 0.55 | Low likelihood |

The weights are set by judgment, not fitted: no labelled duplicate pairs exist. Weights and thresholds should be tuned on a hand-labelled sample before production use. Uncertain pairs are routed to human verification instead of a forced yes/no.

```bash
# from DuplicateWork_Detection/, in order (step 05 downloads the SBERT model on first run)
python DuplicateWork_Detection/notebooks/01_data_inspection.py
# ... through 10
```

---

## Planned

- **Module 4 - Vendor / Agency Risk (`SuspiciousAgencyFlag/`):** gradient boosting on agency red-flag indicators plus Louvain community detection on agency-vendor links. Public data has no vendor or ownership fields, so this needs richer data.
- **Rule-based compliance (`RuleBasedFeatures/`):** checks against the Guidelines, for example the 75-day sanction window (Para 3.12), prohibited work categories (Annexure II), and the Rs 25 lakh inspection threshold (Para 4.17).
- **Aggregation + dashboard:** composite risk index with SHAP-style explanations, role-based views for MPs, nodal authorities, district authorities and the Ministry.

---

## Data

Place the CSV exports in `Datasets/`:

`Works_Sanctioned.csv` · `Works Completed.csv` · `MPLADS.csv` · `MPLADS_ADR_PRS_15thLS.csv` · `Allocated Limit for Honble MPs.csv` · `Amount consented for Calamity.csv` · `RS-Session-251-AU3002-Annexure-I.csv`

These are public MPLADS exports, not a live eSAKSHI extract. Findings describe this dataset, not the scheme as a whole.

## Requirements

Python 3.10+, `pandas`, `numpy`, `scipy`, `scikit-learn`, `tensorflow`, `sentence-transformers`, `rapidfuzz`.

## Limitations

- Flags are statistical signals for review. They are not findings of wrongdoing.
- Modules 2 and 3 have no labelled ground truth; their validation is synthetic or proxy-based.
- Module 3's weights and Module 2's α are heuristic.
- Module 1's current-period agency rate is a disclosed simplification (see above).
- Track-record signals are limited by a short (~24-month) data window.

## Team

