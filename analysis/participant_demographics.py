"""
Participant Demographics - Results Summary.

============================================
Extracts and summarises participant characteristics for the results section.
Produces group-level descriptive statistics, between-group comparisons,
and a publication-ready demographics table.
"""

# %% Setup - Add project root to path
import os
import sys
from pathlib import Path

# Add project root to Python path and set as working directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from utils import Analysis_utils, Subject_utils  # noqa: E402

subject_utils = Subject_utils()
analysis_utils = Analysis_utils()

# %% Configuration
DEMOGRAPHICS_CSV = Path("data/subject_demographics.csv")
SAVE_TABLE = True
OUTPUT_DIR = Path("data/plot_results/participant_demographics")

if SAVE_TABLE:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% Load demographics and filter to subjects included in the analysis
# -----------------------------------------------------------------
# Only keep subjects whose simulation directory exists (i.e., are processed)
included_subjects = subject_utils.get_all_subjects()  # e.g. ['009', '012', ...]

# Parse the integer IDs used in the demographics CSV (no leading zeros)
included_ids_int = [str(int(s)) for s in included_subjects]

df_all = pd.read_csv(DEMOGRAPHICS_CSV)

# Normalise the Subject column to plain integers for matching
df_all["Subject_str"] = df_all["Subject"].astype(str).str.strip()

df = df_all[df_all["Subject_str"].isin(included_ids_int)].copy()

print(f"Subjects in simulation directory : {len(included_subjects)}")
print(f"Subjects matched in demographics : {len(df)}")
print()

# %% Map coded values to readable labels
# Sex: 1 = Male, 2 = Female
df["Sex_label"] = df["Sex"].map({1: "Male", 2: "Female"})


# %% Helper: mean ± SD string
def mean_sd(series: pd.Series, decimals: int = 1) -> str:
    """Return 'mean ± SD' string, ignoring NaN values."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return "N/A"
    return f"{series.mean():.{decimals}f} ± {series.std():.{decimals}f}"


def median_iqr(series: pd.Series, decimals: int = 1) -> str:
    """Return 'median [Q1–Q3]' string, ignoring NaN values."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return "N/A"
    q1, med, q3 = series.quantile([0.25, 0.5, 0.75])
    return f"{med:.{decimals}f} [{q1:.{decimals}f}–{q3:.{decimals}f}]"


def count_pct(series: pd.Series, value) -> str:
    """Return 'n (%)' string for a categorical value."""
    n = (series == value).sum()
    pct = 100 * n / len(series) if len(series) > 0 else 0
    return f"{n} ({pct:.0f}%)"


# %% Between-group comparison helpers
def independent_t_or_mw(group_a: pd.Series, group_b: pd.Series):
    """
    Run an independent-samples t-test or Mann-Whitney U depending on normality
    (Shapiro-Wilk; alpha = 0.05).  Returns (test_name, statistic, p_value).
    """
    a = pd.to_numeric(group_a, errors="coerce").dropna().values
    b = pd.to_numeric(group_b, errors="coerce").dropna().values

    if len(a) < 3 or len(b) < 3:
        return ("N/A", np.nan, np.nan)

    _, p_a = stats.shapiro(a)
    _, p_b = stats.shapiro(b)

    if p_a > 0.05 and p_b > 0.05:
        stat, p = stats.ttest_ind(a, b)
        return ("t-test", stat, p)
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return ("Mann-Whitney U", stat, p)


def chi2_test(series: pd.Series, group_col: pd.Series):
    """Chi-squared test for a categorical variable across groups."""
    ct = pd.crosstab(series, group_col)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return ("Chi²", chi2, p)


# %% Overall sample descriptive statistics
print("=" * 60)
print("OVERALL SAMPLE")
print("=" * 60)
print(f"  N                : {len(df)}")
print(f"  Age (yr)         : {mean_sd(df['Age'])}")
print(f"  Sex              : {count_pct(df['Sex_label'], 'Male')} male, {count_pct(df['Sex_label'], 'Female')} female")
print(f"  Weight (kg)      : {mean_sd(df['Weight'])}")
print(f"  Height (cm)      : {mean_sd(df['Height'])}")
print(f"  BMI (kg/m²)      : {mean_sd(df['BMI'])}")
print()

# %% Sex breakdown (Male vs Female)
sex_data: dict[str, pd.DataFrame] = {
    "Male": df[df["Sex_label"] == "Male"],
    "Female": df[df["Sex_label"] == "Female"],
}

print("=" * 60)
print("SEX BREAKDOWN")
print("=" * 60)
for sex_label, sdf in sex_data.items():
    print(f"\n  {sex_label}  (n = {len(sdf)})")
    print(f"    Age (yr)        : {mean_sd(sdf['Age'])}")
    print(f"    Weight (kg)     : {mean_sd(sdf['Weight'])}")
    print(f"    Height (cm)     : {mean_sd(sdf['Height'])}")
    print(f"    BMI (kg/m²)     : {mean_sd(sdf['BMI'])}")
print()

# %% Between-sex statistical comparisons
print("=" * 60)
print("BETWEEN-SEX COMPARISONS: Male vs Female")
print("=" * 60)

continuous_vars = {
    "Age (yr)": "Age",
    "Weight (kg)": "Weight",
    "Height (cm)": "Height",
    "BMI (kg/m²)": "BMI",
}

sex_comparison_rows = []
for label, col in continuous_vars.items():
    test_name, stat, p = independent_t_or_mw(sex_data["Male"][col], sex_data["Female"][col])
    sig = "*" if (not np.isnan(p) and p < 0.05) else ""
    print(f"  {label:<20}: {test_name}, p = {p:.3f} {sig}")
    sex_comparison_rows.append(
        {
            "Variable": label,
            "Male (mean ± SD)": mean_sd(pd.to_numeric(sex_data["Male"][col], errors="coerce")),
            "Female (mean ± SD)": mean_sd(pd.to_numeric(sex_data["Female"][col], errors="coerce")),
            "Test": test_name,
            "p-value": f"{p:.3f}" if not np.isnan(p) else "N/A",
            "Significant": sig,
        }
    )
print()
print(pd.DataFrame(sex_comparison_rows).to_string(index=False))

# %% Normalized fiber length responders vs non-responders
# Uses hamstring muscles (bflh, semiten, semimem) by default.
# A responder has a positive % change in peak normalized fiber length baseline → last trial.
responders_dict = analysis_utils.get_norm_fiber_length_responders_ids()
non_responders_dict = analysis_utils.get_norm_fiber_length_non_responders_ids()

# Subjects who respond in ANY of the default muscles
responder_ids_any = sorted(set(str(sid).zfill(3) for ids in responders_dict.values() for sid in ids))
non_responder_ids_any = [s for s in included_subjects if s not in responder_ids_any]

# Subjects who respond in ALL of the default muscles
responder_ids_all = sorted(
    set.intersection(*[set(str(sid).zfill(3) for sid in ids) for ids in responders_dict.values()])
)

print("\n" + "=" * 60)
print("NORM FIBER LENGTH RESPONDERS vs NON-RESPONDERS")
print("=" * 60)
print(f"  Default muscles: {list(responders_dict.keys())}")
print()
print(f"  Responders (any muscle)  : n = {len(responder_ids_any)}, IDs: {responder_ids_any}")
print(f"  Responders (all muscles) : n = {len(responder_ids_all)}, IDs: {responder_ids_all}")
print(f"  Non-responders (any)     : n = {len(non_responder_ids_any)}, IDs: {non_responder_ids_any}")
print()

# Per-muscle counts
print("  Per-muscle responder counts:")
for muscle, ids in responders_dict.items():
    nr_ids = non_responders_dict.get(muscle, [])
    print(f"    {muscle:<12}: {len(ids):>3} responders, {len(nr_ids):>3} non-responders")
print()

# Map responder IDs (no leading zeros in analysis_utils) back to demographics
responder_ids_int = [str(int(s)) for s in responder_ids_any]
non_responder_ids_int = [str(int(s)) for s in non_responder_ids_any]

df_resp = df[df["Subject_str"].isin(responder_ids_int)]
df_non_resp = df[df["Subject_str"].isin(non_responder_ids_int)]

for label, rdf in [("Responders", df_resp), ("Non-responders", df_non_resp)]:
    print(f"  {label}  (n = {len(rdf)})")
    print(f"    Age (yr)        : {mean_sd(rdf['Age'])}")
    print(
        f"    Sex             : {count_pct(rdf['Sex_label'], 'Male')} male, "
        f"{count_pct(rdf['Sex_label'], 'Female')} female"
    )
    print(f"    Weight (kg)     : {mean_sd(rdf['Weight'])}")
    print(f"    Height (cm)     : {mean_sd(rdf['Height'])}")
    print(f"    BMI (kg/m²)     : {mean_sd(rdf['BMI'])}")
    print()

# Between-group comparison: responders vs non-responders
print("=" * 60)
print("RESPONDER vs NON-RESPONDER COMPARISONS")
print("=" * 60)

resp_comparison_rows = []
for label, col in continuous_vars.items():
    test_name, stat, p = independent_t_or_mw(df_resp[col], df_non_resp[col])
    sig = "*" if (not np.isnan(p) and p < 0.05) else ""
    print(f"  {label:<20}: {test_name}, p = {p:.3f} {sig}")
    resp_comparison_rows.append(
        {
            "Variable": label,
            "Responders (mean ± SD)": mean_sd(pd.to_numeric(df_resp[col], errors="coerce")),
            "Non-responders (mean ± SD)": mean_sd(pd.to_numeric(df_non_resp[col], errors="coerce")),
            "Test": test_name,
            "p-value": f"{p:.3f}" if not np.isnan(p) else "N/A",
            "Significant": sig,
        }
    )

# Sex distribution
if len(df_resp) > 0 and len(df_non_resp) > 0:
    resp_sex_series = pd.concat([df_resp["Sex_label"], df_non_resp["Sex_label"]], ignore_index=True)
    resp_group_series = pd.concat(
        [
            pd.Series(["Responder"] * len(df_resp)),
            pd.Series(["Non-responder"] * len(df_non_resp)),
        ],
        ignore_index=True,
    )
    test_name, stat, p = chi2_test(resp_sex_series, resp_group_series)
    sig = "*" if p < 0.05 else ""
    print(f"  {'Sex (M/F)':<20}: {test_name}, p = {p:.3f} {sig}")
    resp_comparison_rows.append(
        {
            "Variable": "Sex (M/F)",
            "Responders (mean ± SD)": f"{count_pct(df_resp['Sex_label'], 'Male')} M, "
            f"{count_pct(df_resp['Sex_label'], 'Female')} F",
            "Non-responders (mean ± SD)": f"{count_pct(df_non_resp['Sex_label'], 'Male')} M, "
            f"{count_pct(df_non_resp['Sex_label'], 'Female')} F",
            "Test": test_name,
            "p-value": f"{p:.3f}",
            "Significant": sig,
        }
    )

print()
print(pd.DataFrame(resp_comparison_rows).to_string(index=False))

# %% Measured / dominant leg counts
print("\n" + "=" * 60)
print("LEG INFORMATION")
print("=" * 60)
print(df["Measured Leg"].value_counts().rename("Measured leg").to_string())
print()
print(df["Dominant Leg "].value_counts().rename("Dominant leg").to_string())

# %% Build publication-ready demographics table
print("\n" + "=" * 60)
print("DEMOGRAPHICS TABLE (publication format)")
print("=" * 60)


def build_row(gdf: pd.DataFrame) -> dict:
    male_n = (gdf["Sex_label"] == "Male").sum()
    female_n = (gdf["Sex_label"] == "Female").sum()
    return {
        "N": len(gdf),
        "Age (yr)": mean_sd(gdf["Age"]),
        "Sex (M/F)": f"{male_n}/{female_n}",
        "Weight (kg)": mean_sd(gdf["Weight"]),
        "Height (cm)": mean_sd(gdf["Height"]),
        "BMI (kg/m²)": mean_sd(gdf["BMI"]),
    }


table_rows = {
    "Overall": build_row(df),
    "Male": build_row(sex_data["Male"]),
    "Female": build_row(sex_data["Female"]),
    "Responders": build_row(df_resp),
    "Non-responders": build_row(df_non_resp),
}

pub_table = pd.DataFrame(table_rows).T
pub_table.index.name = "Group"
print(pub_table.to_string())

if SAVE_TABLE:
    out_path = OUTPUT_DIR / "demographics_table.csv"
    pub_table.to_csv(out_path)
    print(f"\nTable saved to: {out_path}")

# %% Per-subject summary (useful for supplementary material)
per_subject_cols = [
    "Subject_str",
    "Age",
    "Sex_label",
    "Group",
    "Weight",
    "Height",
    "BMI",
    "Alpha angle",
    "Centre edge angle",
    "Measured Leg",
    "Dominant Leg ",
]

existing_cols = [c for c in per_subject_cols if c in df.columns]
per_subject_df = df[existing_cols].copy()
per_subject_df.rename(columns={"Subject_str": "Subject", "Sex_label": "Sex"}, inplace=True)
per_subject_df = per_subject_df.sort_values("Subject")

print("\n" + "=" * 60)
print("PER-SUBJECT SUMMARY")
print("=" * 60)
print(per_subject_df.to_string(index=False))

if SAVE_TABLE:
    out_path = OUTPUT_DIR / "per_subject_demographics.csv"
    per_subject_df.to_csv(out_path, index=False)
    print(f"\nPer-subject table saved to: {out_path}")

# %%
