"""
Responder Analysis - Fatigue Response Classification.

====================================================
Classify subjects as responders/non-responders based on fatigue metrics
and analyze differences between groups.
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

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src import Simulation_analyzer  # noqa: E402
from utils import (  # noqa: E402
    Analysis_utils,
    Data_utils,
    File_utils,
    Plot_utils,
    Subject_utils,
)

# Initialize utilities
plots = Plot_utils()
file_utils = File_utils()
subject_utils = Subject_utils()
data_utils = Data_utils()
analysis_utils = Analysis_utils()
tools = Simulation_analyzer()

# %% Configuration
SAVE_FIGURES: bool = True  # Set to True to save figures
OUTPUT_DIR = Path("data/plot_results/responder_analysis")
SUBJECT_SPECIFIC_DIR = Path(OUTPUT_DIR / "subject_specific_plots")


# Create directories if they don't exist
if SAVE_FIGURES:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBJECT_SPECIFIC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directories created at: {OUTPUT_DIR}")

# %%
responders_dict = analysis_utils.get_norm_fiber_length_responders_ids()
speed_df = analysis_utils.get_speed_df()

# Get all unique responders across all muscles
responders = list(set().union(*responders_dict.values()))
# %%

responders_speed_df = speed_df[speed_df["subject_id"].isin(responders)].copy()


# Add column showing which muscle(s) each subject responded to
def get_responding_muscles(subject_id):
    muscles = []
    for muscle, responder_ids in responders_dict.items():
        if subject_id in responder_ids:
            muscles.append(muscle)
    return ", ".join(muscles) if muscles else "none"


responders_speed_df.insert(
    1,
    "responding_muscles",
    responders_speed_df["subject_id"].apply(get_responding_muscles),
)

# Filter for only change columns
change_cols = ["subject_id", "responding_muscles"] + [
    col for col in responders_speed_df.columns if "change" in col.lower()
]
responders_speed_df = responders_speed_df[change_cols]
responders_speed_df.head()
# %%
# === Master plots with seperate responders ===
# Compute SPM analysis once for all plots (uses all subjects)
print("Computing SPM analysis...")
spm_sig_idx = analysis_utils.get_spm_sig_index(p=0.0042)

# %%
try:
    for responder in responders:
        result = plots.plot_master_plot_before_after_all_exclude_responders(
            responder=responder,
            spm_sig_idx=spm_sig_idx,
        )
        fig, ax = result
        if fig is not None and SAVE_FIGURES:
            fig.savefig(
                SUBJECT_SPECIFIC_DIR / f"master_plot_exclude_responders_{responder}.png",
                dpi=300,
            )
            print(
                f"Master plot saved to: {SUBJECT_SPECIFIC_DIR / f'master_plot_exclude_responders_{responder}.png'}",
            )
except RuntimeError as e:
    print(f"Error generating master plot: {e}")

# %% # %% Color coded individual plots non responders vs responders pre (INTERACTIVE)
fig = plots.plot_responder_comparison_individual(trial_type="pre", interactive=True)
if fig is not None and SAVE_FIGURES:
    output_path = OUTPUT_DIR / "master_plot_individual_groups_pre_interactive.html"
    fig.write_html(str(output_path))
    print(f"Interactive plot saved to: {output_path}")

# %% Color coded individual plots non responders vs responders pre (STATIC)
fig, ax = plots.plot_responder_comparison_individual(
    trial_type="pre",
    interactive=False,
)
if fig is not None and SAVE_FIGURES:
    output_path = OUTPUT_DIR / "master_plot_individual_groups_pre.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Static plot saved to: {output_path}")
    plt.close(fig)

# %% Color coded individual plots non responders vs responders post (INTERACTIVE)
fig = plots.plot_responder_comparison_individual(trial_type="post", interactive=True)
if fig is not None and SAVE_FIGURES:
    output_path = OUTPUT_DIR / "master_plot_individual_groups_post_interactive.html"
    fig.write_html(str(output_path))
    print(f"Interactive plot saved to: {output_path}")

# %% Color coded individual plots non responders vs responders post (STATIC)
fig, ax = plots.plot_responder_comparison_individual(
    trial_type="post",
    interactive=False,
)
if fig is not None and SAVE_FIGURES:
    output_path = OUTPUT_DIR / "master_plot_individual_groups_post.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Static plot saved to: {output_path}")
    plt.close(fig)
# %% Sex responder comparison
# Stats-only analysis (no plots):
# 1) Is sex distribution different between responders and non-responders?
# 2) Within responders, are Male vs Female change metrics different?
# 3) Within non-responders, are Male vs Female change metrics different?
DEMOGRAPHICS_CSV = Path("data/subject_demographics.csv")


def count_pct(series, value: str) -> str:
    """Return 'n (%)' for a categorical level."""
    n = int((series == value).sum())
    pct = (100 * n / len(series)) if len(series) > 0 else 0
    return f"{n} ({pct:.0f}%)"


def pct(numerator: int, denominator: int) -> float:
    """Return percentage with safe divide."""
    if denominator == 0:
        return np.nan
    return (numerator / denominator) * 100


included_subjects = subject_utils.get_all_subjects(use_leading_zeros=False)
demographics_df = pd.read_csv(DEMOGRAPHICS_CSV)
demographics_df["Subject_str"] = demographics_df["Subject"].astype(str).str.strip()
demographics_df = demographics_df[demographics_df["Subject_str"].isin(included_subjects)].copy()
demographics_df["Sex_label"] = demographics_df["Sex"].map({1: "Male", 2: "Female"})

# Responder status based on ANY muscle responder criterion in default muscles.
responder_ids_any = sorted(set(str(sid) for ids in responders_dict.values() for sid in ids))

demographics_df["Responder_status"] = np.where(
    demographics_df["Subject_str"].isin(responder_ids_any),
    "Responder",
    "Non-responder",
)

df_resp = demographics_df[demographics_df["Responder_status"] == "Responder"].copy()
df_non_resp = demographics_df[demographics_df["Responder_status"] == "Non-responder"].copy()

print("\n" + "=" * 60)
print("SEX DIFFERENCE ANALYSIS (STATS ONLY)")
print("=" * 60)

print("\nSex distribution by responder status")
print(
    f"  Responders     : {count_pct(df_resp['Sex_label'], 'Male')} Male, {count_pct(df_resp['Sex_label'], 'Female')} Female"
)
print(
    f"  Non-responders : {count_pct(df_non_resp['Sex_label'], 'Male')} Male, "
    f"{count_pct(df_non_resp['Sex_label'], 'Female')} Female",
)

sex_by_status_ct = pd.crosstab(demographics_df["Sex_label"], demographics_df["Responder_status"])
if sex_by_status_ct.shape == (2, 2):
    odds_ratio, p_fisher = stats.fisher_exact(sex_by_status_ct)
    print(f"  Fisher exact test (Sex x Responder status): p = {p_fisher:.3f}")
else:
    chi2, p_chi2, _, _ = stats.chi2_contingency(sex_by_status_ct)
    print(f"  Chi-squared test (Sex x Responder status): p = {p_chi2:.3f}")

male_total = int((demographics_df["Sex_label"] == "Male").sum())
female_total = int((demographics_df["Sex_label"] == "Female").sum())
male_resp_any = int((df_resp["Sex_label"] == "Male").sum())
female_resp_any = int((df_resp["Sex_label"] == "Female").sum())

print("\nResponder rate by sex (ANY muscle responder definition)")
print(
    f"  Male responders   : {male_resp_any}/{male_total} ({pct(male_resp_any, male_total):.1f}%)",
)
print(
    f"  Female responders : {female_resp_any}/{female_total} ({pct(female_resp_any, female_total):.1f}%)",
)

print("\nPer-muscle responder rate by sex")
per_muscle_rows = []

for muscle, muscle_responder_ids in responders_dict.items():
    muscle_responder_ids_str = set(str(sid) for sid in muscle_responder_ids)

    muscle_resp_df = demographics_df[demographics_df["Subject_str"].isin(muscle_responder_ids_str)]
    muscle_resp_male_n = int((muscle_resp_df["Sex_label"] == "Male").sum())
    muscle_resp_female_n = int((muscle_resp_df["Sex_label"] == "Female").sum())

    row = {
        "muscle": muscle,
        "n_responders": len(muscle_resp_df),
        "responders_male_n": muscle_resp_male_n,
        "responders_female_n": muscle_resp_female_n,
        "responders_male_pct_of_all_males": round(pct(muscle_resp_male_n, male_total), 1),
        "responders_female_pct_of_all_females": round(pct(muscle_resp_female_n, female_total), 1),
    }
    per_muscle_rows.append(row)

per_muscle_sex_df = pd.DataFrame(per_muscle_rows)
print(per_muscle_sex_df.to_string(index=False))

# %% ================================================
# %%    FINALIZATION
# %% ================================================

print("\n" + "=" * 60)
print("Fatigue analysis complete!")
print(f"Figures saved to: {OUTPUT_DIR.absolute()}")
print("=" * 60)

# %% Generate HTML index
if SAVE_FIGURES:
    print("\nGenerating HTML index...")
    html_path = plots.generate_html_index(
        OUTPUT_DIR,
        title="Responder Analysis - Plot Overview",
        columns=3,
        analysis_type="responder_analysis",
    )
    print(f"📄 HTML index created: {html_path.absolute()}")

    # Generate central hub
    hub_path = plots.generate_hub_index()
    print(f"🎯 Central hub updated: {hub_path.absolute()}")
    print("Open hub in browser to access all analyses")
# %%
