"""
Shared biomechanics computation helpers.

This module provides the canonical implementations of eccentric-window
identification and muscle mechanics calculations used across the analysis
pipeline. Import from here rather than re-implementing locally.
"""

import numpy as np
import pandas as pd

_MIN_REGION_LEN = 5


def eccentric_mask(grp: pd.DataFrame) -> np.ndarray:
    """Boolean mask for the second negative knee joint-power phase.

    The eccentric window is defined as the second contiguous region where
    knee joint power (angular_velocity × knee_moment) is negative and spans
    at least _MIN_REGION_LEN frames. This corresponds to the hamstring
    lengthening phase during late swing / early stance in sprint acceleration.
    """
    time = np.asarray(grp["time"], dtype=float)
    ang_vel = np.gradient(np.asarray(grp["knee_angle"], dtype=float), time)
    power = ang_vel * np.asarray(grp["knee_angle_moment"], dtype=float)
    neg = (power < 0).astype(int)
    diff = np.diff(neg, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    valid = [(s, e) for s, e in zip(starts, ends) if (e - s) >= _MIN_REGION_LEN]
    mask = np.zeros(len(time), dtype=bool)
    if len(valid) >= 2:
        s, e = valid[1]
        mask[s:e] = True
    return mask


def muscle_moment(grp: pd.DataFrame, muscle: str) -> np.ndarray:
    """Instantaneous muscle moment = fiber_force × knee_moment_arm (N·m)."""
    return (
        np.asarray(grp[f"{muscle}_fiber_force"], dtype=float)
        * np.asarray(grp[f"{muscle}_knee_ma"], dtype=float)
    )


def muscle_power(grp: pd.DataFrame, muscle: str) -> np.ndarray:
    """Eccentric muscle power = force × (−velocity) (W).

    Positive values indicate energy absorption (lengthening contraction).
    Uses inverted velocity convention consistent with negative_work.py.
    """
    force = np.asarray(grp[f"{muscle}_fiber_force"], dtype=float)
    velocity = np.asarray(grp[f"{muscle}_fiber_velocity"], dtype=float)
    return force * (-velocity)


def _calculate_negative_work(grp: pd.DataFrame, muscle: str) -> float:
    """Eccentric work within the eccentric window (N·m). Negative = lengthening."""
    time = np.asarray(grp["time"], dtype=float)
    force = np.asarray(grp[f"{muscle}_fiber_force"], dtype=float)
    velocity = np.asarray(grp[f"{muscle}_fiber_velocity"], dtype=float)
    power = force * (-velocity)
    mask = eccentric_mask(grp)
    power_ecc = np.where(mask, power, 0)
    return float(np.trapezoid(np.minimum(power_ecc, 0), time))


# Canonical per-subject metrics dict. Import and extend as needed.
METRICS: dict = {
    "Eccentric Impulse (N·s)": lambda grp, m: np.trapezoid(
        np.where(eccentric_mask(grp), np.asarray(grp[f"{m}_fiber_force"], dtype=float), 0),
        np.asarray(grp["time"], dtype=float),
    ),
    "Eccentric Knee Moment (N·m·s)": lambda grp, m: np.trapezoid(
        np.where(
            eccentric_mask(grp),
            np.asarray(grp[f"{m}_fiber_force"], dtype=float)
            * np.asarray(grp[f"{m}_knee_ma"], dtype=float),
            0,
        ),
        np.asarray(grp["time"], dtype=float),
    ),
    "Eccentric Work (J)": _calculate_negative_work,
    "Avg EMG": lambda grp, m: np.asarray(grp[f"{m}_adjusted_emg"], dtype=float)[eccentric_mask(grp)].mean(),
    "Peak EMG": lambda grp, m: np.asarray(grp[f"{m}_adjusted_emg"], dtype=float)[eccentric_mask(grp)].max(),
    "Peak norm. fiber length ($l/l_0$)": lambda grp, m: np.asarray(grp[f"{m}_fiber_lengths"], dtype=float)[
        eccentric_mask(grp)
    ].max(),
}

_metric_cache: dict = {}


def compute_metric_per_subject(df: "pd.DataFrame", muscle: str, fn) -> "pd.Series":
    """Compute a metric function per subject, with memoisation keyed on DataFrame identity."""
    key = (id(df), muscle, id(fn))
    if key not in _metric_cache:
        result = {}
        for subj, grp in df.groupby("subject_id"):
            result[subj] = fn(grp, muscle)
        _metric_cache[key] = pd.Series(result)
    return _metric_cache[key]
