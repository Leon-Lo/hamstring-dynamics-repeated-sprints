"""
Extended METRICS dictionary with negative work calculations for thesis_feedback.py

This file defines additional metrics based on muscle negative work (force × velocity)
that can be incorporated into the main analysis alongside the existing eccentric
impulse and knee moment metrics.

Usage:
    In thesis_feedback.py, import and extend the METRICS dictionary:

    from analysis.core.work_metrics import METRICS_NEGATIVE_WORK
    METRICS.update(METRICS_NEGATIVE_WORK)
"""

import numpy as np
import pandas as pd


_MIN_REGION_LEN = 5


def _eccentric_mask(grp: pd.DataFrame) -> np.ndarray:
    """
    Identify the eccentric window as the second negative knee joint-power phase.
    Knee joint power = angular_velocity × knee_moment
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


def _calculate_muscle_power(grp: pd.DataFrame, muscle: str) -> np.ndarray:
    force = np.asarray(grp[f"{muscle}_fiber_force"], dtype=float)
    velocity = np.asarray(grp[f"{muscle}_fiber_velocity"], dtype=float)
    return force * (-velocity)  # Negative power during lengthening (eccentric phase)


def _calculate_negative_work_eccentric_window(grp: pd.DataFrame, muscle: str) -> float:
    """
    Calculate eccentric work (N·m) within the eccentric window only.

    Eccentric work = ∫ min(power, 0) dt
    This is the work done BY the muscle while lengthening (eccentric phase).
    Negative power = muscle lengthening under tension.
    """
    time = np.asarray(grp["time"], dtype=float)
    power = _calculate_muscle_power(grp, muscle)

    # Restrict to eccentric window
    mask = _eccentric_mask(grp)
    power = np.where(mask, power, 0)

    # Keep only negative power values (lengthening/eccentric work)
    negative_work = np.minimum(power, 0)

    # Integrate using trapezoidal rule
    work = np.trapezoid(negative_work, time)
    return float(work)


def _calculate_negative_work_full_stride(grp: pd.DataFrame, muscle: str) -> float:
    """
    Calculate eccentric work (N·m) across the entire stride, not just eccentric window.

    This captures all muscle lengthening work throughout the movement.
    Negative power = muscle lengthening under tension.
    """
    time = np.asarray(grp["time"], dtype=float)
    power = _calculate_muscle_power(grp, muscle)

    # Keep only negative power values (muscle lengthening/eccentric work)
    negative_work = np.minimum(power, 0)

    # Integrate using trapezoidal rule
    work = np.trapezoid(negative_work, time)
    return float(work)


def _calculate_peak_negative_power(grp: pd.DataFrame, muscle: str) -> float:
    """
    Calculate the peak (minimum) muscle power during lengthening within the eccentric window.

    Units: Watts (W)
    Interpretation: Most negative (maximum absolute) power during eccentric contraction
    Returns magnitude of most negative power (most intense eccentric work).
    """
    power = _calculate_muscle_power(grp, muscle)
    mask = _eccentric_mask(grp)
    power_in_ecc = power[mask]

    if len(power_in_ecc) > 0:
        return float(np.nanmin(power_in_ecc))  # Peak negative value (most intense eccentric work)
    return float(np.nan)


def _calculate_mean_negative_power(grp: pd.DataFrame, muscle: str) -> float:
    """
    Calculate the mean muscle power during lengthening within the eccentric window.

    Units: Watts (W)
    Only negative power values (lengthening/eccentric work) are averaged.
    Returns mean of negative power values (average eccentric work intensity).
    """
    power = _calculate_muscle_power(grp, muscle)
    mask = _eccentric_mask(grp)
    power_in_ecc = power[mask]

    if len(power_in_ecc) > 0:
        negative_power = power_in_ecc[power_in_ecc < 0]
        if len(negative_power) > 0:
            return float(np.nanmean(negative_power))
    return float(np.nan)


# =============================================================================
# METRICS_NEGATIVE_WORK Dictionary
# =============================================================================
# These metrics can be added to the main METRICS dictionary in thesis_feedback.py
# Each metric is a lambda function that takes (grp, muscle) and returns a scalar

METRICS_NEGATIVE_WORK = {
    "Eccentric Work - Eccentric Window (N·m)": lambda grp, m: _calculate_negative_work_eccentric_window(grp, m),
    "Eccentric Work - Full Stride (N·m)": lambda grp, m: _calculate_negative_work_full_stride(grp, m),
    "Peak Eccentric Power (W)": lambda grp, m: _calculate_peak_negative_power(grp, m),
    "Mean Eccentric Power (W)": lambda grp, m: _calculate_mean_negative_power(grp, m),
}


# =============================================================================
# Alternative: Combined METRICS Dictionary
# =============================================================================
# This combines the original metrics with negative work metrics

def get_extended_metrics():
    """
    Return a comprehensive metrics dictionary combining standard metrics
    with negative work metrics for a full biomechanical analysis.
    """
    original_metrics = {
        "Eccentric Impulse (N·s)": lambda grp, m: np.trapezoid(
            np.where(
                _eccentric_mask(grp),
                np.asarray(grp[f"{m}_fiber_force"], dtype=float),
                0,
            ),
            np.asarray(grp["time"], dtype=float),
        ),
        "Eccentric Knee Moment (N·m·s)": lambda grp, m: np.trapezoid(
            np.where(
                _eccentric_mask(grp),
                np.asarray(grp[f"{m}_fiber_force"], dtype=float)
                * np.asarray(grp[f"{m}_knee_ma"], dtype=float),
                0,
            ),
            np.asarray(grp["time"], dtype=float),
        ),
    }

    combined = original_metrics.copy()
    combined.update(METRICS_NEGATIVE_WORK)
    return combined


if __name__ == "__main__":
    print("Negative Work Metrics Module")
    print("=" * 60)
    print("\nAvailable metrics:")
    for metric_name in METRICS_NEGATIVE_WORK.keys():
        print(f"  - {metric_name}")
    print("\nTo use:")
    print("  from analysis.core.work_metrics import METRICS_NEGATIVE_WORK")
    print("  METRICS.update(METRICS_NEGATIVE_WORK)")
