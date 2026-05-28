"""
Muscle Moment Metrics Module

Provides metrics for calculating individual muscle contributions to knee
extension moment during the eccentric phase of sprinting.

These metrics can be integrated into thesis_feedback.py or used standalone
for muscle moment analysis.

Usage:
    from analysis.core.moment_metrics import METRICS_MUSCLE_MOMENT
    METRICS.update(METRICS_MUSCLE_MOMENT)
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


def _calculate_muscle_moment(grp: pd.DataFrame, muscle: str) -> np.ndarray:
    """
    Calculate instantaneous muscle moment = Force × Moment Arm.

    Units: N·m
    """
    force = np.asarray(grp[f"{muscle}_fiber_force"], dtype=float)
    moment_arm = np.asarray(grp[f"{muscle}_knee_ma"], dtype=float)
    return force * moment_arm


def _calculate_peak_moment_eccentric(grp: pd.DataFrame, muscle: str) -> float:
    """
    Calculate peak knee extension moment during eccentric window.

    Returns:
        Peak moment in N·m
    """
    moment = _calculate_muscle_moment(grp, muscle)
    mask = _eccentric_mask(grp)
    moment_in_ecc = moment[mask]

    if len(moment_in_ecc) > 0:
        return float(np.nanmax(moment_in_ecc))
    return float(np.nan)


def _calculate_moment_impulse_eccentric(grp: pd.DataFrame, muscle: str) -> float:
    """
    Calculate moment impulse (N·m·s) during eccentric window.

    Impulse = ∫ Moment dt

    Returns:
        Moment impulse in N·m·s
    """
    time = np.asarray(grp["time"], dtype=float)
    moment = _calculate_muscle_moment(grp, muscle)
    mask = _eccentric_mask(grp)
    moment_ecc = np.where(mask, moment, 0)

    impulse = np.trapezoid(moment_ecc, time)
    return float(impulse)


def _calculate_mean_moment_eccentric(grp: pd.DataFrame, muscle: str) -> float:
    """
    Calculate mean moment during eccentric window.

    Returns:
        Mean moment in N·m
    """
    moment = _calculate_muscle_moment(grp, muscle)
    mask = _eccentric_mask(grp)
    moment_in_ecc = moment[mask]

    if len(moment_in_ecc) > 0:
        return float(np.nanmean(moment_in_ecc))
    return float(np.nan)


def _calculate_moment_contribution_pct(grp: pd.DataFrame, muscle: str) -> float:
    """
    Calculate percentage contribution to total hamstring moment impulse.

    Returns:
        Percentage (0-100)
    """
    muscles = ["bflh", "semiten", "semimem"]
    impulses = {
        m: _calculate_moment_impulse_eccentric(grp, m) for m in muscles
    }
    total = sum(impulses.values())

    if total > 0:
        return float(100.0 * impulses[muscle] / total)
    return float(np.nan)


# =============================================================================
# METRICS_MUSCLE_MOMENT Dictionary
# =============================================================================

METRICS_MUSCLE_MOMENT = {
    "Peak Moment (N·m)": lambda grp, m: _calculate_peak_moment_eccentric(grp, m),
    "Mean Moment (N·m)": lambda grp, m: _calculate_mean_moment_eccentric(grp, m),
    "Moment Impulse (N·m·s)": lambda grp, m: _calculate_moment_impulse_eccentric(grp, m),
    "Moment % Contribution": lambda grp, m: _calculate_moment_contribution_pct(grp, m),
}


if __name__ == "__main__":
    print("Muscle Moment Metrics Module")
    print("=" * 60)
    print("\nAvailable metrics:")
    for metric_name in METRICS_MUSCLE_MOMENT.keys():
        print(f"  - {metric_name}")
    print("\nTo use:")
    print("  from analysis.core.moment_metrics import METRICS_MUSCLE_MOMENT")
    print("  METRICS.update(METRICS_MUSCLE_MOMENT)")
