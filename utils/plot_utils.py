import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spm1d as spm
from matplotlib.figure import Figure
from plotly.subplots import make_subplots
from scipy import stats

from utils.analysis_utils import Analysis_utils
from utils.data_utils import Data_utils
from utils.file_utils import File_utils
from utils.subject_utils import Subject_utils


class Plot_utils:
    def __init__(self, style="seaborn-v0_8-whitegrid") -> None:
        """Initialize the Plot_utils instance. All plot functions only have a singular input: master_results.csv"""
        self.file_utils = File_utils()
        self.data_utils = Data_utils()
        self.master_df = pd.read_csv(self.file_utils.get_master_results_file_path())

        # Rename long muscle names to abbreviated OpenSim names for compatibility
        self.master_df = self._rename_muscle_columns(self.master_df)

        self.save_dir = Path("data/plot_results/individual_subject_plots")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use(style)

    @staticmethod
    def _rename_muscle_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename long muscle names to abbreviated OpenSim names.

        This handles CSVs created with old naming convention where columns used
        full names like 'biceps_femoris_long_head_*' instead of abbreviated 'bflh_*'.

        Args:
            df: DataFrame with potentially long muscle names

        Returns:
            DataFrame with renamed columns

        """
        rename_map = {
            "biceps_femoris_long_head": "bflh",
            "semimembranosus": "semimem",
            "semitendinosus": "semiten",
        }

        # Create column rename dictionary
        col_rename = {}
        for col in df.columns:
            for long_name, abbrev in rename_map.items():
                if col.startswith(long_name + "_"):
                    # Replace the long name prefix with abbreviated name
                    new_col = col.replace(long_name, abbrev, 1)
                    col_rename[col] = new_col
                    break

        if col_rename:
            df = df.rename(columns=col_rename)

        return df

    @staticmethod
    def calculate_confidence_intervals(
        df_std: pd.DataFrame,
        df_count: pd.DataFrame,
        columns: list,
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        """
        Calculate confidence intervals using t-distribution.

        Args:
            df_std: DataFrame with standard deviations
            df_count: DataFrame with sample counts
            columns: List of column names to calculate CI for
            confidence: Confidence level (default: 0.95 for 95% CI)

        Returns:
            DataFrame with confidence interval values

        """
        from scipy import stats

        # Calculate alpha for two-tailed test
        alpha = 1 - confidence
        percentile = 1 - alpha / 2

        df_se = df_std.copy()
        df_ci = df_std.copy()

        for col in columns:
            # Calculate standard error
            df_se[col] = df_std[col] / np.sqrt(df_count[col])
            # Get t-critical value using degrees of freedom (n-1)
            t_critical = df_count[col].apply(
                lambda n: stats.t.ppf(percentile, n - 1) if n > 1 else stats.norm.ppf(percentile),  # type: ignore
            )
            # Calculate confidence interval
            df_ci[col] = df_se[col] * t_critical

        return df_ci

    def _filter_subject_trial(self, subject_id: str, trial_name: str):
        """
        Helper method to filter master_df by subject and trial.

        Args:
            subject_id: Subject ID as string
            trial_name: Trial name

        Returns:
            Filtered DataFrame

        Raises:
            Exception: If trial not found in data

        """
        subject_id_int = int(subject_id)

        if trial_name not in self.master_df["trial"].unique():
            raise Exception(
                f"Error: Trial {trial_name} not found in master results data.",
            )

        return self.master_df[
            (self.master_df["subject_id"] == subject_id_int) & (self.master_df["trial"] == trial_name)
        ]

    def _setup_hover_annotations(self, fig, ax, lines_by_axis, num_subplots=3):
        """
        Helper method to add hover annotations to plots.

        Args:
            fig: Matplotlib figure
            ax: Array of axes
            lines_by_axis: List of lists containing (line, label) tuples for each axis
            num_subplots: Number of subplots (default: 3)

        """
        annots = []
        for idx in range(num_subplots):
            annot = ax[idx].annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                arrowprops=dict(arrowstyle="->"),
            )
            annot.set_visible(False)
            annots.append(annot)

        def update_annot(annot, line, label, x, y):
            annot.xy = (x, y)
            annot.set_text(label)
            annot.get_bbox_patch().set_facecolor(line.get_color())
            annot.get_bbox_patch().set_alpha(0.8)

        def hover(event):
            vis_changed = False
            for idx in range(num_subplots):
                if event.inaxes == ax[idx]:
                    for line, label in lines_by_axis[idx]:
                        cont, ind = line.contains(event)
                        if cont:
                            x, y = line.get_data()
                            update_annot(
                                annots[idx],
                                line,
                                label,
                                x[ind["ind"][0]],
                                y[ind["ind"][0]],
                            )
                            annots[idx].set_visible(True)
                            vis_changed = True
                            break
                    else:
                        if annots[idx].get_visible():
                            annots[idx].set_visible(False)
                            vis_changed = True
            if vis_changed:
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

    def _get_tendon_curve_and_fmax_by_muscle(
        self,
        subject_id: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """
        Load tendon force-strain reference curve and per-muscle max isometric force.

        Args:
            subject_id: Subject identifier

        Returns:
            Tuple of (curve_x, curve_y, max_iso_force_by_muscle)

        """
        model_path = self.file_utils.get_ceinms_cal_model_file_path(subject_id)
        root = ET.parse(str(model_path)).getroot()

        mtu_default = root.find("mtuDefault")
        if mtu_default is None:
            raise ValueError("mtuDefault not found in CEINMS model XML.")

        tendon_curve = None
        for curve in mtu_default.findall("curve"):
            name_el = curve.find("name")
            if name_el is not None and name_el.text == "tendonForceStrain":
                tendon_curve = curve
                break

        if tendon_curve is None:
            raise ValueError("tendonForceStrain curve not found in CEINMS model XML.")

        x_points_el = tendon_curve.find("xPoints")
        y_points_el = tendon_curve.find("yPoints")
        if x_points_el is None or y_points_el is None:
            raise ValueError("xPoints/yPoints missing from tendonForceStrain curve.")
        if x_points_el.text is None or y_points_el.text is None:
            raise ValueError("xPoints/yPoints values are empty in tendonForceStrain curve.")

        curve_x = np.array([float(v) for v in x_points_el.text.split()], dtype=float)
        curve_y = np.array([float(v) for v in y_points_el.text.split()], dtype=float)

        mtu_set = root.find("mtuSet")
        if mtu_set is None:
            raise ValueError("mtuSet not found in CEINMS model XML.")

        max_iso_force_by_muscle: dict[str, float] = {}
        for mtu in mtu_set.findall("mtu"):
            name_el = mtu.find("name")
            max_iso_el = mtu.find("maxIsometricForce")
            strength_coeff_el = mtu.find("strengthCoefficient")

            if name_el is None or name_el.text is None:
                continue
            if max_iso_el is None or max_iso_el.text is None:
                continue

            max_iso = float(max_iso_el.text)
            strength_coeff = (
                float(strength_coeff_el.text)
                if strength_coeff_el is not None and strength_coeff_el.text is not None
                else 1.0
            )
            max_iso_force_by_muscle[name_el.text] = max_iso * strength_coeff

        return curve_x, curve_y, max_iso_force_by_muscle

    def plot_norm_tendon_length_and_scaled_reference_force(
        self,
        subject_id: str,
        trial_name: str,
        muscles: list[str] | None = None,
    ):
        """
        Plot tendon signals in two rows for selected muscles:
        1) normalized tendon length
        2) normalized tendon length * interpolated tendon-force-strain reference y * max isometric force

        Args:
            subject_id: Subject identifier
            trial_name: Trial name (e.g., Run_baselineA1)
            muscles: Optional muscle base names (without side suffix), defaults to hamstrings

        Returns:
            fig, axes, output_df

        """
        if muscles is None:
            muscles = ["bflh", "semiten", "semimem"]

        leg = Subject_utils().get_measured_leg(subject_id).lower()
        leg_suffix = f"_{leg}"

        filtered_df = self._filter_subject_trial(subject_id, trial_name).sort_values("index")
        if filtered_df.empty:
            raise ValueError(
                f"No data found for subject {subject_id} and trial {trial_name}.",
            )

        curve_x, curve_y, max_iso_force_by_muscle = self._get_tendon_curve_and_fmax_by_muscle(
            subject_id,
        )

        n_cols = len(muscles)
        fig, axes = plt.subplots(
            2,
            n_cols,
            figsize=(5 * n_cols, 7),
            sharex=True,
            constrained_layout=True,
        )

        if n_cols == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        fig.suptitle(
            f"Subject {subject_id} - {trial_name}: Tendon Length and Scaled Reference",
            fontsize=14,
        )

        output_df = pd.DataFrame(
            {
                "subject_id": filtered_df["subject_id"].values,
                "trial": filtered_df["trial"].values,
                "index": filtered_df["index"].values,
                "time": filtered_df["time"].values,
            },
        )

        for col_idx, muscle_base in enumerate(muscles):
            tendon_col = f"{muscle_base}_tendon_lengths"
            muscle_with_side = f"{muscle_base}{leg_suffix}"

            if tendon_col not in filtered_df.columns:
                raise KeyError(
                    f"Column '{tendon_col}' not found in master results for plotting.",
                )
            if muscle_with_side not in max_iso_force_by_muscle:
                raise KeyError(
                    f"Muscle '{muscle_with_side}' not found in CEINMS model MTU set.",
                )

            norm_tendon_length = filtered_df[tendon_col].to_numpy(dtype=float)
            ref_y = np.interp(
                norm_tendon_length,
                curve_x,
                curve_y,
                left=curve_y[0],
                right=curve_y[-1],
            )
            max_iso_force = max_iso_force_by_muscle[muscle_with_side]

            scaled_force = norm_tendon_length * ref_y * max_iso_force

            axes[0, col_idx].plot(filtered_df["index"], norm_tendon_length, color="tab:blue")
            axes[0, col_idx].set_title(muscle_base)
            axes[0, col_idx].set_ylabel("Norm Tendon Length")
            axes[0, col_idx].grid(alpha=0.3)

            axes[1, col_idx].plot(filtered_df["index"], scaled_force, color="tab:orange")
            axes[1, col_idx].set_ylabel("NormLen * RefY * Fmax")
            axes[1, col_idx].set_xlabel("Sprint Cycle (%)")
            axes[1, col_idx].grid(alpha=0.3)

            output_df[f"{muscle_base}_norm_tendon_length"] = norm_tendon_length
            output_df[f"{muscle_base}_ref_y"] = ref_y
            output_df[f"{muscle_base}_scaled_force"] = scaled_force

        return fig, axes, output_df

    def _plot_all_subjects_interactive(self, trial, plot_configs, title, ylabel):
        """
        Create interactive plotly version of all subjects plots.

        Args:
            trial: Trial name
            plot_configs: List of (axis_index, column, title) tuples
            title: Figure title
            ylabel: Y-axis label

        Returns:
            Plotly figure object or None

        """
        master_df = self.master_df
        subject_ids = master_df["subject_id"].unique()

        if trial not in master_df["trial"].unique():
            return None

        # Create subplots
        subplot_titles = [config[2] for config in plot_configs]
        fig = make_subplots(  # type: ignore
            rows=1,
            cols=len(plot_configs),
            subplot_titles=subplot_titles,
            horizontal_spacing=0.08,
        )

        # Plot individual subjects
        for subject_id in subject_ids:
            subject_id_int = int(subject_id)
            filtered_df = master_df[(master_df["subject_id"] == subject_id_int) & (master_df["trial"] == trial)]

            for idx, column, _ in plot_configs:
                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=filtered_df["index"],
                        y=filtered_df[column],
                        mode="lines",
                        name=f"Subject {subject_id}",
                        line=dict(width=1),
                        opacity=0.5,
                        legendgroup=f"subject_{subject_id}",
                        showlegend=(idx == 0),  # Only show in legend once
                        hovertemplate=f"Subject {subject_id}<br>%{{y:.2f}}<extra></extra>",
                    ),
                    row=1,
                    col=idx + 1,
                )

        # Calculate and plot mean
        trial_df = master_df[master_df["trial"] == trial]
        mean_columns = [column for _, column, _ in plot_configs]
        mean_df = trial_df.groupby("index")[mean_columns].mean().reset_index()

        for idx, column, _ in plot_configs:
            fig.add_trace(
                go.Scatter(  # type: ignore
                    x=mean_df["index"],
                    y=mean_df[column],
                    mode="lines",
                    name="Mean",
                    line=dict(color="black", width=2.5, dash="dash"),
                    legendgroup="mean",
                    showlegend=(idx == 0),
                    hovertemplate="Mean<br>%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=idx + 1,
            )

        # Update layout
        fig.update_xaxes(title_text="Sprint Cycle (%)")
        fig.update_yaxes(title_text=ylabel, row=1, col=1)
        fig.update_layout(
            title_text=f"{title} - Interactive",
            height=500,
            width=1600,
            showlegend=True,
            hovermode="closest",
        )

        return fig

    def _plot_all_subjects_with_mean(
        self,
        trial,
        plot_configs,
        title,
        ylabel,
        return_lines=False,
    ):
        """
        Helper method to plot data for all subjects with mean line.

        Args:
            trial: Trial name
            plot_configs: List of (axis_index, column, title) tuples
            title: Figure title
            ylabel: Y-axis label
            return_lines: Whether to return line objects for hover

        Returns:
            (fig, ax, lines_by_axis) if return_lines else (fig, ax)

        """
        master_df = self.master_df
        subject_ids = master_df["subject_id"].unique()

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(title)

        if trial not in master_df["trial"].unique():
            raise Exception(f"Error: Trial {trial} not found in master results data.")

        lines_by_axis = [[], [], []]

        for subject_id in subject_ids:
            subject_id_int = int(subject_id)
            filtered_df = master_df[(master_df["subject_id"] == subject_id_int) & (master_df["trial"] == trial)]

            for idx, column, _ in plot_configs:
                (line,) = ax[idx].plot(
                    filtered_df["index"],
                    filtered_df[column],
                    label=f"Subject {subject_id}",
                    alpha=0.5,
                )
                lines_by_axis[idx].append((line, f"Subject {subject_id}"))

        # Calculate and plot mean
        trial_df = master_df[master_df["trial"] == trial]
        mean_columns = [column for _, column, _ in plot_configs]
        mean_df = trial_df.groupby("index")[mean_columns].mean().reset_index()

        for idx, column, plot_title in plot_configs:
            ax[idx].plot(
                mean_df["index"],
                mean_df[column],
                label="Mean",
                color="black",
                linestyle="--",
                linewidth=2.5,
            )
            ax[idx].set_title(plot_title)
            ax[idx].grid()

        ax[0].set_ylabel(ylabel)
        fig.text(0.5, 0.04, "Sprint Cycle (%)", ha="center", va="center")

        if return_lines:
            return fig, ax, lines_by_axis
        return fig, ax

    def _plot_single_subject(
        self,
        subject_id: str,
        trial_name: str,
        plot_configs,
        title_prefix,
        ylabel,
    ):
        """
        Helper method to plot data for a single subject across 3 subplots.

        Args:
            subject_id: Subject ID
            trial_name: Trial name
            plot_configs: List of (axis_index, column, label, color, title) tuples
            title_prefix: Prefix for figure title (e.g., "Planar Kinematic Data")
            ylabel: Y-axis label

        Returns:
            fig, ax: Figure and axes objects

        """
        filtered_df = self._filter_subject_trial(subject_id, trial_name)

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"{title_prefix} for Subject {subject_id}, Trial {trial_name}")

        for idx, column, label, color, title in plot_configs:
            ax[idx].plot(
                filtered_df["index"],
                filtered_df[column],
                label=label,
                color=color,
            )
            ax[idx].set_title(title)
            ax[idx].grid()

        ax[0].set_ylabel(ylabel)
        fig.text(0.5, 0.04, "Sprint Cycle (%)", ha="center", va="center")

        return fig, ax

    def plot_trial_data(
        self,
        axis,
        trial_name,
        column,
        label,
        color,
        df_mean,
        df_ci,
        alpha=0.2,
    ):
        """
        Helper function to plot trial data with confidence intervals.

        Args:
            axis: Matplotlib axis to plot on
            trial_name: Name of the trial to plot
            column: Column name from the dataframe to plot
            label: Label for the plot legend
            color: Color for the line and shaded area
            df_mean: DataFrame containing mean values
            df_ci: DataFrame containing confidence interval value
            alpha: Transparency for confidence interval shading (default: 0.2)

        """
        baseline = df_mean[df_mean["trial"] == trial_name]
        ci_data = df_ci[df_ci["trial"] == trial_name]

        axis.plot(
            baseline["index"],
            baseline[column],
            label=label,
            color=color,
            linestyle="-",
        )
        axis.fill_between(
            baseline["index"],
            baseline[column] - ci_data[column],
            baseline[column] + ci_data[column],
            color=color,
            alpha=alpha,
        )

    def plot_single_ik(self, subject_id: str, trial_name: str):
        """
        Plot Inverse Kinematics (IK) marker data over time for hip_flexion, knee_angle, and ankle_angle for single subject and trial.

        Args:
            time: Array-like time data
            marker_data: 2D array-like marker data (rows: time points, columns: markers)
            marker_labels: List of marker labels corresponding to columns in marker_data

        """
        plot_configs = [
            (0, "hip_flexion", "Hip Flexion", "blue", "Hip"),
            (1, "knee_angle", "Knee Flexion", "orange", "Knee"),
            (2, "ankle_angle", "Ankle Flexion", "green", "Ankle"),
        ]

        return self._plot_single_subject(
            subject_id,
            trial_name,
            plot_configs,
            "Planar Kinematic Data",
            "Angle (degrees)",
        )

    def plot_single_id(self, subject_id: str, trial_name: str):
        plot_configs = [
            (0, "hip_flexion_moment", "Hip Flexion Moment", "blue", "Hip"),
            (1, "knee_angle_moment", "Knee Flexion Moment", "orange", "Knee"),
            (2, "ankle_angle_moment", "Ankle Flexion Moment", "green", "Ankle"),
        ]

        return self._plot_single_subject(
            subject_id,
            trial_name,
            plot_configs,
            "Planar Kinetic Data",
            "Moment (Nm/kg)",
        )

    def plot_all_ik(self, trial):
        plot_configs = [
            (0, "hip_flexion", "Hip"),
            (1, "knee_angle", "Knee"),
            (2, "ankle_angle", "Ankle"),
        ]

        title = f"Planar Kinematic Data for Trial {trial} Across All Participants"
        return self._plot_all_subjects_with_mean(
            trial,
            plot_configs,
            title,
            "Angle (degrees)",
        )

    def plot_all_id(self, trial, return_plotly=False):
        """
        Plot inverse dynamics moments for all subjects.

        Args:
            trial: Trial name
            return_plotly: If True and plotly is available, return interactive plotly figure for HTML

        Returns:
            If return_plotly=True: plotly figure object or None
            Otherwise: (matplotlib fig, ax)

        """
        plot_configs = [
            (0, "hip_flexion_moment", "Hip"),
            (1, "knee_angle_moment", "Knee"),
            (2, "ankle_angle_moment", "Ankle"),
        ]

        if return_plotly:
            return self._plot_all_subjects_interactive(
                trial,
                plot_configs,
                f"Planar Kinetic Data for Trial {trial} Across All Participants",
                "Moment (Nm/kg)",
            )

        title = f"Planar Kinetic Data for Trial {trial} Across All Participants"
        fig, ax, lines_by_axis = self._plot_all_subjects_with_mean(  # type: ignore
            trial,
            plot_configs,
            title,
            "Moment (Nm/kg)",
            return_lines=True,
        )

        self._setup_hover_annotations(fig, ax, lines_by_axis)
        return fig, ax

    def plot_all_norm_forces(self, trial, return_plotly=False):
        """
        Plot normalized muscle forces for all subjects.

        Args:
            trial: Trial name
            return_plotly: If True and plotly is available, return interactive plotly figure for HTML

        Returns:
            If return_plotly=True: plotly figure object or None
            Otherwise: (matplotlib fig, ax)

        """
        plot_configs = [
            (0, "bflh_norm_force", "M. Biceps Femoris Long Head"),
            (1, "semiten_norm_force", "M. Semitendinosus"),
            (2, "semimem_norm_force", "M. Semimembranosus"),
        ]

        if return_plotly:
            return self._plot_all_subjects_interactive(
                trial,
                plot_configs,
                f"Muscle Normalized Forces for Trial {trial} Across All Participants",
                "Normalized Force (F/Fmax)",
            )

        title = f"Muscle Normalized Forces for Trial {trial} Across All Participants"
        fig, ax, lines_by_axis = self._plot_all_subjects_with_mean(  # type: ignore
            trial,
            plot_configs,
            title,
            "Normalized Force (F/Fmax)",
            return_lines=True,
        )

        self._setup_hover_annotations(fig, ax, lines_by_axis)
        return fig, ax

    def plot_all_strain_rates(self, trial):
        master_df = self.master_df
        subject_ids = master_df["subject_id"].unique()

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))

        fig.suptitle(f"Muscle Strains for Trial {trial} Across All Participants")

        if trial not in master_df["trial"].unique():
            raise Exception(f"Error: Trial {trial} not found in master results data.")

        # Define plot configurations: (axis_index, column, title)
        plot_configs = [
            (0, "bflh_strain", "M. Biceps Femoris Long Head"),
            (1, "semiten_strain", "M. Semitendinosus"),
            (2, "semimem_strain", "M. Semimembranosus"),
        ]

        for subject_id in subject_ids:
            subject_id_int = int(subject_id)
            filtered_df = master_df[(master_df["subject_id"] == subject_id_int) & (master_df["trial"] == trial)]

            total_time = filtered_df["time"].iloc[-1] - filtered_df["time"].iloc[0]
            fs = len(filtered_df) / total_time

            for idx, column, _ in plot_configs:
                y_filtered = Analysis_utils().butter_lowpass_filter(
                    filtered_df[column],
                    cutoff=10,
                    fs=fs,
                    order=2,
                )
                ax[idx].plot(
                    filtered_df["index"],
                    y_filtered,
                    label=f"Subject {subject_id}",
                )

        for idx, _, title in plot_configs:
            ax[idx].set_title(title)
            ax[idx].grid()

        ax[0].set_ylabel("Strain Rate (Δl/l)")
        fig.text(0.5, 0.04, "Sprint Cycle (%)", ha="center", va="center")

        return fig, ax

    def plot_single_master_plot(
        self,
        subject_id: str,
        trial_name: str,
        ax=None,
        linestyle="-",
    ):
        """
        Plot biomechanical data for a subject and trial.

        Args:
            subject_id: Subject ID
            trial_name: Trial name
            ax: Existing axes grid (7x3). If None, creates a new figure
            linestyle: Line style for this trial (e.g., '-', '--', '-.', ':')

        Returns:
            fig, ax: Figure and axes objects

        """
        master_df = self.master_df
        subject_id_int = int(subject_id)

        if trial_name not in master_df["trial"].unique():
            raise Exception(
                f"Error: Trial {trial_name} not found in master results data.",
            )

        filtered_df = master_df[(master_df["subject_id"] == subject_id_int) & (master_df["trial"] == trial_name)]

        if filtered_df.empty:
            print(
                f"Warning: No data found for subject {subject_id}, trial {trial_name}",
            )
            return None, None

        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(
                7,
                3,
                figsize=(16, 16),
                constrained_layout=True,
                sharex=True,
            )
            is_new_fig = True
        else:
            fig = ax[0, 0].get_figure()
            is_new_fig = False

        # Determine color based on trial name (for before/after plots)
        # Baseline trials = blue (Pre), RunL1 = orange (Post)
        if trial_name in ["Run_baselineA1", "Run_baselineB1"]:
            line_color = "blue"
        elif trial_name == "RunL1":
            line_color = "orange"
        else:
            line_color = "blue"  # Default to blue for other trials

        # Define plot configurations: (row, col, column, label)
        plot_configs = [
            # Row 0: Angles
            (0, 0, "hip_flexion", f"Hip Flexion - {trial_name}"),
            (0, 1, "knee_angle", f"Knee Flexion - {trial_name}"),
            (0, 2, "ankle_angle", f"Ankle Flexion - {trial_name}"),
            # Row 1: Moments
            (1, 0, "hip_flexion_moment", f"Hip Flexion Moment - {trial_name}"),
            (1, 1, "knee_angle_moment", f"Knee Flexion Moment - {trial_name}"),
            (1, 2, "ankle_angle_moment", f"Ankle Flexion Moment - {trial_name}"),
            # Row 2: Adjusted EMG (Activation)
            (
                2,
                0,
                "bflh_adjusted_emg",
                f"Biceps Femoris Activation - {trial_name}",
            ),
            (
                2,
                1,
                "semiten_adjusted_emg",
                f"Semitendinosus Activation - {trial_name}",
            ),
            (
                2,
                2,
                "semimem_adjusted_emg",
                f"Semimembranosus Activation - {trial_name}",
            ),
            # Row 3: Normalized Forces
            (
                3,
                0,
                "bflh_norm_force",
                f"Biceps Femoris Long Head - {trial_name}",
            ),
            (
                3,
                1,
                "semiten_norm_force",
                f"Semitendinosus - {trial_name}",
            ),
            (
                3,
                2,
                "semimem_norm_force",
                f"Semimembranosus - {trial_name}",
            ),
            # Row 4: Active Forces
            (
                4,
                0,
                "bflh_active_force",
                f"Biceps Femoris Active Force - {trial_name}",
            ),
            (
                4,
                1,
                "semiten_active_force",
                f"Semitendinosus Active Force - {trial_name}",
            ),
            (
                4,
                2,
                "semimem_active_force",
                f"Semimembranosus Active Force - {trial_name}",
            ),
            # Row 5: Passive Forces
            (
                5,
                0,
                "bflh_passive_force",
                f"Biceps Femoris Passive Force - {trial_name}",
            ),
            (
                5,
                1,
                "semiten_passive_force",
                f"Semitendinosus Passive Force - {trial_name}",
            ),
            (
                5,
                2,
                "semimem_passive_force",
                f"Semimembranosus Passive Force - {trial_name}",
            ),
            # Row 6: Normalized Fiber Lengths
            (6, 0, "bflh_fiber_lengths", f"Biceps Femoris Normalized Fiber Length - {trial_name}"),
            (6, 1, "semiten_fiber_lengths", f"Semitendinosus Normalized Fiber Length - {trial_name}"),
            (6, 2, "semimem_fiber_lengths", f"Semimembranosus Normalized Fiber Length - {trial_name}"),
        ]

        # Plot all data
        for row, col, column, label in plot_configs:
            ax[row, col].plot(
                filtered_df["index"],
                filtered_df[column],
                label=label,
                color=line_color,
                linestyle=linestyle,
            )

        # Only set axis properties if this is a new figure
        if is_new_fig:
            ax[0, 0].set_title("Hip")
            ax[0, 1].set_title("Knee")
            ax[0, 2].set_title("Ankle")
            ax[2, 0].set_title("M. Biceps Femoris Long Head")
            ax[2, 1].set_title("M. Semitendinosus")
            ax[2, 2].set_title("M. Semimembranosus")

            ax[0, 0].set_ylabel("Angle (degrees)")
            ax[1, 0].set_ylabel("Moment (Nm/kg)")
            ax[2, 0].set_ylabel("Adjusted EMG (0-1)")
            ax[3, 0].set_ylabel("Normalized Force (F/Fmax)")
            ax[4, 0].set_ylabel("Active Force (N)")
            ax[5, 0].set_ylabel("Passive Force (N)")
            ax[6, 0].set_ylabel("Normalized Fiber Length (l/ofl)")

            fig.supxlabel("Sprint Cycle (%)")

        return fig, ax

    def plot_master_plot_before_after(self, subject_id: str):
        # Get all trials for this subject to find the first valid one
        subject_id_int = int(subject_id)
        subject_trials = self.master_df[self.master_df["subject_id"] == subject_id_int]["trial"].unique()

        # Try baseline trial names in order of preference
        if "Run_baselineA1" in subject_trials:
            first_trial = "Run_baselineA1"
        elif "Run_baselineB1" in subject_trials:
            first_trial = "Run_baselineB1"
        else:
            print(f"Error: No baseline trial found for subject {subject_id}")
            return None, None

        fig, ax = self.plot_single_master_plot(subject_id, first_trial, linestyle="-")

        if fig is None:
            print(f"Error: Could not create plot for subject {subject_id}")
            return None

        fig, ax = self.plot_single_master_plot(
            subject_id,
            "RunL1",
            ax=ax,
            linestyle="-",
        )

        if fig is None:
            print(f"Error: Could not create plot for subject {subject_id}")
            return None

        # Create custom legend with Pre and Post
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="blue", linestyle="-"),
            Line2D([0], [0], color="orange", linestyle="-"),
        ]
        fig.legend(custom_lines, ["Pre", "Post"], loc="outside upper right", ncol=2)

        # Add title with subject ID
        fig.suptitle(f"Subject {subject_id}", fontsize=16, y=0.995)

        return fig, ax

    def plot_master_plot_before_after_all(self, p=0.0042):
        """Plot before/after master plots for all Participants on a single figure."""
        # Prepare data
        master_df = self.master_df
        analysis_utils = Analysis_utils()
        spm_sig_idx = analysis_utils.get_spm_sig_index(p=p)

        # Get mean foot contact times for pre and post
        initial_contact_df = analysis_utils.get_initial_foot_contact_time_df(
            time_normalized=True,
        )
        if not initial_contact_df.empty:
            mean_contact_pre = initial_contact_df["initial_contact_time_pre"].mean()
            mean_contact_post = initial_contact_df["initial_contact_time_post"].mean()
        else:
            mean_contact_pre = None
            mean_contact_post = None

        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        pre_df = master_df[master_df["trial"].isin(pre_trials)].copy()
        pre_df["trial"] = "Run_baseline"  # Standardize pre-trial name

        cols_to_process = [
            col for col in master_df.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]
        pre_consolidated = pre_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()
        pre_consolidated["trial"] = "Run_baseline"

        post_df = master_df[master_df["trial"].isin(post_trials)].copy()

        df_filtered = pd.concat([pre_consolidated, post_df], ignore_index=True)

        df_mean = df_filtered.groupby(["trial", "index"])[cols_to_process].mean().reset_index()

        df_std = df_filtered.groupby(["trial", "index"])[cols_to_process].std().reset_index()

        df_count = df_filtered.groupby(["trial", "index"])[cols_to_process].count().reset_index()

        # Calculate 95% confidence interval
        df_ci = self.calculate_confidence_intervals(df_std, df_count, cols_to_process)

        fig, ax = plt.subplots(
            7,
            3,
            figsize=(16, 16),
            constrained_layout=True,
            sharex=True,
        )

        # fig.suptitle(f"n={df_filtered['subject_id'].nunique()} Participants")

        norm_force_limits = (-0.1, 1)

        adjusted_emg_limits = (-0.1, 1)

        for col in range(3):
            ax[2, col].set_ylim(adjusted_emg_limits)
            ax[3, col].set_ylim(norm_force_limits)

        ax[0, 0].set_title("Hip")
        ax[0, 1].set_title("Knee")
        ax[0, 2].set_title("Ankle")
        ax[2, 0].set_title("M. Biceps Femoris Long Head")
        ax[2, 1].set_title("M. Semitendinosus")
        ax[2, 2].set_title("M. Semimembranosus")
        ax[0, 0].set_ylabel("Angle (degrees)")
        ax[1, 0].set_ylabel("Moment (Nm/kg)")
        ax[2, 0].set_ylabel("Adjusted EMG (0-1)")
        ax[3, 0].set_ylabel("Normalized Force (F/Fmax)")
        ax[4, 0].set_ylabel("Active Force (N)")
        ax[5, 0].set_ylabel("Passive Force (N)")
        ax[6, 0].set_ylabel("Normalized Fiber Length (l/ofl)")

        plot_configs = [
            # Row 0: Angles
            (0, 0, "hip_flexion", "Hip Flexion"),
            (0, 1, "knee_angle", "Knee Flexion"),
            (0, 2, "ankle_angle", "Ankle Flexion"),
            # Row 1: Moments
            (1, 0, "hip_flexion_moment", "Hip Flexion Moment"),
            (1, 1, "knee_angle_moment", "Knee Flexion Moment"),
            (1, 2, "ankle_angle_moment", "Ankle Flexion Moment"),
            # Row 2: Adjusted EMG
            (
                2,
                0,
                "bflh_adjusted_emg",
                "Biceps Femoris Activation",
            ),
            (2, 1, "semiten_adjusted_emg", "Semitendinosus Activation"),
            (2, 2, "semimem_adjusted_emg", "Semimembranosus Activation"),
            # Row 3: Normalized Forces
            (3, 0, "bflh_norm_force", "Biceps Femoris Long Head"),
            (3, 1, "semiten_norm_force", "Semitendinosus"),
            (3, 2, "semimem_norm_force", "Semimembranosus"),
            # Row 4: Active Forces
            (4, 0, "bflh_active_force", "Biceps Femoris Active Force"),
            (4, 1, "semiten_active_force", "Semitendinosus Active Force"),
            (4, 2, "semimem_active_force", "Semimembranosus Active Force"),
            # Row 5: Passive Forces
            (5, 0, "bflh_passive_force", "Biceps Femoris Passive Force"),
            (5, 1, "semiten_passive_force", "Semitendinosus Passive Force"),
            (5, 2, "semimem_passive_force", "Semimembranosus Passive Force"),
            # Row 6: Normalized Fiber Lengths
            (6, 0, "bflh_fiber_lengths", "Biceps Femoris Normalized Fiber Length"),
            (6, 1, "semiten_fiber_lengths", "Semitendinosus Normalized Fiber Length"),
            (6, 2, "semimem_fiber_lengths", "Semimembranosus Normalized Fiber Length"),
        ]

        # Only keep rows that exist in the current master_df
        plot_configs = [cfg for cfg in plot_configs if cfg[2] in cols_to_process]

        # Loop through all plot configurations
        for row, col, column, label in plot_configs:
            # Plot Pre (baseline) data
            self.plot_trial_data(
                ax[row, col],
                "Run_baseline",
                column,
                f"{label} - Pre",
                "blue",
                df_mean,
                df_ci,
            )

            # Plot Post (RunL1) data
            self.plot_trial_data(
                ax[row, col],
                "RunL1",
                column,
                f"{label} - Post",
                "orange",
                df_mean,
                df_ci,
            )

            # Add significance shading if available
            sig_idx = spm_sig_idx.get(column)
            if sig_idx is not None:
                ax[row, col].fill_between(
                    range(len(sig_idx)),
                    0,
                    1,
                    where=sig_idx,
                    color="red",
                    alpha=0.1,
                    label="Significant Difference",
                    transform=ax[row, col].get_xaxis_transform(),
                )

            # Add colored dashed vertical lines for mean foot contact (pre=blue, post=orange)
            if mean_contact_pre is not None:
                ax[row, col].axvline(
                    x=mean_contact_pre,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                )
            if mean_contact_post is not None:
                ax[row, col].axvline(
                    x=mean_contact_post,
                    color="orange",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                )

        min_y_norm_force = (
            df_filtered[
                [
                    "bflh_norm_force",
                    "semiten_norm_force",
                    "semimem_norm_force",
                ]
            ]
            .min(skipna=True)
            .min(skipna=True)
        )
        max_y_norm_force = (
            df_filtered[
                [
                    "bflh_norm_force",
                    "semiten_norm_force",
                    "semimem_norm_force",
                ]
            ]
            .max(skipna=True)
            .max(skipna=True)
        )

        # Use fixed axis limits for force/fiber rows to support direct muscle-to-muscle comparison.
        # Keep force curves anchored around 0 with a small visual margin below the axis floor.
        active_force_limits = (-200, 2000)
        passive_force_limits = (-80, 800)
        for col in range(3):
            ax[4, col].set_ylim(active_force_limits)
            ax[5, col].set_ylim(passive_force_limits)
            ax[6, col].set_ylim(0.5, 1.2)

        if (
            pd.isna(min_y_norm_force)
            or pd.isna(max_y_norm_force)
            or np.isinf(min_y_norm_force)
            or np.isinf(max_y_norm_force)
        ):
            print("Warning: NaN or Inf values in consolidated data")
            return None, None

        # Add legend for Pre/Post, significance shading, and mean contact lines
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        custom_lines = [
            Line2D([0], [0], color="blue", linestyle="-"),
            Line2D([0], [0], color="orange", linestyle="-"),
            Line2D([0], [0], color="blue", linestyle="--", linewidth=1.5),
            Line2D([0], [0], color="orange", linestyle="--", linewidth=1.5),
            Patch(facecolor="red", alpha=0.1),
        ]
        fig.legend(
            custom_lines,
            ["Pre", "Post", "Mean Contact Pre", "Mean Contact Post", "p<0.01"],
            loc="outside upper right",
            ncol=3,
        )

        fig.supxlabel("Sprint Cycle (%)")

        return fig, ax

    def plot_master_plot_before_after_all_with_spm(
        self,
        alpha=0.05,
        two_tailed=False,
    ):
        """
        Plot before/after master plots with official SPM1D t-test panels.

        For each variable, the top subplot shows Pre/Post mean +- CI and the bottom
        subplot shows the SPM{t} curve using:
            t = spm1d.stats.ttest_paired(Y0, Y1)
            ti = t.inference(alpha=alpha, two_tailed=two_tailed)
            ti.plot()
        """
        master_df = self.master_df
        analysis_utils = Analysis_utils()

        initial_contact_df = analysis_utils.get_initial_foot_contact_time_df(
            time_normalized=True,
        )
        if not initial_contact_df.empty:
            mean_contact_pre = initial_contact_df["initial_contact_time_pre"].mean()
            mean_contact_post = initial_contact_df["initial_contact_time_post"].mean()
        else:
            mean_contact_pre = None
            mean_contact_post = None

        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        pre_df = master_df[master_df["trial"].isin(pre_trials)].copy()
        pre_df["trial"] = "Run_baseline"

        cols_to_process = [
            col for col in master_df.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]

        pre_consolidated = pre_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()
        pre_consolidated["trial"] = "Run_baseline"
        post_df = master_df[master_df["trial"].isin(post_trials)].copy()

        df_filtered = pd.concat([pre_consolidated, post_df], ignore_index=True)
        df_mean = df_filtered.groupby(["trial", "index"])[cols_to_process].mean().reset_index()
        df_std = df_filtered.groupby(["trial", "index"])[cols_to_process].std().reset_index()
        df_count = df_filtered.groupby(["trial", "index"])[cols_to_process].count().reset_index()
        df_ci = self.calculate_confidence_intervals(df_std, df_count, cols_to_process)

        plot_configs = [
            # Row 0: Angles
            (0, 0, "hip_flexion", "Hip Flexion"),
            (0, 1, "knee_angle", "Knee Flexion"),
            (0, 2, "ankle_angle", "Ankle Flexion"),
            # Row 1: Moments
            (1, 0, "hip_flexion_moment", "Hip Flexion Moment"),
            (1, 1, "knee_angle_moment", "Knee Flexion Moment"),
            (1, 2, "ankle_angle_moment", "Ankle Flexion Moment"),
            # Row 2: Adjusted EMG
            (2, 0, "bflh_adjusted_emg", "Biceps Femoris Activation"),
            (2, 1, "semiten_adjusted_emg", "Semitendinosus Activation"),
            (2, 2, "semimem_adjusted_emg", "Semimembranosus Activation"),
            # Row 3: Normalized Forces
            (3, 0, "bflh_norm_force", "Biceps Femoris Long Head"),
            (3, 1, "semiten_norm_force", "Semitendinosus"),
            (3, 2, "semimem_norm_force", "Semimembranosus"),
            # Row 4: Active Forces
            (4, 0, "bflh_active_force", "Biceps Femoris Active Force"),
            (4, 1, "semiten_active_force", "Semitendinosus Active Force"),
            (4, 2, "semimem_active_force", "Semimembranosus Active Force"),
            # Row 5: Passive Forces
            (5, 0, "bflh_passive_force", "Biceps Femoris Passive Force"),
            (5, 1, "semiten_passive_force", "Semitendinosus Passive Force"),
            (5, 2, "semimem_passive_force", "Semimembranosus Passive Force"),
            # Row 6: Normalized Fiber Lengths
            (6, 0, "bflh_fiber_lengths", "Biceps Femoris Normalized Fiber Length"),
            (6, 1, "semiten_fiber_lengths", "Semitendinosus Normalized Fiber Length"),
            (6, 2, "semimem_fiber_lengths", "Semimembranosus Normalized Fiber Length"),
        ]

        plot_configs = [cfg for cfg in plot_configs if cfg[2] in cols_to_process]

        n_metric_rows = (max(cfg[0] for cfg in plot_configs) + 1) if plot_configs else 0
        fig, ax = plt.subplots(
            n_metric_rows * 2,
            3,
            figsize=(16, max(16, n_metric_rows * 4.2)),
            constrained_layout=True,
            sharex=True,
        )

        if n_metric_rows == 0:
            raise ValueError("No plot configurations available for current master results columns.")

        row_labels = {
            0: "Angle (degrees)",
            1: "Moment (Nm/kg)",
            2: "Adjusted EMG (0-1)",
            3: "Normalized Force (F/Fmax)",
            4: "Active Force (N)",
            5: "Passive Force (N)",
            6: "Normalized Fiber Length (l/ofl)",
        }

        panel_titles = {
            (0, 0): "Hip",
            (0, 1): "Knee",
            (0, 2): "Ankle",
            (2, 0): "M. Biceps Femoris Long Head",
            (2, 1): "M. Semitendinosus",
            (2, 2): "M. Semimembranosus",
        }

        for row, col, column, label in plot_configs:
            metric_ax = ax[row * 2, col]
            spm_ax = ax[row * 2 + 1, col]

            self.plot_trial_data(
                metric_ax,
                "Run_baseline",
                column,
                f"{label} - Pre",
                "blue",
                df_mean,
                df_ci,
            )
            self.plot_trial_data(
                metric_ax,
                "RunL1",
                column,
                f"{label} - Post",
                "orange",
                df_mean,
                df_ci,
            )

            if mean_contact_pre is not None:
                metric_ax.axvline(
                    x=mean_contact_pre,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                )
            if mean_contact_post is not None:
                metric_ax.axvline(
                    x=mean_contact_post,
                    color="orange",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                )

            if (row, col) in panel_titles:
                metric_ax.set_title(panel_titles[(row, col)])

            if col == 0 and row in row_labels:
                metric_ax.set_ylabel(row_labels[row])
                spm_ax.set_ylabel("SPM{t}")

            pre_wide = pre_consolidated.pivot(
                index="subject_id",
                columns="index",
                values=column,
            )
            post_wide = post_df.pivot(
                index="subject_id",
                columns="index",
                values=column,
            )

            common_subjects = pre_wide.index.intersection(post_wide.index)
            common_nodes = pre_wide.columns.intersection(post_wide.columns)

            if len(common_subjects) < 2 or len(common_nodes) < 3:
                spm_ax.text(
                    0.5,
                    0.5,
                    "Insufficient paired data for SPM",
                    transform=spm_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                spm_ax.set_xlim(0, 100)
                continue

            y0 = pre_wide.loc[common_subjects, common_nodes].to_numpy(dtype=float)
            y1 = post_wide.loc[common_subjects, common_nodes].to_numpy(dtype=float)

            valid_rows = np.isfinite(y0).all(axis=1) & np.isfinite(y1).all(axis=1)
            y0 = y0[valid_rows]
            y1 = y1[valid_rows]

            if y0.shape[0] < 2:
                spm_ax.text(
                    0.5,
                    0.5,
                    "Insufficient valid pairs for SPM",
                    transform=spm_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                spm_ax.set_xlim(0, 100)
                continue

            if np.any(np.var(y0, axis=0) == 0) or np.any(np.var(y1, axis=0) == 0):
                spm_ax.text(
                    0.5,
                    0.5,
                    "Zero variance at one or more nodes",
                    transform=spm_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                spm_ax.set_xlim(0, 100)
                continue

            t = spm.stats.ttest_paired(y0, y1)
            ti = t.inference(alpha=alpha, two_tailed=two_tailed)

            plt.sca(spm_ax)
            ti.plot()

            if mean_contact_pre is not None:
                spm_ax.axvline(
                    x=mean_contact_pre,
                    color="blue",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.6,
                )
            if mean_contact_post is not None:
                spm_ax.axvline(
                    x=mean_contact_post,
                    color="orange",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.6,
                )

            spm_ax.set_title("")
            spm_ax.set_xlim(float(min(common_nodes)), float(max(common_nodes)))

        # Keep force/fiber y-limits aligned with the existing master plot style
        active_force_limits = (-200, 2000)
        passive_force_limits = (-80, 800)
        for col in range(3):
            if n_metric_rows > 2:
                ax[2 * 2, col].set_ylim(-0.1, 1)
            if n_metric_rows > 3:
                ax[3 * 2, col].set_ylim(-0.1, 1)
            if n_metric_rows > 4:
                ax[4 * 2, col].set_ylim(active_force_limits)
            if n_metric_rows > 5:
                ax[5 * 2, col].set_ylim(passive_force_limits)
            if n_metric_rows > 6:
                ax[6 * 2, col].set_ylim(0.5, 1.2)

        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="blue", linestyle="-"),
            Line2D([0], [0], color="orange", linestyle="-"),
            Line2D([0], [0], color="blue", linestyle="--", linewidth=1.5),
            Line2D([0], [0], color="orange", linestyle="--", linewidth=1.5),
        ]
        fig.legend(
            custom_lines,
            ["Pre", "Post", "Mean Contact Pre", "Mean Contact Post"],
            loc="outside upper right",
            ncol=2,
        )
        fig.supxlabel("Sprint Cycle (%)")

        return fig, ax

    def plot_master_plot_before_after_all_individual_interactive(self):
        """
        Interactive Plotly version of the master plot showing individual subject lines
        for pre- and post-fatigue conditions.

        Each subject is plotted as a separate line:
        - Blue solid lines = Pre-fatigue (Run_baseline)
        - Orange dashed lines = Post-fatigue (RunL1)

        Hover over any line to see the subject ID and value.

        Returns:
            Plotly Figure object

        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        master_df = self.master_df

        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        cols_to_process = [
            col for col in master_df.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]

        pre_df = master_df[master_df["trial"].isin(pre_trials)].copy()
        pre_consolidated = pre_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()

        post_df = master_df[master_df["trial"].isin(post_trials)].copy()
        post_consolidated = post_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()

        all_subjects = sorted(master_df["subject_id"].unique())

        subplot_titles = [
            "Hip Angle",
            "Knee Angle",
            "Ankle Angle",
            "Hip Moment",
            "Knee Moment",
            "Ankle Moment",
            "M. Biceps Femoris Long Head EMG",
            "M. Semitendinosus EMG",
            "M. Semimembranosus EMG",
            "M. Biceps Femoris Force",
            "M. Semitendinosus Force",
            "M. Semimembranosus Force",
            "M. Biceps Femoris Strain",
            "M. Semitendinosus Strain",
            "M. Semimembranosus Strain",
        ]

        fig = make_subplots(
            rows=5,
            cols=3,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            x_title="Sprint Cycle (%)",
        )

        plot_configs = [
            (1, 1, "hip_flexion", "Angle (degrees)"),
            (1, 2, "knee_angle", "Angle (degrees)"),
            (1, 3, "ankle_angle", "Angle (degrees)"),
            (2, 1, "hip_flexion_moment", "Moment (Nm/kg)"),
            (2, 2, "knee_angle_moment", "Moment (Nm/kg)"),
            (2, 3, "ankle_angle_moment", "Moment (Nm/kg)"),
            (3, 1, "bflh_adjusted_emg", "Adjusted EMG (0-1)"),
            (3, 2, "semiten_adjusted_emg", "Adjusted EMG (0-1)"),
            (3, 3, "semimem_adjusted_emg", "Adjusted EMG (0-1)"),
            (4, 1, "bflh_norm_force", "Normalized Force (F/Fmax)"),
            (4, 2, "semiten_norm_force", "Normalized Force (F/Fmax)"),
            (4, 3, "semimem_norm_force", "Normalized Force (F/Fmax)"),
            (5, 1, "bflh_strain", "Strain (Δfl/ofl)"),
            (5, 2, "semiten_strain", "Strain (Δfl/ofl)"),
            (5, 3, "semimem_strain", "Strain (Δfl/ofl)"),
        ]

        plot_configs = [cfg for cfg in plot_configs if cfg[2] in cols_to_process]

        legend_added_pre = False
        legend_added_post = False

        for subject_id in all_subjects:
            for row, col, column, ylabel in plot_configs:
                # Pre-fatigue (blue, solid)
                subject_pre = pre_consolidated[pre_consolidated["subject_id"] == subject_id]
                if not subject_pre.empty and column in subject_pre.columns:
                    show_in_legend = (not legend_added_pre) and (row == 1 and col == 1)
                    if show_in_legend:
                        legend_added_pre = True
                    fig.add_trace(
                        go.Scatter(
                            x=subject_pre["index"],
                            y=subject_pre[column],
                            mode="lines",
                            name="Pre-fatigue",
                            line=dict(color="blue", width=1.5, dash="solid"),
                            opacity=0.4,
                            showlegend=show_in_legend,
                            legendgroup="pre",
                            hovertemplate=(
                                f"<b>Subject {subject_id}</b><br>"
                                "Phase: Pre-fatigue<br>"
                                "Cycle: %{x:.1f}%<br>"
                                f"{ylabel}: %{{y:.3f}}<br>"
                                "<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )

                # Post-fatigue (orange, dashed)
                subject_post = post_consolidated[post_consolidated["subject_id"] == subject_id]
                if not subject_post.empty and column in subject_post.columns:
                    show_in_legend = (not legend_added_post) and (row == 1 and col == 1)
                    if show_in_legend:
                        legend_added_post = True
                    fig.add_trace(
                        go.Scatter(
                            x=subject_post["index"],
                            y=subject_post[column],
                            mode="lines",
                            name="Post-fatigue",
                            line=dict(color="orange", width=1.5, dash="dash"),
                            opacity=0.4,
                            showlegend=show_in_legend,
                            legendgroup="post",
                            hovertemplate=(
                                f"<b>Subject {subject_id}</b><br>"
                                "Phase: Post-fatigue<br>"
                                "Cycle: %{x:.1f}%<br>"
                                f"{ylabel}: %{{y:.3f}}<br>"
                                "<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )

        n_subjects = len(all_subjects)
        fig.update_layout(
            title_text=f"Individual Subject Master Plot - Pre vs Post Fatigue (n={n_subjects})",
            height=1400,
            showlegend=True,
            hovermode="closest",
            template="plotly_white",
        )

        return fig

    def plot_master_plot_before_after_all_exclude_responders(
        self,
        responder="031",
        spm_sig_idx=None,
    ):
        """
        Plot before/after master plots for all non-responder Participants with responder subject plotted separately.

        This function:
        - Excludes responder subjects from the average and CI calculations
        - Plots the specified responder separately as individual lines on each subplot
        - Shows Pre (baseline) and Post (RunL1) data

        Args:
            responder: Subject ID of the responder to highlight
            spm_sig_idx: Pre-computed SPM significance indices (optional, computed if not provided)

        """
        # Prepare data
        master_df = self.master_df
        analysis_utils = Analysis_utils()

        # Only compute SPM if not provided
        if spm_sig_idx is None:
            spm_sig_idx = analysis_utils.get_spm_sig_index(p=0.0042)

        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        # Get responder IDs for all hamstring muscles
        strain_responders_dict = analysis_utils.get_norm_fiber_length_responders_ids()

        # Combine all responder IDs from all muscles
        responder_ids = set()
        for muscle_responders in strain_responders_dict.values():
            responder_ids.update(muscle_responders)

        pre_df = master_df[master_df["trial"].isin(pre_trials)].copy()
        pre_df["trial"] = "Run_baseline"  # Standardize pre-trial name

        cols_to_process = [
            col for col in master_df.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]
        pre_consolidated = pre_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()
        pre_consolidated["trial"] = "Run_baseline"

        post_df = master_df[master_df["trial"].isin(post_trials)].copy()

        df_all = pd.concat([pre_consolidated, post_df], ignore_index=True)

        # Filter out responders for average/CI calculations
        df_filtered = df_all[~df_all["subject_id"].isin(responder_ids)].copy()

        # Extract subject responder data separately (convert responder string to int)
        responder_id = int(responder)
        subject_responder_df = df_all[df_all["subject_id"] == responder_id].copy()

        # Calculate mean, std, and CI for non-responders only
        df_mean = df_filtered.groupby(["trial", "index"])[cols_to_process].mean().reset_index()

        df_std = df_filtered.groupby(["trial", "index"])[cols_to_process].std().reset_index()

        df_count = df_filtered.groupby(["trial", "index"])[cols_to_process].count().reset_index()

        # Calculate 95% confidence interval
        df_ci = self.calculate_confidence_intervals(df_std, df_count, cols_to_process)

        # Get initial contact times for vertical line markers
        initial_contact_df = analysis_utils.get_initial_foot_contact_time_df(
            time_normalized=True,
        )
        if not initial_contact_df.empty:
            avg_initial_contact_pre = initial_contact_df["initial_contact_time_pre"].mean()
            avg_initial_contact_post = initial_contact_df["initial_contact_time_post"].mean()
        else:
            avg_initial_contact_pre = None
            avg_initial_contact_post = None

        fig, ax = plt.subplots(
            5,
            3,
            figsize=(16, 12),
            constrained_layout=True,
            sharex=True,
        )

        n_non_responders = df_filtered["subject_id"].nunique()
        fig.suptitle(
            f"Non-Responders (n={n_non_responders}) with Subject {responder} Highlighted",
        )

        norm_force_limits = (-0.1, 1)
        # Calculate dynamic strain limits
        strain_cols = ["bflh_strain", "semiten_strain", "semimem_strain"]
        min_y_strain = df_filtered[strain_cols].min(skipna=True).min(skipna=True)
        max_y_strain = df_filtered[strain_cols].max(skipna=True).max(skipna=True)
        # Use consistent strain limits for averaged plots
        strain_limits = (-0.02, 0.03)
        adjusted_emg_limits = (0, 1)

        for col in range(3):
            ax[2, col].set_ylim(adjusted_emg_limits)
            ax[3, col].set_ylim(norm_force_limits)
            ax[4, col].set_ylim(strain_limits)
        ax[0, 0].set_title("Hip")
        ax[0, 1].set_title("Knee")
        ax[0, 2].set_title("Ankle")
        ax[2, 0].set_title("M. Biceps Femoris Long Head")
        ax[2, 1].set_title("M. Semitendinosus")
        ax[2, 2].set_title("M. Semimembranosus")
        ax[0, 0].set_ylabel("Angle (degrees)")
        ax[1, 0].set_ylabel("Moment (Nm/kg)")
        ax[2, 0].set_ylabel("Adjusted EMG (0-1)")
        ax[3, 0].set_ylabel("Normalized Force (F/Fmax)")
        ax[4, 0].set_ylabel("Strain (Δfl/ofl)")

        # Define plot configurations for each subplot
        plot_configs = [
            # Row 0: Angles
            (0, 0, "hip_flexion", "Hip Flexion"),
            (0, 1, "knee_angle", "Knee Flexion"),
            (0, 2, "ankle_angle", "Ankle Flexion"),
            # Row 1: Moments
            (1, 0, "hip_flexion_moment", "Hip Flexion Moment"),
            (1, 1, "knee_angle_moment", "Knee Flexion Moment"),
            (1, 2, "ankle_angle_moment", "Ankle Flexion Moment"),
            # Row 2: Adjusted EMG
            (
                2,
                0,
                "bflh_adjusted_emg",
                "Biceps Femoris Activation",
            ),
            (2, 1, "semiten_adjusted_emg", "Semitendinosus Activation"),
            (2, 2, "semimem_adjusted_emg", "Semimembranosus Activation"),
            # Row 3: Normalized Forces
            (3, 0, "bflh_norm_force", "Biceps Femoris Long Head"),
            (3, 1, "semiten_norm_force", "Semitendinosus"),
            (3, 2, "semimem_norm_force", "Semimembranosus"),
            # Row 4: Strains
            (4, 0, "bflh_strain", "Biceps Femoris Strain"),
            (4, 1, "semiten_strain", "Semitendinosus Strain"),
            (4, 2, "semimem_strain", "Semimembranosus Strain"),
        ]

        # Loop through all plot configurations
        for row, col, column, label in plot_configs:
            # Plot Pre (baseline) data - non-responders average (lower alpha)
            self.plot_trial_data(
                ax[row, col],
                "Run_baseline",
                column,
                f"{label} - Pre",
                "blue",
                df_mean,
                df_ci,
                alpha=0.1,
            )

            # Plot Post (RunL1) data - non-responders average (lower alpha)
            self.plot_trial_data(
                ax[row, col],
                "RunL1",
                column,
                f"{label} - Post",
                "orange",
                df_mean,
                df_ci,
                alpha=0.1,
            )

            # Plot subject responder data separately if available
            subj_responder_pre = subject_responder_df[subject_responder_df["trial"] == "Run_baseline"]
            subj_responder_post = subject_responder_df[subject_responder_df["trial"] == "RunL1"]

            if not subj_responder_pre.empty:
                ax[row, col].plot(
                    subj_responder_pre["index"],
                    subj_responder_pre[column],
                    color="darkblue",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Subject {responder} - Pre",
                )

            if not subj_responder_post.empty:
                ax[row, col].plot(
                    subj_responder_post["index"],
                    subj_responder_post[column],
                    color="darkorange",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Subject {responder} - Post",
                )

            # Add significance shading if available
            sig_idx = spm_sig_idx.get(column)
            if sig_idx is not None:
                ax[row, col].fill_between(
                    range(len(sig_idx)),
                    ax[row, col].get_ylim()[0],
                    ax[row, col].get_ylim()[1],
                    where=sig_idx,
                    color="red",
                    alpha=0.1,
                    label="Significant Difference",
                )

            # Add vertical lines for average initial contact times
            if avg_initial_contact_pre is not None:
                ax[row, col].axvline(
                    x=avg_initial_contact_pre,
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.6,
                    label="Avg Contact Pre" if row == 0 and col == 0 else "",
                )
            if avg_initial_contact_post is not None:
                ax[row, col].axvline(
                    x=avg_initial_contact_post,
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.6,
                    label="Avg Contact Post" if row == 0 and col == 0 else "",
                )

        # Calculate dynamic y-limits for normalized force and strain
        min_y_norm_force = (
            df_filtered[
                [
                    "bflh_norm_force",
                    "semiten_norm_force",
                    "semimem_norm_force",
                ]
            ]
            .min(skipna=True)
            .min(skipna=True)
        )
        max_y_norm_force = (
            df_filtered[
                [
                    "bflh_norm_force",
                    "semiten_norm_force",
                    "semimem_norm_force",
                ]
            ]
            .max(skipna=True)
            .max(skipna=True)
        )
        min_y_strain = (
            df_filtered[
                [
                    "bflh_strain",
                    "semiten_strain",
                    "semimem_strain",
                ]
            ]
            .min(skipna=True)
            .min(skipna=True)
        )
        max_y_strain = (
            df_filtered[
                [
                    "bflh_strain",
                    "semiten_strain",
                    "semimem_strain",
                ]
            ]
            .max(skipna=True)
            .max(skipna=True)
        )

        if (
            pd.isna(min_y_norm_force)
            or pd.isna(max_y_norm_force)
            or pd.isna(min_y_strain)
            or pd.isna(max_y_strain)
            or np.isinf(min_y_norm_force)
            or np.isinf(max_y_norm_force)
            or np.isinf(min_y_strain)
            or np.isinf(max_y_strain)
        ):
            print("Warning: NaN or Inf values in consolidated data")
            return None, None

        # Add legend for Pre, Post, and Subject responder
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="blue", linestyle="-"),
            Line2D([0], [0], color="orange", linestyle="-"),
            Line2D([0], [0], color="darkblue", linestyle="--", linewidth=2),
            Line2D([0], [0], color="darkorange", linestyle="--", linewidth=2),
            Line2D([0], [0], color="black", linestyle=":", linewidth=1.5),
        ]
        fig.legend(
            custom_lines,
            [
                "Non-Resp Pre",
                "Non-Resp Post",
                f"Subject {responder} Pre",
                f"Subject {responder} Post",
                "Avg Initial Contact",
            ],
            loc="upper right",
            ncol=3,
        )

        fig.supxlabel("Sprint Cycle (%)")

        return fig, ax

    def plot_master_plot_with_after_all_group_split(self):
        """
        Plot pre-exercise master plots split into 4 groups:
        - Non-responders, non-CAM
        - Responders, non-CAM
        - Non-responders, CAM
        - Responders, CAM
        """
        # Prepare data
        master_df = self.master_df
        analysis_utils = Analysis_utils()

        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        # Consolidate pre-trial data
        pre_df = master_df[master_df["trial"].isin(pre_trials)].copy()
        pre_df["trial"] = "Run_baseline"

        cols_to_process = [
            col for col in master_df.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]
        pre_consolidated = pre_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()
        pre_consolidated["trial"] = "Run_baseline"

        post_df = master_df[master_df["trial"].isin(post_trials)].copy()

        df_filtered = pd.concat([pre_consolidated, post_df], ignore_index=True)

        # Get subject groups
        # Use biceps femoris long head as the main muscle for responder classification
        muscles = ["bflh"]

        strain_responders_dict = analysis_utils.get_norm_fiber_length_responders_ids(
            muscles=muscles,
        )
        strain_non_responders_dict = analysis_utils.get_norm_fiber_length_non_responders_ids(
            muscles=muscles,
        )
        cam_ids = analysis_utils.get_cam_positive_ids(filter=True)
        non_cam_ids = analysis_utils.get_non_cam_ids()

        # Get responder IDs (combine all muscles)
        responder_ids = set()
        for muscle_responders in strain_responders_dict.values():
            responder_ids.update(muscle_responders)

        non_responder_ids = set()
        for muscle_non_responders in strain_non_responders_dict.values():
            non_responder_ids.update(muscle_non_responders)

        # Create 4 groups
        group_1_ids = list(
            non_responder_ids & set(non_cam_ids),
        )  # Non-responders, non-CAM
        group_2_ids = list(responder_ids & set(non_cam_ids))  # Responders, non-CAM
        group_3_ids = list(non_responder_ids & set(cam_ids))  # Non-responders, CAM
        group_4_ids = list(responder_ids & set(cam_ids))  # Responders, CAM

        print(f"Group 1 (Non-responders, non-CAM): n={len(group_1_ids)}")
        print(f"Group 2 (Responders, non-CAM): n={len(group_2_ids)}")
        print(f"Group 3 (Non-responders, CAM): n={len(group_3_ids)}")
        print(f"Group 4 (Responders, CAM): n={len(group_4_ids)}")

        # Define groups configuration: (ids, color, label_suffix)
        groups_config = [
            (group_1_ids, "blue", "Non-responders, non-CAM"),
            (group_2_ids, "green", "Responders, non-CAM"),
            (group_3_ids, "red", "Non-responders, CAM"),
            (group_4_ids, "orange", "Responders, CAM"),
        ]

        # Calculate means and CIs for each group
        group_data = []
        for group_ids, color, label in groups_config:
            if len(group_ids) == 0:
                group_data.append(None)
                continue

            group_df = df_filtered[df_filtered["subject_id"].isin(group_ids)]

            group_mean = group_df.groupby(["trial", "index"])[cols_to_process].mean().reset_index()

            group_std = group_df.groupby(["trial", "index"])[cols_to_process].std().reset_index()

            group_data.append(
                {
                    "mean": group_mean,
                    "sd": group_std,
                    "color": color,
                    "label": label,
                    "n": len(group_ids),
                },
            )

        # Create figure
        fig, ax = plt.subplots(
            5,
            3,
            figsize=(16, 12),
            constrained_layout=True,
            sharex=True,
        )

        fig.suptitle("Group Comparison: Responders vs Non-responders, CAM vs non-CAM")

        # Set titles and labels
        ax[0, 0].set_title("Hip")
        ax[0, 1].set_title("Knee")
        ax[0, 2].set_title("Ankle")
        ax[2, 0].set_title("M. Biceps Femoris Long Head")
        ax[2, 1].set_title("M. Semitendinosus")
        ax[2, 2].set_title("M. Semimembranosus")
        ax[0, 0].set_ylabel("Angle (degrees)")
        ax[1, 0].set_ylabel("Moment (Nm/kg)")
        ax[2, 0].set_ylabel("Adjusted EMG (0-1)")
        ax[3, 0].set_ylabel("Normalized Force (F/Fmax)")
        ax[4, 0].set_ylabel("Strain (Δfl/ofl)")

        # Define plot configurations for each subplot
        all_plot_configs = [
            # Row 0: Angles
            (0, 0, "hip_flexion", "Hip Flexion"),
            (0, 1, "knee_angle", "Knee Flexion"),
            (0, 2, "ankle_angle", "Ankle Flexion"),
            # Row 1: Moments
            (1, 0, "hip_flexion_moment", "Hip Flexion Moment"),
            (1, 1, "knee_angle_moment", "Knee Flexion Moment"),
            (1, 2, "ankle_angle_moment", "Ankle Flexion Moment"),
            # Row 2: Adjusted EMG
            (
                2,
                0,
                "bflh_adjusted_emg",
                "Biceps Femoris Activation",
            ),
            (2, 1, "semiten_adjusted_emg", "Semitendinosus Activation"),
            (2, 2, "semimem_adjusted_emg", "Semimembranosus Activation"),
            # Row 3: Normalized Forces
            (3, 0, "bflh_norm_force", "Biceps Femoris Long Head"),
            (3, 1, "semiten_norm_force", "Semitendinosus"),
            (3, 2, "semimem_norm_force", "Semimembranosus"),
            # Row 4: Strains
            (4, 0, "bflh_strain", "Biceps Femoris Strain"),
            (4, 1, "semiten_strain", "Semitendinosus Strain"),
            (4, 2, "semimem_strain", "Semimembranosus Strain"),
        ]

        # Filter plot configs to only include columns that exist in cols_to_process
        plot_configs = [config for config in all_plot_configs if config[2] in cols_to_process]

        # Plot each group
        for group_info in group_data:
            if group_info is None:
                continue

            df_mean = group_info["mean"]
            df_sd = group_info["sd"]
            color = group_info["color"]

            # Plot Pre only for this group
            for row, col, column, _ in plot_configs:
                # Plot Pre (baseline) data with solid line
                pre_data = df_mean[df_mean["trial"] == "Run_baseline"]
                pre_sd_data = df_sd[df_sd["trial"] == "Run_baseline"]

                if not pre_data.empty and column in pre_data.columns:
                    ax[row, col].plot(
                        pre_data["index"],
                        pre_data[column],
                        color=color,
                        linestyle="-",
                        alpha=1.0,
                        linewidth=2,
                    )
                    if column in pre_sd_data.columns:
                        ax[row, col].fill_between(
                            pre_sd_data["index"],
                            pre_data[column] - pre_sd_data[column],
                            pre_data[column] + pre_sd_data[column],
                            color=color,
                            alpha=0.15,
                        )

        # Set axis limits
        norm_force_limits = (-0.3, 1.6)
        # Use consistent strain limits for averaged plots
        strain_limits = (-0.02, 0.03)
        emg_limits = (-0.1, 1)

        for col in range(3):
            ax[2, col].set_ylim(emg_limits)
            ax[3, col].set_ylim(norm_force_limits)
            ax[4, col].set_ylim(strain_limits)

        # Create legend for pre data only
        from matplotlib.lines import Line2D

        legend_lines = []
        legend_labels = []

        for group_info in group_data:
            if group_info is None:
                continue

            color = group_info["color"]
            label = group_info["label"]
            n = group_info["n"]

            # Pre line (solid)
            legend_lines.append(
                Line2D([0], [0], color=color, linestyle="-", alpha=1.0, linewidth=2),
            )
            legend_labels.append(f"{label} (n={n})")

        fig.legend(legend_lines, legend_labels, loc="upper right", ncol=2, fontsize=9)

        fig.supxlabel("Sprint Cycle (%)")

        return fig, ax

    def plot_master_plot_with_after_all_group_split_individual(
        self,
        trial_type="pre",
        cam_filter=None,
        responder_filter=None,
    ):
        """
        Plot master plots for individual subjects split into 4 groups:
        - Non-responders, non-CAM
        - Responders, non-CAM
        - Non-responders, CAM
        - Responders, CAM

        Each subject is plotted individually with interactive hover showing subject ID.

        Args:
            trial_type: Type of trial to plot. Options: "pre", "post", or "both" (default: "pre")
            cam_filter: Filter by CAM status. Options: None (all), "cam", "non-cam" (default: None)
            responder_filter: Filter by responder status. Options: None (all), "responder", "non-responder" (default: None)

        """
        # Prepare data
        master_df = self.master_df
        analysis_utils = Analysis_utils()

        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        # Consolidate data based on trial_type
        # Define cols_to_process once
        cols_to_process = [
            col for col in master_df.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]

        # Initialize consolidated dataframes as None
        pre_consolidated = None
        post_consolidated = None

        if trial_type in ["pre", "both"]:
            # Consolidate pre-trial data
            pre_df = master_df[master_df["trial"].isin(pre_trials)].copy()

            pre_consolidated = pre_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()

        if trial_type in ["post", "both"]:
            # Get post-trial data
            post_df = master_df[master_df["trial"].isin(post_trials)].copy()

            post_consolidated = post_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()

        # Get subject groups
        # Use biceps femoris long head as the main muscle for responder classification
        muscles = ["bflh"]

        strain_responders_dict = analysis_utils.get_norm_fiber_length_responders_ids(
            muscles=muscles,
        )
        strain_non_responders_dict = analysis_utils.get_norm_fiber_length_non_responders_ids(
            muscles=muscles,
        )
        cam_ids = analysis_utils.get_cam_positive_ids(filter=True)
        non_cam_ids = analysis_utils.get_non_cam_ids()

        # Get responder IDs (combine all muscles)
        responder_ids = set()
        for muscle_responders in strain_responders_dict.values():
            responder_ids.update(muscle_responders)

        non_responder_ids = set()
        for muscle_non_responders in strain_non_responders_dict.values():
            non_responder_ids.update(muscle_non_responders)

        # Create 4 groups
        group_1_ids = list(
            non_responder_ids & set(non_cam_ids),
        )  # Non-responders, non-CAM
        group_2_ids = list(responder_ids & set(non_cam_ids))  # Responders, non-CAM
        group_3_ids = list(non_responder_ids & set(cam_ids))  # Non-responders, CAM
        group_4_ids = list(responder_ids & set(cam_ids))  # Responders, CAM

        # Apply filters
        filtered_groups = []

        # Group 1: Non-responders, non-CAM
        if (responder_filter is None or responder_filter == "non-responder") and (
            cam_filter is None or cam_filter == "non-cam"
        ):
            filtered_groups.append((group_1_ids, "blue", "Non-responders, non-CAM"))

        # Group 2: Responders, non-CAM
        if (responder_filter is None or responder_filter == "responder") and (
            cam_filter is None or cam_filter == "non-cam"
        ):
            filtered_groups.append((group_2_ids, "green", "Responders, non-CAM"))

        # Group 3: Non-responders, CAM
        if (responder_filter is None or responder_filter == "non-responder") and (
            cam_filter is None or cam_filter == "cam"
        ):
            filtered_groups.append((group_3_ids, "red", "Non-responders, CAM"))

        # Group 4: Responders, CAM
        if (responder_filter is None or responder_filter == "responder") and (
            cam_filter is None or cam_filter == "cam"
        ):
            filtered_groups.append((group_4_ids, "orange", "Responders, CAM"))

        print(f"Group 1 (Non-responders, non-CAM): n={len(group_1_ids)}")
        print(f"Group 2 (Responders, non-CAM): n={len(group_2_ids)}")
        print(f"Group 3 (Non-responders, CAM): n={len(group_3_ids)}")
        print(f"Group 4 (Responders, CAM): n={len(group_4_ids)}")
        print(f"Filtered groups: {len(filtered_groups)}")

        # Define groups configuration: (ids, color, label_suffix)
        groups_config = filtered_groups

        # Create figure
        fig, ax = plt.subplots(
            5,
            3,
            figsize=(16, 12),
            constrained_layout=True,
            sharex=True,
        )

        # Update title based on trial_type and filters
        if trial_type == "pre":
            title_suffix = "Pre-fatigue"
        elif trial_type == "post":
            title_suffix = "Post-fatigue"
        else:
            title_suffix = "Pre vs Post-fatigue"

        # Build title with filters
        filter_parts = []
        if responder_filter:
            filter_parts.append(responder_filter.replace("-", " ").title() + "s")
        if cam_filter:
            filter_parts.append(
                cam_filter.upper() if cam_filter == "cam" else "non-CAM",
            )

        if filter_parts:
            filter_text = " - " + " & ".join(filter_parts)
        else:
            filter_text = ": Responders vs Non-responders, CAM vs non-CAM"

        fig.suptitle(f"Individual Subject Comparison ({title_suffix}){filter_text}")

        # Set axis limits
        norm_force_limits = (-0.3, 2)
        # Calculate dynamic strain limits
        combined_data = []
        if pre_consolidated is not None:
            combined_data.append(pre_consolidated)
        if post_consolidated is not None:
            combined_data.append(post_consolidated)
        if combined_data:
            df_combined = pd.concat(combined_data, ignore_index=True)
            strain_cols = ["bflh_strain", "semiten_strain", "semimem_strain"]
            min_y_strain = df_combined[strain_cols].min(skipna=True).min(skipna=True)
            max_y_strain = df_combined[strain_cols].max(skipna=True).max(skipna=True)
            if not (pd.isna(min_y_strain) or pd.isna(max_y_strain) or np.isinf(min_y_strain) or np.isinf(max_y_strain)):
                # Use consistent strain limits for averaged plots
                strain_limits = (-0.02, 0.03)
            else:
                strain_limits = (-0.02, 0.03)  # Use consistent limits
        else:
            strain_limits = (-0.02, 0.03)
        adjusted_emg_limits = (-0.1, 1.2)

        for col in range(3):
            ax[2, col].set_ylim(adjusted_emg_limits)
            ax[3, col].set_ylim(norm_force_limits)
            ax[4, col].set_ylim(strain_limits)

        # Set titles and labels
        ax[0, 0].set_title("Hip")
        ax[0, 1].set_title("Knee")
        ax[0, 2].set_title("Ankle")
        ax[2, 0].set_title("M. Biceps Femoris Long Head")
        ax[2, 1].set_title("M. Semitendinosus")
        ax[2, 2].set_title("M. Semimembranosus")
        ax[0, 0].set_ylabel("Angle (degrees)")
        ax[1, 0].set_ylabel("Moment (Nm/kg)")
        ax[2, 0].set_ylabel("Adjusted EMG (0-1)")
        ax[3, 0].set_ylabel("Normalized Force (F/Fmax)")
        ax[4, 0].set_ylabel("Strain (Δfl/ofl)")

        # Define plot configurations for each subplot
        all_plot_configs = [
            # Row 0: Angles
            (0, 0, "hip_flexion", "Hip Flexion"),
            (0, 1, "knee_angle", "Knee Flexion"),
            (0, 2, "ankle_angle", "Ankle Flexion"),
            # Row 1: Moments
            (1, 0, "hip_flexion_moment", "Hip Flexion Moment"),
            (1, 1, "knee_angle_moment", "Knee Flexion Moment"),
            (1, 2, "ankle_angle_moment", "Ankle Flexion Moment"),
            # Row 2: Adjusted EMG
            (
                2,
                0,
                "bflh_adjusted_emg",
                "Biceps Femoris Activation",
            ),
            (2, 1, "semiten_adjusted_emg", "Semitendinosus Activation"),
            (2, 2, "semimem_adjusted_emg", "Semimembranosus Activation"),
            # Row 3: Normalized Forces
            (3, 0, "bflh_norm_force", "Biceps Femoris Long Head"),
            (3, 1, "semiten_norm_force", "Semitendinosus"),
            (3, 2, "semimem_norm_force", "Semimembranosus"),
            # Row 4: Strains
            (4, 0, "bflh_strain", "Biceps Femoris Strain"),
            (4, 1, "semiten_strain", "Semitendinosus Strain"),
            (4, 2, "semimem_strain", "Semimembranosus Strain"),
        ]

        # Filter plot configs to only include columns that exist in cols_to_process
        plot_configs = [config for config in all_plot_configs if config[2] in cols_to_process]

        # Store lines for hover functionality - organized by axis position
        lines_by_axis = {}
        for row in range(5):
            for col in range(3):
                lines_by_axis[(row, col)] = []

        # Plot each group and individual subjects
        for group_ids, color, label in groups_config:
            if len(group_ids) == 0:
                continue

            # Plot each subject in the group
            for subject_id in group_ids:
                # Plot pre data if requested
                if trial_type in ["pre", "both"] and pre_consolidated is not None:
                    subject_df_pre = pre_consolidated[pre_consolidated["subject_id"] == subject_id]

                    if not subject_df_pre.empty:
                        # Plot this subject on all subplots (pre data with solid line)
                        for row, col, column, _ in plot_configs:
                            if column not in subject_df_pre.columns:
                                continue

                            (line,) = ax[row, col].plot(
                                subject_df_pre["index"],
                                subject_df_pre[column],
                                color=color,
                                linestyle="-",
                                alpha=0.6,
                                linewidth=1.5,
                            )
                            # Store line with subject ID and trial type for hover functionality
                            lines_by_axis[(row, col)].append((line, subject_id, "Pre"))

                # Plot post data if requested
                if trial_type in ["post", "both"] and post_consolidated is not None:
                    subject_df_post = post_consolidated[post_consolidated["subject_id"] == subject_id]

                    if not subject_df_post.empty:
                        # Plot this subject on all subplots (post data with dashed line)
                        for row, col, column, _ in plot_configs:
                            if column not in subject_df_post.columns:
                                continue

                            (line,) = ax[row, col].plot(
                                subject_df_post["index"],
                                subject_df_post[column],
                                color=color,
                                linestyle="--" if trial_type == "both" else "-",
                                alpha=0.6,
                                linewidth=1.5,
                            )
                            # Store line with subject ID and trial type for hover functionality
                            lines_by_axis[(row, col)].append((line, subject_id, "Post"))

        # Create legend for groups
        from matplotlib.lines import Line2D

        legend_lines = []
        legend_labels = []

        for group_ids, color, label in groups_config:
            if len(group_ids) == 0:
                continue

            if trial_type == "both":
                # Add both solid (pre) and dashed (post) lines
                legend_lines.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="-",
                        alpha=0.6,
                        linewidth=1.5,
                    ),
                )
                legend_labels.append(f"{label} Pre (n={len(group_ids)})")
                legend_lines.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="--",
                        alpha=0.6,
                        linewidth=1.5,
                    ),
                )
                legend_labels.append(f"{label} Post (n={len(group_ids)})")
            else:
                legend_lines.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="-",
                        alpha=0.6,
                        linewidth=1.5,
                    ),
                )
                legend_labels.append(f"{label} (n={len(group_ids)})")

        fig.legend(legend_lines, legend_labels, loc="upper right", ncol=2, fontsize=9)

        # Add interactive hover annotations
        annotations = {}
        for (row, col), lines_data in lines_by_axis.items():
            if not lines_data:
                continue

            annot = ax[row, col].annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="yellow", alpha=0.9),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                zorder=1000,
            )
            annot.set_visible(False)
            annotations[(row, col)] = annot

        def on_hover(event):
            """Handle mouse hover events to show subject ID."""
            if event.inaxes is None:
                return

            # Find which subplot the mouse is in
            subplot_pos = None
            for row in range(5):
                for col in range(3):
                    if event.inaxes == ax[row, col]:
                        subplot_pos = (row, col)
                        break
                if subplot_pos:
                    break

            if subplot_pos is None:
                return

            lines_data = lines_by_axis.get(subplot_pos, [])
            annot = annotations.get(subplot_pos)

            if not lines_data or annot is None:
                return

            # Check if mouse is near any line
            for line, subject_id, trial_label in lines_data:
                if line.contains(event)[0]:
                    # Get the closest point on the line
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()

                    # Find closest x-index
                    if len(x_data) > 0:
                        distances = np.abs(x_data - event.xdata)
                        idx = np.argmin(distances)

                        annot.xy = (x_data[idx], y_data[idx])
                        annot.set_text(f"Subject {subject_id} ({trial_label})")
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return

            # If no line is hovered, hide annotation
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

        # Connect hover event
        fig.canvas.mpl_connect("motion_notify_event", on_hover)

        fig.supxlabel("Sprint Cycle (%)")

        return fig, ax

    def plot_responder_comparison_individual(self, trial_type="pre", interactive=False):
        """
        Plot individual subjects split into 2 groups by responder status only:
        - Non-responders (blue)
        - Responders (green)

        Each subject is plotted individually with interactive hover showing subject ID.

        Args:
            trial_type: Type of trial to plot. Options: "pre", "post", or "both" (default: "pre")
            interactive: If True, return interactive Plotly figure. Otherwise, return matplotlib figure.

        Returns:
            If interactive=True: Plotly figure object
            Otherwise: (matplotlib fig, ax) tuple

        """
        # If interactive requested, create Plotly version
        if interactive:
            return self._plot_responder_comparison_individual_interactive(trial_type)

        # Otherwise create matplotlib version
        # Prepare data
        master_df = self.master_df
        analysis_utils = Analysis_utils()

        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        # Define cols_to_process once
        cols_to_process = [
            col for col in master_df.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]

        # Initialize consolidated dataframes as None
        pre_consolidated = None
        post_consolidated = None

        if trial_type in ["pre", "both"]:
            # Consolidate pre-trial data
            pre_df = master_df[master_df["trial"].isin(pre_trials)].copy()

            pre_consolidated = pre_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()

        if trial_type in ["post", "both"]:
            # Get post-trial data
            post_df = master_df[master_df["trial"].isin(post_trials)].copy()

            post_consolidated = post_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()

        # Get subject groups - Get responders for all three hamstring muscles
        all_muscles = ["bflh", "semiten", "semimem"]

        # Get responders for each muscle
        strain_responders_dict = analysis_utils.get_norm_fiber_length_responders_ids(
            muscles=all_muscles,
        )
        strain_non_responders_dict = analysis_utils.get_norm_fiber_length_non_responders_ids(
            muscles=all_muscles,
        )

        # Get initial contact times for vertical line markers
        initial_contact_df = analysis_utils.get_initial_foot_contact_time_df(
            time_normalized=True,
        )
        if not initial_contact_df.empty:
            avg_initial_contact_pre = initial_contact_df["initial_contact_time_pre"].mean()
            avg_initial_contact_post = initial_contact_df["initial_contact_time_post"].mean()
        else:
            avg_initial_contact_pre = None
            avg_initial_contact_post = None

        print("Responders by muscle:")
        for muscle, ids in strain_responders_dict.items():
            print(f"  {muscle}: n={len(ids)}, IDs={ids}")

        print("\nNon-responders by muscle:")
        for muscle, ids in strain_non_responders_dict.items():
            print(f"  {muscle}: n={len(ids)}, IDs={ids}")

        # Map columns to their corresponding muscle for classification
        # Columns in rows 2-4 (EMG, force, strain) will use muscle-specific classification
        # Columns in rows 0-1 (angles, moments) will use bflh classification
        column_to_muscle = {
            # Row 2: EMG
            (2, 0): "bflh",
            (2, 1): "semiten",
            (2, 2): "semimem",
            # Row 3: Forces
            (3, 0): "bflh",
            (3, 1): "semiten",
            (3, 2): "semimem",
            # Row 4: Strains
            (4, 0): "bflh",
            (4, 1): "semiten",
            (4, 2): "semimem",
        }

        # Create figure
        fig, ax = plt.subplots(
            5,
            3,
            figsize=(16, 12),
            constrained_layout=True,
            sharex=True,
        )

        # Update title based on trial_type
        if trial_type == "pre":
            title_suffix = "Pre-fatigue"
        elif trial_type == "post":
            title_suffix = "Post-fatigue"
        else:
            title_suffix = "Pre vs Post-fatigue"

        fig.suptitle(
            f"Individual Subject Comparison: Responders vs Non-responders ({title_suffix})",
        )

        # Calculate adaptive axis limits based on actual data
        # Normalized force columns
        norm_force_cols = [
            "bflh_norm_force",
            "semiten_norm_force",
            "semimem_norm_force",
        ]
        norm_force_cols = [col for col in norm_force_cols if col in cols_to_process]

        # Strain columns
        strain_cols = ["bflh_strain", "semiten_strain", "semimem_strain"]
        strain_cols = [col for col in strain_cols if col in cols_to_process]

        # Calculate min/max for normalized forces
        norm_force_values = []
        if pre_consolidated is not None and norm_force_cols:
            for col in norm_force_cols:
                if col in pre_consolidated.columns:
                    norm_force_values.extend(pre_consolidated[col].dropna().values)
        if post_consolidated is not None and norm_force_cols:
            for col in norm_force_cols:
                if col in post_consolidated.columns:
                    norm_force_values.extend(post_consolidated[col].dropna().values)

        if norm_force_values:
            norm_force_min = min(norm_force_values)
            norm_force_max = max(norm_force_values)
            norm_force_margin = (norm_force_max - norm_force_min) * 0.1
            norm_force_limits = (
                norm_force_min - norm_force_margin,
                norm_force_max + norm_force_margin,
            )
        else:
            norm_force_limits = (-0.1, 1.2)  # Fallback

        # Calculate min/max for strains
        strain_values = []
        if pre_consolidated is not None and strain_cols:
            for col in strain_cols:
                if col in pre_consolidated.columns:
                    strain_values.extend(pre_consolidated[col].dropna().values)
        if post_consolidated is not None and strain_cols:
            for col in strain_cols:
                if col in post_consolidated.columns:
                    strain_values.extend(post_consolidated[col].dropna().values)

        # Use consistent strain limits for averaged plots
        strain_limits = (-0.02, 0.03)

        # EMG limits remain fixed since it's normalized 0-1
        adjusted_emg_limits = (-0.05, 1.05)
        for col in range(3):
            ax[2, col].set_ylim(adjusted_emg_limits)
            ax[3, col].set_ylim(norm_force_limits)
            ax[4, col].set_ylim(strain_limits)

        # Set titles and labels
        ax[0, 0].set_title("Hip")
        ax[0, 1].set_title("Knee")
        ax[0, 2].set_title("Ankle")
        ax[2, 0].set_title("M. Biceps Femoris Long Head")
        ax[2, 1].set_title("M. Semitendinosus")
        ax[2, 2].set_title("M. Semimembranosus")
        ax[0, 0].set_ylabel("Angle (degrees)")
        ax[1, 0].set_ylabel("Moment (Nm/kg)")
        ax[2, 0].set_ylabel("Adjusted EMG (0-1)")
        ax[3, 0].set_ylabel("Normalized Force (F/Fmax)")
        ax[4, 0].set_ylabel("Strain (Δfl/ofl)")

        # Define plot configurations for each subplot
        all_plot_configs = [
            # Row 0: Angles
            (0, 0, "hip_flexion", "Hip Flexion"),
            (0, 1, "knee_angle", "Knee Flexion"),
            (0, 2, "ankle_angle", "Ankle Flexion"),
            # Row 1: Moments
            (1, 0, "hip_flexion_moment", "Hip Flexion Moment"),
            (1, 1, "knee_angle_moment", "Knee Flexion Moment"),
            (1, 2, "ankle_angle_moment", "Ankle Flexion Moment"),
            # Row 2: Adjusted EMG
            (2, 0, "bflh_adjusted_emg", "Biceps Femoris Activation"),
            (2, 1, "semiten_adjusted_emg", "Semitendinosus Activation"),
            (2, 2, "semimem_adjusted_emg", "Semimembranosus Activation"),
            # Row 3: Normalized Forces
            (3, 0, "bflh_norm_force", "Biceps Femoris Long Head"),
            (3, 1, "semiten_norm_force", "Semitendinosus"),
            (3, 2, "semimem_norm_force", "Semimembranosus"),
            # Row 4: Strains
            (4, 0, "bflh_strain", "Biceps Femoris Strain"),
            (4, 1, "semiten_strain", "Semitendinosus Strain"),
            (4, 2, "semimem_strain", "Semimembranosus Strain"),
        ]

        # Filter plot configs to only include columns that exist in cols_to_process
        plot_configs = [config for config in all_plot_configs if config[2] in cols_to_process]

        # Store lines for hover functionality - organized by axis position
        lines_by_axis = {}
        for row in range(5):
            for col in range(3):
                lines_by_axis[(row, col)] = []

        # Plot muscle-specific responders for each column
        # For rows 0-1 (angles, moments), use bflh classification for all columns
        # For rows 2-4 (EMG, force, strain), use muscle-specific classification per column
        for row, col, column, _ in plot_configs:
            # Determine which muscle to use for this subplot
            if (row, col) in column_to_muscle:
                muscle = column_to_muscle[(row, col)]
            else:
                # For angles and moments (rows 0-1), use bflh
                muscle = "bflh"

            # Get the responder/non-responder IDs for this muscle
            muscle_responder_ids = strain_responders_dict[muscle]
            muscle_non_responder_ids = strain_non_responders_dict[muscle]

            # Plot non-responders first (low alpha, green)
            for subject_id in muscle_non_responder_ids:
                # Plot pre data if requested
                if trial_type in ["pre", "both"] and pre_consolidated is not None:
                    subject_df_pre = pre_consolidated[pre_consolidated["subject_id"] == subject_id]

                    if not subject_df_pre.empty and column in subject_df_pre.columns:
                        (line,) = ax[row, col].plot(
                            subject_df_pre["index"],
                            subject_df_pre[column],
                            color="green",
                            linestyle="-",
                            alpha=0.3,
                            linewidth=1.5,
                        )
                        lines_by_axis[(row, col)].append((line, subject_id, "Pre"))

                # Plot post data if requested
                if trial_type in ["post", "both"] and post_consolidated is not None:
                    subject_df_post = post_consolidated[post_consolidated["subject_id"] == subject_id]

                    if not subject_df_post.empty and column in subject_df_post.columns:
                        (line,) = ax[row, col].plot(
                            subject_df_post["index"],
                            subject_df_post[column],
                            color="green",
                            linestyle="--" if trial_type == "both" else "-",
                            alpha=0.3,
                            linewidth=1.5,
                        )
                        lines_by_axis[(row, col)].append((line, subject_id, "Post"))

            # Plot responders on top (full alpha, red)
            for subject_id in muscle_responder_ids:
                # Plot pre data if requested
                if trial_type in ["pre", "both"] and pre_consolidated is not None:
                    subject_df_pre = pre_consolidated[pre_consolidated["subject_id"] == subject_id]

                    if not subject_df_pre.empty and column in subject_df_pre.columns:
                        (line,) = ax[row, col].plot(
                            subject_df_pre["index"],
                            subject_df_pre[column],
                            color="red",
                            linestyle="-",
                            alpha=1.0,
                            linewidth=1.5,
                        )
                        lines_by_axis[(row, col)].append((line, subject_id, "Pre"))

                # Plot post data if requested
                if trial_type in ["post", "both"] and post_consolidated is not None:
                    subject_df_post = post_consolidated[post_consolidated["subject_id"] == subject_id]

                    if not subject_df_post.empty and column in subject_df_post.columns:
                        (line,) = ax[row, col].plot(
                            subject_df_post["index"],
                            subject_df_post[column],
                            color="red",
                            linestyle="--" if trial_type == "both" else "-",
                            alpha=1.0,
                            linewidth=1.5,
                        )
                        lines_by_axis[(row, col)].append((line, subject_id, "Post"))

            # Add vertical lines for average initial contact times
            if trial_type in ["pre", "both"] and avg_initial_contact_pre is not None:
                ax[row, col].axvline(
                    x=avg_initial_contact_pre,
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.5,
                    zorder=0,
                )
            if trial_type in ["post", "both"] and avg_initial_contact_post is not None:
                ax[row, col].axvline(
                    x=avg_initial_contact_post,
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.5,
                    zorder=0,
                )

        # Create legend showing overall counts (combined across all muscles)
        # Create legend showing muscle-specific counts
        from matplotlib.lines import Line2D

        legend_lines = []
        legend_labels = []

        if trial_type == "both":
            # Non-responders
            legend_lines.append(
                Line2D([0], [0], color="green", linestyle="-", alpha=0.3, linewidth=1.5),
            )
            legend_labels.append("Non-responders Pre (muscle-specific)")
            legend_lines.append(
                Line2D(
                    [0],
                    [0],
                    color="green",
                    linestyle="--",
                    alpha=0.3,
                    linewidth=1.5,
                ),
            )
            legend_labels.append("Non-responders Post (muscle-specific)")

            # Responders
            legend_lines.append(
                Line2D([0], [0], color="red", linestyle="-", alpha=1.0, linewidth=1.5),
            )
            legend_labels.append("Responders Pre (muscle-specific)")
            legend_lines.append(
                Line2D([0], [0], color="red", linestyle="--", alpha=1.0, linewidth=1.5),
            )
            legend_labels.append("Responders Post (muscle-specific)")

            # Initial contact markers
            legend_lines.append(
                Line2D([0], [0], color="blue", linestyle=":", linewidth=1.5, alpha=0.5),
            )
            legend_labels.append("Avg Initial Contact Pre")
            legend_lines.append(
                Line2D(
                    [0],
                    [0],
                    color="orange",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.5,
                ),
            )
            legend_labels.append("Avg Initial Contact Post")
        else:
            # Non-responders
            legend_lines.append(
                Line2D([0], [0], color="green", linestyle="-", alpha=0.3, linewidth=1.5),
            )
            legend_labels.append("Non-responders (muscle-specific)")

            # Responders
            legend_lines.append(
                Line2D([0], [0], color="red", linestyle="-", alpha=1.0, linewidth=1.5),
            )
            legend_labels.append("Responders (muscle-specific)")

            # Initial contact marker
            if trial_type == "pre" and avg_initial_contact_pre is not None:
                legend_lines.append(
                    Line2D(
                        [0],
                        [0],
                        color="blue",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.5,
                    ),
                )
                legend_labels.append("Avg Initial Contact Pre")
            elif trial_type == "post" and avg_initial_contact_post is not None:
                legend_lines.append(
                    Line2D(
                        [0],
                        [0],
                        color="orange",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.5,
                    ),
                )
                legend_labels.append("Avg Initial Contact Post")

        fig.legend(legend_lines, legend_labels, loc="upper right", ncol=2, fontsize=9)

        # Add text annotation showing muscle-specific counts
        muscle_counts_text = "Muscle-specific responder counts:\n"
        for muscle in all_muscles:
            n_resp = len(strain_responders_dict[muscle])
            n_non_resp = len(strain_non_responders_dict[muscle])
            muscle_counts_text += f"{muscle}: R={n_resp}, NR={n_non_resp}\n"

        fig.text(
            0.02,
            0.98,
            muscle_counts_text,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Add interactive hover annotations
        annotations = {}
        for (row, col), lines_data in lines_by_axis.items():
            if not lines_data:
                continue

            annot = ax[row, col].annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="yellow", alpha=0.9),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                zorder=1000,
            )
            annot.set_visible(False)
            annotations[(row, col)] = annot

        def on_hover(event):
            """Handle mouse hover events to show subject ID."""
            if event.inaxes is None:
                return

            # Find which subplot the mouse is in
            subplot_pos = None
            for row in range(5):
                for col in range(3):
                    if event.inaxes == ax[row, col]:
                        subplot_pos = (row, col)
                        break
                if subplot_pos:
                    break

            if subplot_pos is None:
                return

            lines_data = lines_by_axis.get(subplot_pos, [])
            annot = annotations.get(subplot_pos)

            if not lines_data or annot is None:
                return

            # Check if mouse is near any line
            for line, subject_id, trial_label in lines_data:
                if line.contains(event)[0]:
                    # Get the closest point on the line
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()

                    # Find closest x-index
                    if len(x_data) > 0:
                        distances = np.abs(x_data - event.xdata)
                        idx = np.argmin(distances)

                        annot.xy = (x_data[idx], y_data[idx])
                        annot.set_text(f"Subject {subject_id} ({trial_label})")
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return

            # If no line is hovered, hide annotation
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

        # Connect hover event
        fig.canvas.mpl_connect("motion_notify_event", on_hover)

        fig.supxlabel("Sprint Cycle (%)")

        return fig, ax

    def _plot_responder_comparison_individual_interactive(self, trial_type="pre"):
        """
        Create interactive Plotly version of responder comparison plot.
        Hover over any line to see subject ID and details.

        Args:
            trial_type: Type of trial to plot. Options: "pre", "post", or "both" (default: "pre")

        Returns:
            Plotly Figure object

        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Prepare data
        master_df = self.master_df
        analysis_utils = Analysis_utils()

        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        # Define cols_to_process once
        cols_to_process = [
            col for col in master_df.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]

        # Initialize consolidated dataframes as None
        pre_consolidated = None
        post_consolidated = None

        if trial_type in ["pre", "both"]:
            # Consolidate pre-trial data
            pre_df = master_df[master_df["trial"].isin(pre_trials)].copy()
            pre_consolidated = pre_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()

        if trial_type in ["post", "both"]:
            # Get post-trial data
            post_df = master_df[master_df["trial"].isin(post_trials)].copy()
            post_consolidated = post_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()

        # Get subject groups
        all_muscles = ["bflh", "semiten", "semimem"]
        strain_responders_dict = analysis_utils.get_norm_fiber_length_responders_ids(
            muscles=all_muscles,
        )
        strain_non_responders_dict = analysis_utils.get_norm_fiber_length_non_responders_ids(
            muscles=all_muscles,
        )

        # Get initial contact times for vertical line markers
        initial_contact_df = analysis_utils.get_initial_foot_contact_time_df(
            time_normalized=True,
        )
        if not initial_contact_df.empty:
            avg_initial_contact_pre = initial_contact_df["initial_contact_time_pre"].mean()
            avg_initial_contact_post = initial_contact_df["initial_contact_time_post"].mean()
        else:
            avg_initial_contact_pre = None
            avg_initial_contact_post = None

        # Map columns to their corresponding muscle
        column_to_muscle = {
            # Row 2: EMG
            (2, 0): "bflh",
            (2, 1): "semiten",
            (2, 2): "semimem",
            # Row 3: Forces
            (3, 0): "bflh",
            (3, 1): "semiten",
            (3, 2): "semimem",
            # Row 4: Strains
            (4, 0): "bflh",
            (4, 1): "semiten",
            (4, 2): "semimem",
        }

        # Create subplot titles
        subplot_titles = [
            "Hip Angle",
            "Knee Angle",
            "Ankle Angle",
            "Hip Moment",
            "Knee Moment",
            "Ankle Moment",
            "M. Biceps Femoris Long Head EMG",
            "M. Semitendinosus EMG",
            "M. Semimembranosus EMG",
            "M. Biceps Femoris Force",
            "M. Semitendinosus Force",
            "M. Semimembranosus Force",
            "M. Biceps Femoris Strain",
            "M. Semitendinosus Strain",
            "M. Semimembranosus Strain",
        ]

        # Create figure with subplots
        fig = make_subplots(
            rows=5,
            cols=3,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            row_heights=[0.2, 0.2, 0.2, 0.2, 0.2],
            x_title="Sprint Cycle (%)",
        )

        # Update title based on trial_type
        if trial_type == "pre":
            title_suffix = "Pre-fatigue"
        elif trial_type == "post":
            title_suffix = "Post-fatigue"
        else:
            title_suffix = "Pre vs Post-fatigue"

        # Plot configurations
        plot_configs = [
            # Row 0: Angles
            (1, 1, "hip_flexion", "Angle (degrees)"),
            (1, 2, "knee_angle", "Angle (degrees)"),
            (1, 3, "ankle_angle", "Angle (degrees)"),
            # Row 1: Moments
            (2, 1, "hip_flexion_moment", "Moment (Nm/kg)"),
            (2, 2, "knee_angle_moment", "Moment (Nm/kg)"),
            (2, 3, "ankle_angle_moment", "Moment (Nm/kg)"),
            # Row 2: Adjusted EMG
            (3, 1, "bflh_adjusted_emg", "Adjusted EMG (0-1)"),
            (3, 2, "semiten_adjusted_emg", "Adjusted EMG (0-1)"),
            (3, 3, "semimem_adjusted_emg", "Adjusted EMG (0-1)"),
            # Row 3: Normalized Forces
            (4, 1, "bflh_norm_force", "Normalized Force"),
            (4, 2, "semiten_norm_force", "Normalized Force"),
            (4, 3, "semimem_norm_force", "Normalized Force"),
            # Row 4: Strains
            (5, 1, "bflh_strain", "Strain (Δfl/ofl)"),
            (5, 2, "semiten_strain", "Strain (Δfl/ofl)"),
            (5, 3, "semimem_strain", "Strain (Δfl/ofl)"),
        ]

        # Filter to only existing columns
        plot_configs = [config for config in plot_configs if config[2] in cols_to_process]

        # Plot data
        for row, col, column, ylabel in plot_configs:
            # Determine which muscle to use for this subplot
            subplot_key = (row - 1, col - 1)  # Adjust for 0-indexing
            if subplot_key in column_to_muscle:
                muscle = column_to_muscle[subplot_key]
            else:
                muscle = "bflh"  # Default for angles and moments

            muscle_responder_ids = strain_responders_dict[muscle]
            muscle_non_responder_ids = strain_non_responders_dict[muscle]

            # Plot non-responders (green, semi-transparent)
            for subject_id in muscle_non_responder_ids:
                if trial_type in ["pre", "both"] and pre_consolidated is not None:
                    subject_df = pre_consolidated[pre_consolidated["subject_id"] == subject_id]
                    if not subject_df.empty and column in subject_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=subject_df["index"],
                                y=subject_df[column],
                                mode="lines",
                                name=f"Subject {subject_id}",
                                line=dict(color="green", width=1.5, dash="solid"),
                                opacity=0.3,
                                showlegend=False,
                                hovertemplate=f"<b>Subject {subject_id}</b><br>"
                                f"Status: Non-responder ({muscle})<br>"
                                "Phase: Pre-fatigue<br>"
                                "Cycle: %{{x:.1f}}%<br>"
                                "Value: %{{y:.3f}}<br>"
                                "<extra></extra>",
                                legendgroup="non-responder",
                            ),
                            row=row,
                            col=col,
                        )

                if trial_type in ["post", "both"] and post_consolidated is not None:
                    subject_df = post_consolidated[post_consolidated["subject_id"] == subject_id]
                    if not subject_df.empty and column in subject_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=subject_df["index"],
                                y=subject_df[column],
                                mode="lines",
                                name=f"Subject {subject_id}",
                                line=dict(
                                    color="green",
                                    width=1.5,
                                    dash="dash" if trial_type == "both" else "solid",
                                ),
                                opacity=0.3,
                                showlegend=False,
                                hovertemplate=f"<b>Subject {subject_id}</b><br>"
                                f"Status: Non-responder ({muscle})<br>"
                                "Phase: Post-fatigue<br>"
                                "Cycle: %{{x:.1f}}%<br>"
                                "Value: %{{y:.3f}}<br>"
                                "<extra></extra>",
                                legendgroup="non-responder",
                            ),
                            row=row,
                            col=col,
                        )

            # Plot responders (red, full opacity)
            for subject_id in muscle_responder_ids:
                if trial_type in ["pre", "both"] and pre_consolidated is not None:
                    subject_df = pre_consolidated[pre_consolidated["subject_id"] == subject_id]
                    if not subject_df.empty and column in subject_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=subject_df["index"],
                                y=subject_df[column],
                                mode="lines",
                                name=f"Subject {subject_id}",
                                line=dict(color="red", width=1.5, dash="solid"),
                                opacity=1.0,
                                showlegend=False,
                                hovertemplate=f"<b>Subject {subject_id}</b><br>"
                                f"Status: Responder ({muscle})<br>"
                                "Phase: Pre-fatigue<br>"
                                "Cycle: %{{x:.1f}}%<br>"
                                "Value: %{{y:.3f}}<br>"
                                "<extra></extra>",
                                legendgroup="responder",
                            ),
                            row=row,
                            col=col,
                        )

                if trial_type in ["post", "both"] and post_consolidated is not None:
                    subject_df = post_consolidated[post_consolidated["subject_id"] == subject_id]
                    if not subject_df.empty and column in subject_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=subject_df["index"],
                                y=subject_df[column],
                                mode="lines",
                                name=f"Subject {subject_id}",
                                line=dict(
                                    color="red",
                                    width=1.5,
                                    dash="dash" if trial_type == "both" else "solid",
                                ),
                                opacity=1.0,
                                showlegend=False,
                                hovertemplate=f"<b>Subject {subject_id}</b><br>"
                                f"Status: Responder ({muscle})<br>"
                                "Phase: Post-fatigue<br>"
                                "Cycle: %{{x:.1f}}%<br>"
                                "Value: %{{y:.3f}}<br>"
                                "<extra></extra>",
                                legendgroup="responder",
                            ),
                            row=row,
                            col=col,
                        )

        # Add vertical lines for initial contact times
        for row, col, column, ylabel in plot_configs:
            if trial_type in ["pre", "both"] and avg_initial_contact_pre is not None:
                fig.add_vline(
                    x=avg_initial_contact_pre,
                    line_dash="dot",
                    line_color="black",
                    line_width=1.5,
                    opacity=0.5,
                    row=row,  # type: ignore
                    col=col,  # type: ignore
                )
            if trial_type in ["post", "both"] and avg_initial_contact_post is not None:
                fig.add_vline(
                    x=avg_initial_contact_post,
                    line_dash="dot",
                    line_color="black",
                    line_width=1.5,
                    opacity=0.5,
                    row=row,  # type: ignore
                    col=col,  # type: ignore
                )

        # Update layout
        fig.update_layout(
            title_text=f"Individual Subject Comparison: Responders vs Non-responders ({title_suffix})",
            height=1200,
            showlegend=True,
            hovermode="closest",
            template="plotly_white",
        )

        # Add custom legend
        if trial_type == "both":
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="green", width=2, dash="solid"),
                    opacity=0.3,
                    name="Non-responders Pre",
                    showlegend=True,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="green", width=2, dash="dash"),
                    opacity=0.3,
                    name="Non-responders Post",
                    showlegend=True,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="red", width=2, dash="solid"),
                    opacity=1.0,
                    name="Responders Pre",
                    showlegend=True,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    opacity=1.0,
                    name="Responders Post",
                    showlegend=True,
                ),
            )
            # Initial contact marker
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="black", width=2, dash="dot"),
                    opacity=0.5,
                    name="Avg Initial Contact",
                    showlegend=True,
                ),
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="green", width=2),
                    opacity=0.3,
                    name="Non-responders (muscle-specific)",
                    showlegend=True,
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="red", width=2),
                    opacity=1.0,
                    name="Responders (muscle-specific)",
                    showlegend=True,
                ),
            )
            # Initial contact marker
            if (trial_type == "pre" and avg_initial_contact_pre is not None) or (
                trial_type == "post" and avg_initial_contact_post is not None
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color="black", width=2, dash="dot"),
                        opacity=0.5,
                        name="Avg Initial Contact",
                        showlegend=True,
                    ),
                )

        # Update y-axis labels
        for i in range(1, 6):
            if i == 1:
                ylabel = "Angle (degrees)"
            elif i == 2:
                ylabel = "Moment (Nm/kg)"
            elif i == 3:
                ylabel = "Adjusted EMG (0-1)"
            elif i == 4:
                ylabel = "Normalized Force"
            else:
                ylabel = "Strain (Δfl/ofl)"

            fig.update_yaxes(title_text=ylabel, row=i, col=1)

        return fig

    def master_plot_with_peak_norm_fiber_lengths(self, p=0.0042):
        """Plot before/after master plots with peak normalized fiber lengths in bottom row."""
        # Prepare data
        master_df = self.master_df
        analysis_utils_inst = Analysis_utils()
        spm_sig_idx = analysis_utils_inst.get_spm_sig_index(p=p)

        # Get mean foot contact times for pre and post
        initial_contact_df = analysis_utils_inst.get_initial_foot_contact_time_df(
            time_normalized=True,
        )
        if not initial_contact_df.empty:
            mean_contact_pre = initial_contact_df["initial_contact_time_pre"].mean()
            mean_contact_post = initial_contact_df["initial_contact_time_post"].mean()
        else:
            mean_contact_pre = None
            mean_contact_post = None

        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        pre_df = master_df[master_df["trial"].isin(pre_trials)].copy()
        pre_df["trial"] = "Run_baseline"  # Standardize pre-trial name

        cols_to_process = [
            col for col in master_df.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]
        pre_consolidated = pre_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()
        pre_consolidated["trial"] = "Run_baseline"

        post_df = master_df[master_df["trial"].isin(post_trials)].copy()

        df_filtered = pd.concat([pre_consolidated, post_df], ignore_index=True)

        df_mean = df_filtered.groupby(["trial", "index"])[cols_to_process].mean().reset_index()

        df_std = df_filtered.groupby(["trial", "index"])[cols_to_process].std().reset_index()

        df_count = df_filtered.groupby(["trial", "index"])[cols_to_process].count().reset_index()

        # Calculate 95% confidence interval
        df_ci = self.calculate_confidence_intervals(df_std, df_count, cols_to_process)

        # Calculate subject-level peak normalized fiber lengths for pre/post trials.
        peak_fiber_length = (
            df_filtered.groupby(["subject_id", "trial"])[
                [
                    "bflh_fiber_lengths",
                    "semiten_fiber_lengths",
                    "semimem_fiber_lengths",
                ]
            ]
            .max()
            .reset_index()
        )

        peak_fiber_length_wide = peak_fiber_length.pivot(
            index="subject_id",
            columns="trial",
            values=[
                "bflh_fiber_lengths",
                "semiten_fiber_lengths",
                "semimem_fiber_lengths",
            ],
        )
        peak_fiber_length_wide.columns = [f"{col}_{trial}" for col, trial in peak_fiber_length_wide.columns]
        peak_fiber_length_wide = peak_fiber_length_wide.reset_index()

        # Create subplots with shared x-axis for rows 0-2, independent for row 3
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3)

        # Create axes with shared x for rows 0-2
        ax = []
        first_ax = None
        for row in range(3):
            row_axes = []
            for col in range(3):
                if row == 0 and col == 0:
                    # First subplot - no sharing yet
                    axis = fig.add_subplot(gs[row, col])
                    first_ax = axis
                else:
                    # Share x with first subplot in rows 0-2
                    axis = fig.add_subplot(gs[row, col], sharex=first_ax)
                row_axes.append(axis)
            ax.append(row_axes)

        # Row 3 has independent x-axes
        row_3_axes = []
        for col in range(3):
            axis = fig.add_subplot(gs[3, col])
            row_3_axes.append(axis)
        ax.append(row_3_axes)

        # Convert to numpy array for easier indexing
        ax = np.array(ax)

        norm_force_limits = (-0.1, 1)

        ax[2, 0].set_ylim(norm_force_limits)
        ax[2, 1].set_ylim(norm_force_limits)
        ax[2, 2].set_ylim(norm_force_limits)
        ax[0, 0].set_title("Hip")
        ax[0, 1].set_title("Knee")
        ax[0, 2].set_title("Ankle")
        ax[2, 0].set_title("M. Biceps Femoris Long Head")
        ax[2, 1].set_title("M. Semitendinosus")
        ax[2, 2].set_title("M. Semimembranosus")
        ax[0, 0].set_ylabel("Angle (degrees)")
        ax[1, 0].set_ylabel("Moment (Nm/kg)")
        ax[2, 0].set_ylabel("Normalized Force (F/Fmax)")
        ax[3, 0].set_ylabel("Peak Norm Fiber Length (fl/ofl)")

        # Define plot configurations for each subplot
        plot_configs = [
            # Row 0: Angles
            (0, 0, "hip_flexion", "Hip Flexion"),
            (0, 1, "knee_angle", "Knee Flexion"),
            (0, 2, "ankle_angle", "Ankle Flexion"),
            # Row 1: Moments
            (1, 0, "hip_flexion_moment", "Hip Flexion Moment"),
            (1, 1, "knee_angle_moment", "Knee Flexion Moment"),
            (1, 2, "ankle_angle_moment", "Ankle Flexion Moment"),
            # Row 2: Normalized Forces
            (2, 0, "bflh_norm_force", "Biceps Femoris Long Head"),
            (2, 1, "semiten_norm_force", "Semitendinosus"),
            (2, 2, "semimem_norm_force", "Semimembranosus"),
        ]

        # Row 3: Peak normalized fiber lengths
        fiber_length_plot_configs = [
            (
                3,
                0,
                "bflh_fiber_lengths_Run_baseline",
                "bflh_fiber_lengths_RunL1",
                "M. Biceps Femoris Long Head",
            ),
            (
                3,
                1,
                "semiten_fiber_lengths_Run_baseline",
                "semiten_fiber_lengths_RunL1",
                "M. Semitendinosus",
            ),
            (
                3,
                2,
                "semimem_fiber_lengths_Run_baseline",
                "semimem_fiber_lengths_RunL1",
                "M. Semimembranosus",
            ),
        ]

        # Loop through all plot configurations
        for row, col, column, label in plot_configs:
            # Plot Pre (baseline) data
            self.plot_trial_data(
                ax[row, col],
                "Run_baseline",
                column,
                f"{label} - Pre",
                "blue",
                df_mean,
                df_ci,
            )

            # Plot Post (RunL1) data
            self.plot_trial_data(
                ax[row, col],
                "RunL1",
                column,
                f"{label} - Post",
                "orange",
                df_mean,
                df_ci,
            )

            # Add significance shading if available
            sig_idx = spm_sig_idx.get(column)
            if sig_idx is not None:
                ax[row, col].fill_between(
                    range(len(sig_idx)),
                    ax[row, col].get_ylim()[0],
                    ax[row, col].get_ylim()[1],
                    where=sig_idx,
                    color="red",
                    alpha=0.1,
                    label="Significant Difference",
                )

            # Add colored dashed vertical lines for mean foot contact (pre=blue, post=orange)
            if mean_contact_pre is not None:
                ax[row, col].axvline(
                    x=mean_contact_pre,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                )
            if mean_contact_post is not None:
                ax[row, col].axvline(
                    x=mean_contact_post,
                    color="orange",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                )

        for row, col, pre_col, post_col, label in fiber_length_plot_configs:
            # Sort indices: plot decreased values first (green), then increased (red)
            # This ensures red lines overlay green lines
            indices_decreased = []
            indices_increased = []
            for idx in range(len(peak_fiber_length_wide)):
                pre_val = peak_fiber_length_wide[pre_col].iloc[idx]
                post_val = peak_fiber_length_wide[post_col].iloc[idx]
                if post_val < pre_val:
                    indices_decreased.append(idx)
                else:
                    indices_increased.append(idx)

            # Plot decreased values first (underneath)
            for idx in indices_decreased:
                pre_val = peak_fiber_length_wide[pre_col].iloc[idx]
                post_val = peak_fiber_length_wide[post_col].iloc[idx]
                ax[row, col].plot(
                    [1, 2],
                    [pre_val, post_val],
                    marker="o",
                    color="green",
                    alpha=0.5,
                )

            # Plot increased values second (on top)
            for idx in indices_increased:
                pre_val = peak_fiber_length_wide[pre_col].iloc[idx]
                post_val = peak_fiber_length_wide[post_col].iloc[idx]
                ax[row, col].plot(
                    [1, 2],
                    [pre_val, post_val],
                    marker="o",
                    color="red",
                    alpha=0.5,
                )

            ax[row, col].set_xticks([1, 2])
            ax[row, col].set_xticklabels(["Pre", "Post"])
            ax[row, col].boxplot(
                peak_fiber_length_wide[pre_col].dropna(),
                positions=[1],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="blue", alpha=0.3),
                medianprops=dict(color="blue", linewidth=2),
                whiskerprops=dict(color="blue"),
                capprops=dict(color="blue"),
                flierprops=dict(marker="", linestyle="none"),
                manage_ticks=False,
            )
            ax[row, col].boxplot(
                peak_fiber_length_wide[post_col].dropna(),
                positions=[2],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="orange", alpha=0.3),
                medianprops=dict(color="orange", linewidth=2),
                whiskerprops=dict(color="orange"),
                capprops=dict(color="orange"),
                flierprops=dict(marker="", linestyle="none"),
                manage_ticks=False,
            )

        min_y_norm_force = (
            df_filtered[
                [
                    "bflh_norm_force",
                    "semiten_norm_force",
                    "semimem_norm_force",
                ]
            ]
            .min(skipna=True)
            .min(skipna=True)
        )
        max_y_norm_force = (
            df_filtered[
                [
                    "bflh_norm_force",
                    "semiten_norm_force",
                    "semimem_norm_force",
                ]
            ]
            .max(skipna=True)
            .max(skipna=True)
        )
        min_y_fiber_length = (
            df_filtered[
                [
                    "bflh_fiber_lengths",
                    "semiten_fiber_lengths",
                    "semimem_fiber_lengths",
                ]
            ]
            .min(skipna=True)
            .min(skipna=True)
        )
        max_y_fiber_length = (
            df_filtered[
                [
                    "bflh_fiber_lengths",
                    "semiten_fiber_lengths",
                    "semimem_fiber_lengths",
                ]
            ]
            .max(skipna=True)
            .max(skipna=True)
        )

        if (
            pd.isna(min_y_norm_force)
            or pd.isna(max_y_norm_force)
            or pd.isna(min_y_fiber_length)
            or pd.isna(max_y_fiber_length)
            or np.isinf(min_y_norm_force)
            or np.isinf(max_y_norm_force)
            or np.isinf(min_y_fiber_length)
            or np.isinf(max_y_fiber_length)
        ):
            print("Warning: NaN or Inf values in consolidated data")
            return None, None

        # Set x-axis label only for bottom row of shared x-axes (row 2)
        for col in range(3):
            ax[2, col].set_xlabel("Sprint Cycle (%)")

        # Add legend for Pre and Post
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        custom_lines = [
            Line2D([0], [0], color="blue", linewidth=2, label="Pre"),
            Line2D([0], [0], color="orange", linewidth=2, label="Post"),
            Patch(facecolor="blue", alpha=0.2, label="95% CI (Pre)"),
            Patch(facecolor="orange", alpha=0.2, label="95% CI (Post)"),
            Line2D([0], [0], color="blue", linestyle="--", linewidth=1.5, label="Mean Contact Pre"),
            Line2D([0], [0], color="orange", linestyle="--", linewidth=1.5, label="Mean Contact Post"),
            Patch(facecolor="red", alpha=0.1, label="p<0.01"),
        ]
        fig.legend(
            custom_lines,
            ["Pre", "Post", "95% CI (Pre)", "95% CI (Post)", "Mean Contact Pre", "Mean Contact Post", "p<0.01"],
            loc="upper right",
            ncol=4,
            fontsize=9,
        )

        # Add figure description text below the plots
        # description_text = (
        #     f"Ensemble averages (n = {df_filtered['subject_id'].nunique()}) of hip, knee, and ankle angles, moments, normalized muscle forces, "
        #     "and peak muscle strain for a sprint cycle pre (solid blue) and post (solid orange) a repeated sprints protocol. "
        #     "Shaded regions indicate 95% confidence intervals. Red shaded regions denote statistically significant differences (p<0.01, Statistical Parametric Mapping). "
        #     "The bottom row shows individual participant responses (connecting lines) and group means (bars) for peak muscle strain. "
        #     "Green connecting lines indicate a decrease in peak strain, while red lines indicate an increase. "
        #     "Sprint cycles are defined from toe-off to toe-off and all data were normalized to 101 points (0-100%)."
        # )

        # Adjust layout to reduce whitespace and make room for text
        fig.subplots_adjust(
            left=0.06,
            right=0.98,
            top=0.94,
            bottom=0.07,
            hspace=0.25,
            wspace=0.25,
        )
        # fig.text(
        #     0.52,
        #     0.02,
        #     r"$\mathbf{Fig.\ 1}$: " + description_text,
        #     ha="center",
        #     va="center",
        #     fontsize=10,
        #     wrap=True,
        # )

        return fig, ax

    def master_plot_with_peak_strains(self, p=0.0042):
        """Backward-compatible wrapper for renamed peak normalized fiber length master plot."""
        return self.master_plot_with_peak_norm_fiber_lengths(p=p)

    def plot_peak_norm_force(self, interactive=False):
        """
        Plot peak normalized force with optional interactivity.

        Args:
            interactive: If True and plotly is available, return interactive plotly figure

        Returns:
            If interactive=True: plotly figure object or None
            Otherwise: (matplotlib fig, ax)

        """
        peak_norm_force = Analysis_utils().get_peak_norm_force()

        # If plotly requested and available, create interactive plot
        if interactive:
            return self._plot_peak_norm_force_interactive(peak_norm_force)

        mean_peak_norm_force = peak_norm_force.mean()

        fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        fig.suptitle("Peak Muscle Normalized Force Across All Participants")

        ax[0].set_title("M. Biceps Femoris Long Head")
        ax[1].set_title("M. Semitendinosus")
        ax[2].set_title("M. Semimembranosus")
        ax[0].set_ylabel("Peak Normalized Force (F/Fmax)")

        plot_configs = [
            (
                ax[0],
                "bflh_peak_norm_force_pre",
                "bflh_peak_norm_force_post",
                "M. Biceps Femoris Long Head",
            ),
            (
                ax[1],
                "semiten_peak_norm_force_pre",
                "semiten_peak_norm_force_post",
                "M. Semitendinosus",
            ),
            (
                ax[2],
                "semimem_peak_norm_force_pre",
                "semimem_peak_norm_force_post",
                "M. Semimembranosus",
            ),
        ]

        for axis, pre_col, post_col, label in plot_configs:
            for idx in range(len(peak_norm_force)):
                pre_val = peak_norm_force[pre_col].iloc[idx]
                post_val = peak_norm_force[post_col].iloc[idx]
                color = "green" if post_val < pre_val else "red"
                axis.plot(
                    [1, 2],
                    [pre_val, post_val],
                    marker="o",
                    color=color,
                    alpha=0.5,
                )

            axis.set_xticks([1, 2])
            axis.set_xticklabels(["Pre", "Post"])
            axis.set_title(label)
            axis.bar(
                [1],
                mean_peak_norm_force[pre_col],
                label="Pre",
                color="blue",
                alpha=0.3,
            )
            axis.bar(
                [2],
                mean_peak_norm_force[post_col],
                label="Post",
                color="orange",
                alpha=0.3,
            )

        return fig, ax

    def plot_peak_norm_fiber_length(self, interactive=False):
        """
        Plot peak muscle normalized fiber length with optional interactivity.

        Args:
            interactive: If True and plotly is available, return interactive plotly figure

        Returns:
            If interactive=True: plotly figure object or None
            Otherwise: (matplotlib fig, ax)

        """
        # Load peak normalized fiber length data from CSV
        peak_norm_fiber_length_csv = Path("data/peak_norm_fiber_length.csv")
        legacy_peak_strain_csv = Path("data/peak_strain.csv")
        if peak_norm_fiber_length_csv.exists():
            peak_norm_fiber_length = pd.read_csv(peak_norm_fiber_length_csv, index_col=0)
        elif legacy_peak_strain_csv.exists():
            # Legacy fallback
            peak_norm_fiber_length = pd.read_csv(legacy_peak_strain_csv, index_col=0)
        else:
            # Fallback to computing if CSV doesn't exist
            peak_norm_fiber_length = Analysis_utils().get_peak_norm_fiber_length_df(
                muscles=[
                    "biceps_femoris_long_head",
                    "semitendinosus",
                    "semimembranosus",
                ],
            )

        # If plotly requested and available, create interactive plot
        if interactive:
            return self._plot_peak_norm_fiber_length_interactive(peak_norm_fiber_length)

        mean_peak_norm_fiber_length = peak_norm_fiber_length.mean()

        fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        fig.suptitle("Peak Muscle Normalized Fiber Length Across All Participants")

        ax[0].set_title("M. Biceps Femoris Long Head")
        ax[1].set_title("M. Semitendinosus")
        ax[2].set_title("M. Semimembranosus")
        ax[0].set_ylabel("Peak Norm Fiber Length (fl/ofl)")

        plot_configs = [
            (
                ax[0],
                "biceps_femoris_long_head_peak_norm_fiber_length_pre",
                "biceps_femoris_long_head_peak_norm_fiber_length_post",
                "M. Biceps Femoris Long Head",
            ),
            (
                ax[1],
                "semitendinosus_peak_norm_fiber_length_pre",
                "semitendinosus_peak_norm_fiber_length_post",
                "M. Semitendinosus",
            ),
            (
                ax[2],
                "semimembranosus_peak_norm_fiber_length_pre",
                "semimembranosus_peak_norm_fiber_length_post",
                "M. Semimembranosus",
            ),
        ]

        for axis, pre_col, post_col, label in plot_configs:
            # Sort indices: plot decreased values first (green), then increased (red)
            # This ensures red lines overlay green lines
            indices_decreased = []
            indices_increased = []
            for idx in range(len(peak_norm_fiber_length)):
                pre_val = peak_norm_fiber_length[pre_col].iloc[idx]
                post_val = peak_norm_fiber_length[post_col].iloc[idx]
                if post_val < pre_val:
                    indices_decreased.append(idx)
                else:
                    indices_increased.append(idx)

            # Plot decreased strain first (underneath)
            for idx in indices_decreased:
                pre_val = peak_norm_fiber_length[pre_col].iloc[idx]
                post_val = peak_norm_fiber_length[post_col].iloc[idx]
                axis.plot(
                    [1, 2],
                    [pre_val, post_val],
                    marker="o",
                    color="green",
                    alpha=0.5,
                )

            # Plot increased strain second (on top)
            for idx in indices_increased:
                pre_val = peak_norm_fiber_length[pre_col].iloc[idx]
                post_val = peak_norm_fiber_length[post_col].iloc[idx]
                axis.plot(
                    [1, 2],
                    [pre_val, post_val],
                    marker="o",
                    color="red",
                    alpha=0.5,
                )

            axis.set_xticks([1, 2])
            axis.set_xticklabels(["Pre", "Post"])
            axis.set_title(label)
            axis.bar(
                [1],
                mean_peak_norm_fiber_length[pre_col],
                label="Pre",
                color="blue",
                alpha=0.3,
            )
            axis.bar(
                [2],
                mean_peak_norm_fiber_length[post_col],
                label="Post",
                color="orange",
                alpha=0.3,
            )

        return fig, ax

    def plot_peak_strain(self, interactive=False):
        """Backward-compatible wrapper for plot_peak_norm_fiber_length."""
        return self.plot_peak_norm_fiber_length(interactive=interactive)

    def _plot_peak_norm_fiber_length_interactive(self, peak_norm_fiber_length):
        """Create interactive plotly version of peak normalized fiber length plot."""
        mean_peak_norm_fiber_length = peak_norm_fiber_length.mean()

        # Create subplot
        fig = make_subplots(  # type: ignore
            rows=1,
            cols=3,
            subplot_titles=(
                "M. Biceps Femoris Long Head",
                "M. Semitendinosus",
                "M. Semimembranosus",
            ),
        )

        plot_configs = [
            (
                1,
                "biceps_femoris_long_head_peak_norm_fiber_length_pre",
                "biceps_femoris_long_head_peak_norm_fiber_length_post",
            ),
            (
                2,
                "semitendinosus_peak_norm_fiber_length_pre",
                "semitendinosus_peak_norm_fiber_length_post",
            ),
            (
                3,
                "semimembranosus_peak_norm_fiber_length_pre",
                "semimembranosus_peak_norm_fiber_length_post",
            ),
        ]

        for col_idx, pre_col, post_col in plot_configs:
            # Add mean bars
            fig.add_trace(
                go.Bar(  # type: ignore
                    x=["Pre"],
                    y=[mean_peak_norm_fiber_length[pre_col]],
                    marker_color="blue",
                    opacity=0.3,
                    name="Pre (mean)",
                    showlegend=(col_idx == 1),
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Bar(  # type: ignore
                    x=["Post"],
                    y=[mean_peak_norm_fiber_length[post_col]],
                    marker_color="orange",
                    opacity=0.3,
                    name="Post (mean)",
                    showlegend=(col_idx == 1),
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )

            # Sort indices: plot decreased strain first (green), then increased (red)
            # This ensures red lines overlay green lines
            indices_decreased = []
            indices_increased = []
            for idx in range(len(peak_norm_fiber_length)):
                pre_val = peak_norm_fiber_length[pre_col].iloc[idx]
                post_val = peak_norm_fiber_length[post_col].iloc[idx]
                if post_val < pre_val:
                    indices_decreased.append(idx)
                else:
                    indices_increased.append(idx)

            # Plot decreased strain first (underneath)
            for idx in indices_decreased:
                subject_id = (
                    peak_norm_fiber_length.index[idx]
                    if "subject_id" not in peak_norm_fiber_length.columns
                    else peak_norm_fiber_length["subject_id"].iloc[idx]
                )
                pre_val = peak_norm_fiber_length[pre_col].iloc[idx]
                post_val = peak_norm_fiber_length[post_col].iloc[idx]

                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=["Pre", "Post"],
                        y=[pre_val, post_val],
                        mode="lines+markers",
                        line=dict(color="green", width=1.5),
                        marker=dict(size=6),
                        opacity=0.5,
                        name=f"Subject {subject_id}",
                        legendgroup=f"subject_{idx}",
                        showlegend=False,
                        hovertemplate=f"Subject {subject_id}<br>%{{y:.4f}} (l-ofl)/ofl<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

            # Plot increased strain second (on top)
            for idx in indices_increased:
                subject_id = (
                    peak_norm_fiber_length.index[idx]
                    if "subject_id" not in peak_norm_fiber_length.columns
                    else peak_norm_fiber_length["subject_id"].iloc[idx]
                )
                pre_val = peak_norm_fiber_length[pre_col].iloc[idx]
                post_val = peak_norm_fiber_length[post_col].iloc[idx]

                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=["Pre", "Post"],
                        y=[pre_val, post_val],
                        mode="lines+markers",
                        line=dict(color="red", width=1.5),
                        marker=dict(size=6),
                        opacity=0.5,
                        name=f"Subject {subject_id}",
                        legendgroup=f"subject_{idx}",
                        showlegend=False,
                        hovertemplate=f"Subject {subject_id}<br>%{{y:.4f}} (l-ofl)/ofl<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

        # Update layout
        fig.update_yaxes(title_text="Peak Norm Fiber Length (fl/ofl)", row=1, col=1)
        fig.update_layout(
            title_text="Peak Muscle Normalized Fiber Length Across All Participants (Interactive)",
            height=500,
            width=1600,
            hovermode="closest",
        )

        return fig

    def _plot_peak_strain_interactive(self, peak_strain):
        """Backward-compatible wrapper for _plot_peak_norm_fiber_length_interactive."""
        return self._plot_peak_norm_fiber_length_interactive(peak_strain)

    def _plot_peak_norm_force_interactive(self, peak_norm_force):
        """Create interactive plotly version of peak normalized force plot."""
        mean_peak_norm_force = peak_norm_force.mean()

        # Create subplot
        fig = make_subplots(  # type: ignore
            rows=1,
            cols=3,
            subplot_titles=(
                "M. Biceps Femoris Long Head",
                "M. Semitendinosus",
                "M. Semimembranosus",
            ),
        )

        plot_configs = [
            (1, "bflh_peak_norm_force_pre", "bflh_peak_norm_force_post"),
            (2, "semiten_peak_norm_force_pre", "semiten_peak_norm_force_post"),
            (3, "semimem_peak_norm_force_pre", "semimem_peak_norm_force_post"),
        ]

        for col_idx, pre_col, post_col in plot_configs:
            # Add mean bars
            fig.add_trace(
                go.Bar(  # type: ignore
                    x=["Pre"],
                    y=[mean_peak_norm_force[pre_col]],
                    marker_color="blue",
                    opacity=0.3,
                    name="Pre (mean)",
                    showlegend=(col_idx == 1),
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Bar(  # type: ignore
                    x=["Post"],
                    y=[mean_peak_norm_force[post_col]],
                    marker_color="orange",
                    opacity=0.3,
                    name="Post (mean)",
                    showlegend=(col_idx == 1),
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )

            # Add individual subject lines
            for idx in range(len(peak_norm_force)):
                subject_id = (
                    peak_norm_force.index[idx]
                    if "subject_id" not in peak_norm_force.columns
                    else peak_norm_force["subject_id"].iloc[idx]
                )
                pre_val = peak_norm_force[pre_col].iloc[idx]
                post_val = peak_norm_force[post_col].iloc[idx]
                color = "green" if post_val < pre_val else "red"

                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=["Pre", "Post"],
                        y=[pre_val, post_val],
                        mode="lines+markers",
                        line=dict(color=color, width=1.5),
                        marker=dict(size=6),
                        opacity=0.5,
                        name=f"Subject {subject_id}",
                        legendgroup=f"subject_{idx}",
                        showlegend=False,
                        hovertemplate=f"Subject {subject_id}<br>%{{y:.3f}} F/Fmax<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

        # Update layout
        fig.update_yaxes(title_text="Peak Normalized Force (F/Fmax)", row=1, col=1)
        fig.update_layout(
            title_text="Peak Muscle Normalized Force Across All Participants (Interactive)",
            height=500,
            width=1600,
            hovermode="closest",
        )

        return fig

    def plot_peak_norm_force_change_vs_peak_norm_fiber_length_change(self, interactive=False):
        """
        Plot peak force change vs peak normalized fiber length change with optional interactivity.

        Args:
            interactive: If True and plotly is available, return interactive plotly figure

        Returns:
            If interactive=True: plotly figure object or None
            Otherwise: (matplotlib fig, ax)

        """
        peak_norm_fiber_length = Analysis_utils().get_peak_norm_fiber_length_df()
        peak_norm_force = Analysis_utils().get_peak_norm_force()

        # Merge dataframes on subject_id
        merged_df = pd.merge(
            peak_norm_fiber_length,
            peak_norm_force,
            on="subject_id",
            suffixes=("_norm_fiber_length", "_norm_force"),
        )

        # If plotly requested and available, create interactive plot
        if interactive:
            return self._plot_peak_force_norm_fiber_length_interactive(merged_df)

        fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        fig.suptitle(
            "Change in Peak Normalized Force vs Change in Peak Norm Fiber Length Across All Participants",
        )

        ax[0].set_title("M. Biceps Femoris Long Head")
        ax[1].set_title("M. Semitendinosus")
        ax[2].set_title("M. Semimembranosus")
        ax[0].set_ylabel("Change in Peak Normalized Force (F/Fmax)")
        ax[0].set_xlabel("Change in Peak Norm Fiber Length (fl/ofl)")
        ax[1].set_xlabel("Change in Peak Norm Fiber Length (fl/ofl)")
        ax[2].set_xlabel("Change in Peak Norm Fiber Length (fl/ofl)")

        plot_configs = [
            (
                ax[0],
                "bflh_peak_norm_fiber_length",
                "bflh_peak_norm_force",
                "M. Biceps Femoris Long Head",
            ),
            (
                ax[1],
                "semiten_peak_norm_fiber_length",
                "semiten_peak_norm_force",
                "M. Semitendinosus",
            ),
            (
                ax[2],
                "semimem_peak_norm_fiber_length",
                "semimem_peak_norm_force",
                "M. Semimembranosus",
            ),
        ]

        for axis, strain_col, norm_force_col, label in plot_configs:
            x = merged_df[f"{strain_col}_post"] - merged_df[f"{strain_col}_pre"]
            y = merged_df[f"{norm_force_col}_post"] - merged_df[f"{norm_force_col}_pre"]
            axis.scatter(x, y)

            # Fit linear regression line
            if len(x) > 1:
                m, b = np.polyfit(x, y, 1)
                axis.plot(x, m * x + b, color="red", linestyle="--")
            axis.set_title(label)
        return fig, ax

    def plot_peak_norm_force_change_vs_peak_strain_change(self, interactive=False):
        """Backward-compatible wrapper for normalized fiber length-based force-change plot."""
        return self.plot_peak_norm_force_change_vs_peak_norm_fiber_length_change(
            interactive=interactive,
        )

    def _plot_peak_force_norm_fiber_length_interactive(self, merged_df):
        """Create interactive plotly version of peak force vs normalized fiber length change plot."""
        # Create subplot
        fig = make_subplots(  # type: ignore
            rows=1,
            cols=3,
            subplot_titles=(
                "M. Biceps Femoris Long Head",
                "M. Semitendinosus",
                "M. Semimembranosus",
            ),
        )

        plot_configs = [
            (1, "bflh_peak_norm_fiber_length", "bflh_peak_norm_force"),
            (2, "semiten_peak_norm_fiber_length", "semiten_peak_norm_force"),
            (3, "semimem_peak_norm_fiber_length", "semimem_peak_norm_force"),
        ]

        for col_idx, strain_col, norm_force_col in plot_configs:
            x = merged_df[f"{strain_col}_post"] - merged_df[f"{strain_col}_pre"]
            y = merged_df[f"{norm_force_col}_post"] - merged_df[f"{norm_force_col}_pre"]

            # Add scatter points with subject IDs
            subject_ids = (
                merged_df["subject_id"].values if "subject_id" in merged_df.columns else merged_df.index.values
            )

            fig.add_trace(
                go.Scatter(  # type: ignore
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=8, color="blue", opacity=0.6),
                    text=[f"Subject {sid}" for sid in subject_ids],
                    hovertemplate="<b>%{text}</b><br>Δ Norm Fiber Length: %{x:.4f}<br>Δ Force: %{y:.4f}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )

            # Add regression line
            if len(x) > 1:
                m, b = np.polyfit(x, y, 1)
                x_line = np.array([x.min(), x.max()])
                y_line = m * x_line + b

                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="red", dash="dash", width=2),
                        name="Regression" if col_idx == 1 else None,
                        showlegend=(col_idx == 1),
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col_idx,
                )

        # Update layout
        fig.update_xaxes(title_text="Change in Peak Strain (Δfl/ofl)")
        fig.update_yaxes(
            title_text="Change in Peak Normalized Force (F/Fmax)",
            row=1,
            col=1,
        )
        fig.update_layout(
            title_text="Change in Peak Normalized Force vs Change in Peak Strain (Interactive)",
            height=500,
            width=1600,
            hovermode="closest",
        )

        return fig

    def _plot_peak_force_strain_interactive(self, merged_df):
        """Backward-compatible wrapper for _plot_peak_force_norm_fiber_length_interactive."""
        return self._plot_peak_force_norm_fiber_length_interactive(merged_df)

    def plot_emg_adjusted_emg_single(self, subject_id: str):
        """
        Plot EMG vs Adjusted EMG for a single subject comparing Pre and Post trials.

        Args:
            subject_id: Subject ID to plot

        Returns:
            fig, ax: Figure and axes objects

        """
        # Define plot configurations: (row, emg_column, adjusted_emg_column, muscle_name)
        plot_configs = [
            (
                0,
                "bflh_emg",
                "bflh_adjusted_emg",
                "M. Biceps Femoris Long Head",
            ),
            (
                1,
                "semimem_emg",
                "semiten_adjusted_emg",
                "M. Semitendinosus",
            ),
            (
                2,
                "semimem_emg",
                "semimem_adjusted_emg",
                "M. Semimembranosus",
            ),
        ]

        subject_id_int = int(subject_id)

        # Filter master_df for this subject
        subject_df = self.master_df[self.master_df["subject_id"] == subject_id_int]

        if subject_df.empty:
            raise Exception(
                f"Error: No data found for subject {subject_id} in master results data.",
            )

        # Get available trials for this subject
        subject_trials = subject_df["trial"].unique()

        # Determine which baseline trial to use
        if "Run_baselineA1" in subject_trials:
            pre_trial = "Run_baselineA1"
        elif "Run_baselineB1" in subject_trials:
            pre_trial = "Run_baselineB1"
        else:
            raise Exception(f"Error: No baseline trial found for subject {subject_id}")

        post_trial = "RunL1"

        if post_trial not in subject_trials:
            raise Exception(
                f"Error: Post trial {post_trial} not found for subject {subject_id}",
            )

        # Filter data for pre and post trials
        pre_df = subject_df[subject_df["trial"] == pre_trial]
        post_df = subject_df[subject_df["trial"] == post_trial]

        # Calculate RMSE and R² for EMG vs Adjusted EMG
        analysis_utils = Analysis_utils()
        rmse_results = {}
        r_squared_results = {}

        for emg_col, adjusted_emg_col in [
            (
                "bflh_emg",
                "bflh_adjusted_emg",
            ),
            ("semiten_emg", "semiten_adjusted_emg"),
            ("semimem_emg", "semimem_adjusted_emg"),
        ]:
            muscle_key = adjusted_emg_col.replace("_adjusted_emg", "")

            # Pre trial metrics
            rmse_pre = analysis_utils.get_rmse(
                pre_df[emg_col].values,
                pre_df[adjusted_emg_col].values,
            )
            r_squared_pre = analysis_utils.get_r_squared(
                pre_df[emg_col].values,
                pre_df[adjusted_emg_col].values,
            )
            # Calculate range for pre trial
            combined_data_pre = np.concatenate(
                [pre_df[emg_col].to_numpy(), pre_df[adjusted_emg_col].to_numpy()],
            )
            range_pre = np.max(combined_data_pre) - np.min(combined_data_pre)
            rmse_pct_pre = (rmse_pre / range_pre * 100) if range_pre > 0 else 0

            # Post trial metrics
            rmse_post = analysis_utils.get_rmse(
                post_df[emg_col].values,
                post_df[adjusted_emg_col].values,
            )
            r_squared_post = analysis_utils.get_r_squared(
                post_df[emg_col].values,
                post_df[adjusted_emg_col].values,
            )
            # Calculate range for post trial
            combined_data_post = np.concatenate(
                [post_df[emg_col].to_numpy(), post_df[adjusted_emg_col].to_numpy()],
            )
            range_post = np.max(combined_data_post) - np.min(combined_data_post)
            rmse_pct_post = (rmse_post / range_post * 100) if range_post > 0 else 0

            rmse_results[muscle_key] = {"pre": rmse_pct_pre, "post": rmse_pct_post}
            r_squared_results[muscle_key] = {
                "pre": r_squared_pre,
                "post": r_squared_post,
            }

        fig, ax = plt.subplots(
            3,
            2,
            figsize=(16, 9),
            constrained_layout=True,
            sharex=True,
            sharey="row",
        )

        fig.suptitle(f"EMG vs Adjusted EMG for Subject {subject_id}")

        # Set column titles
        ax[0, 0].set_title("Pre")
        ax[0, 1].set_title("Post")

        # Set row labels
        ax[0, 0].set_ylabel("M. Biceps Femoris Long Head\n(Activation: 0-1)")
        ax[1, 0].set_ylabel("M. Semitendinosus\n(Activation: 0-1)")
        ax[2, 0].set_ylabel("M. Semimembranosus\n(Activation: 0-1)")

        # Loop through all plot configurations
        for row, emg_col, adjusted_emg_col, muscle_name in plot_configs:
            # Get muscle key for lookups
            muscle_key = adjusted_emg_col.replace("_adjusted_emg", "")

            # Get RMSE and R² for this muscle
            rmse_pre = rmse_results[muscle_key]["pre"]
            r_squared_pre = r_squared_results[muscle_key]["pre"]
            rmse_post = rmse_results[muscle_key]["post"]
            r_squared_post = r_squared_results[muscle_key]["post"]

            # PRE column (left)
            # Plot EMG data (solid line)
            ax[row, 0].plot(
                pre_df["index"],
                pre_df[emg_col],
                label="EMG",
                color="red",
                linestyle="-",
            )

            # Plot Adjusted EMG data (dashed line)
            ax[row, 0].plot(
                pre_df["index"],
                pre_df[adjusted_emg_col],
                label="Adjusted EMG",
                color="blue",
            )

            # Add RMSE and R² text
            textstr_pre = f"RMSE: {rmse_pre:.1f}%\n$R^2$: {r_squared_pre:.3f}"
            ax[row, 0].text(
                0.95,
                0.95,
                textstr_pre,
                transform=ax[row, 0].transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
            )

            ax[row, 0].grid()

            # POST column (right)
            # Plot EMG data (solid line)
            ax[row, 1].plot(
                post_df["index"],
                post_df[emg_col],
                label="EMG",
                color="red",
                linestyle="-",
            )

            # Plot Adjusted EMG data
            ax[row, 1].plot(
                post_df["index"],
                post_df[adjusted_emg_col],
                label="Adjusted EMG",
                color="blue",
            )

            # Add RMSE and R² text
            textstr_post = f"RMSE: {rmse_post:.1f}%\n$R^2$: {r_squared_post:.3f}"
            ax[row, 1].text(
                0.95,
                0.95,
                textstr_post,
                transform=ax[row, 1].transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
            )

            ax[row, 1].grid()

        # Set consistent y-axis limits for all rows
        for row in range(3):
            max_y = 1
            # Set the same y-limit for both columns (0 to max_y)
            ax[row, 0].set_ylim(0, max_y)
            ax[row, 1].set_ylim(0, max_y)

        # Add a single legend at the top
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.98),
        )

        fig.supxlabel("Sprint Cycle (%)")

        return fig, ax

    def plot_emg_adjusted_emg(self):
        # Define plot configurations: (row, emg_column, adjusted_emg_column, muscle_name)
        # Note: Raw EMG (emg.mot) only has BFLH and SEMIMEM, not SEMITEN
        # For semitendinosus, semiten_emg will be NaN, so only adjusted EMG line will show
        plot_configs = [
            (
                0,
                "bflh_emg",
                "bflh_adjusted_emg",
                "M. Biceps Femoris Long Head",
            ),
            (
                1,
                "semiten_emg",
                "semiten_adjusted_emg",
                "M. Semitendinosus",
            ),
            (
                2,
                "semimem_emg",
                "semimem_adjusted_emg",
                "M. Semimembranosus",
            ),
        ]

        pre_df = Analysis_utils().prep_pre_df()
        post_df = Analysis_utils().prep_post_df()

        # Consolidate pre trials to a single "Run_baseline" label
        pre_df["trial"] = "Run_baseline"

        # Note: Only bflh_emg and semimem_emg have raw EMG data from emg.mot
        # semiten_emg will be NaN (no raw EMG available for semitendinosus)
        cols_to_process = [
            "bflh_emg",
            "bflh_adjusted_emg",
            "semiten_emg",
            "semiten_adjusted_emg",
            "semimem_emg",
            "semimem_adjusted_emg",
        ]

        df_filtered = pd.concat([pre_df, post_df], ignore_index=True)

        # Calculate RMSE and R² per subject for EMG vs Adjusted EMG
        analysis_utils = Analysis_utils()
        subject_ids = df_filtered["subject_id"].unique()

        rmse_results = []
        r_squared_results = []

        for subject_id in subject_ids:
            subject_data = df_filtered[df_filtered["subject_id"] == subject_id]

            for trial in ["Run_baseline", "RunL1"]:
                trial_data = subject_data[subject_data["trial"] == trial]

                if len(trial_data) == 0:
                    continue

                for emg_col, adjusted_emg_col in [
                    (
                        "bflh_emg",
                        "bflh_adjusted_emg",
                    ),
                    ("semiten_emg", "semiten_adjusted_emg"),
                    ("semimem_emg", "semimem_adjusted_emg"),
                ]:
                    rmse = analysis_utils.get_rmse(
                        trial_data[emg_col].values,
                        trial_data[adjusted_emg_col].values,
                    )
                    r_squared = analysis_utils.get_r_squared(
                        trial_data[emg_col].values,
                        trial_data[adjusted_emg_col].values,
                    )

                    # Calculate range and RMSE percentage
                    combined_data = np.concatenate(
                        [
                            trial_data[emg_col].to_numpy(),
                            trial_data[adjusted_emg_col].to_numpy(),
                        ],
                    )
                    data_range = np.max(combined_data) - np.min(combined_data)
                    rmse_pct = (rmse / data_range * 100) if data_range > 0 else 0

                    # Use adjusted_emg_col to identify the muscle correctly
                    muscle_key = adjusted_emg_col.replace("_adjusted_emg", "")

                    rmse_results.append(
                        {
                            "subject_id": subject_id,
                            "trial": trial,
                            "muscle": muscle_key,
                            "rmse": rmse_pct,
                        },
                    )

                    r_squared_results.append(
                        {
                            "subject_id": subject_id,
                            "trial": trial,
                            "muscle": muscle_key,
                            "r_squared": r_squared,
                        },
                    )

        rmse_df = pd.DataFrame(rmse_results)
        r_squared_df = pd.DataFrame(r_squared_results)

        # Calculate mean RMSE and R² for each trial and muscle
        rmse_summary = rmse_df.groupby(["trial", "muscle"])["rmse"].mean()
        r_squared_summary = r_squared_df.groupby(["trial", "muscle"])["r_squared"].mean()

        df_mean = df_filtered.groupby(["trial", "index"])[cols_to_process].mean().reset_index()

        df_std = df_filtered.groupby(["trial", "index"])[cols_to_process].std().reset_index()

        df_count = df_filtered.groupby(["trial", "index"])[cols_to_process].count().reset_index()

        # Calculate 95% confidence interval
        df_ci = self.calculate_confidence_intervals(df_std, df_count, cols_to_process)

        fig, ax = plt.subplots(
            3,
            2,
            figsize=(16, 9),
            constrained_layout=True,
            sharex=True,
            sharey="row",
        )

        fig.suptitle("EMG vs Adjusted EMG for All Participants")

        # Set column titles
        ax[0, 0].set_title("Pre")
        ax[0, 1].set_title("Post")

        # Set row labels
        ax[0, 0].set_ylabel("M. Biceps Femoris Long Head\n(Activation: 0-1)")
        ax[1, 0].set_ylabel("M. Semitendinosus\n(Activation: 0-1)")
        ax[2, 0].set_ylabel("M. Semimembranosus\n(Activation: 0-1)")

        # Get data for plotting
        baseline = df_mean[df_mean["trial"] == "Run_baseline"]
        ci_baseline = df_ci[df_ci["trial"] == "Run_baseline"]
        post = df_mean[df_mean["trial"] == "RunL1"]
        ci_post = df_ci[df_ci["trial"] == "RunL1"]

        # Loop through all plot configurations
        for row, emg_col, adjusted_emg_col, muscle_name in plot_configs:
            # Get muscle key for lookups - use adjusted_emg_col to get the correct muscle
            muscle_key = adjusted_emg_col.replace("_adjusted_emg", "")

            # Get RMSE and R² for this muscle
            rmse_pre = rmse_summary.loc[("Run_baseline", muscle_key)]
            r_squared_pre = r_squared_summary.loc[("Run_baseline", muscle_key)]
            rmse_post = rmse_summary.loc[("RunL1", muscle_key)]
            r_squared_post = r_squared_summary.loc[("RunL1", muscle_key)]

            # PRE column (left)
            # Plot EMG data (solid line)
            ax[row, 0].plot(
                baseline["index"],
                baseline[emg_col],
                label="EMG",
                color="red",
                linestyle="-",
            )
            ax[row, 0].fill_between(
                baseline["index"],
                baseline[emg_col] - ci_baseline[emg_col],
                baseline[emg_col] + ci_baseline[emg_col],
                color="red",
                alpha=0.2,
            )

            # Plot Adjusted EMG data (dashed line)
            ax[row, 0].plot(
                baseline["index"],
                baseline[adjusted_emg_col],
                label="Adjusted EMG",
                color="blue",
            )
            ax[row, 0].fill_between(
                baseline["index"],
                baseline[adjusted_emg_col] - ci_baseline[adjusted_emg_col],
                baseline[adjusted_emg_col] + ci_baseline[adjusted_emg_col],
                color="blue",
                alpha=0.2,
            )

            # Add RMSE and R² text
            textstr_pre = f"RMSE: {rmse_pre:.1f}%\n$R^2$: {r_squared_pre:.3f}"
            ax[row, 0].text(
                0.95,
                0.95,
                textstr_pre,
                transform=ax[row, 0].transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
            )

            ax[row, 0].grid()

            # POST column (right)
            # Plot EMG data (solid line)
            ax[row, 1].plot(
                post["index"],
                post[emg_col],
                label="EMG",
                color="red",
                linestyle="-",
            )
            ax[row, 1].fill_between(
                post["index"],
                post[emg_col] - ci_post[emg_col],
                post[emg_col] + ci_post[emg_col],
                color="red",
                alpha=0.2,
            )

            # Plot Adjusted EMG data
            ax[row, 1].plot(
                post["index"],
                post[adjusted_emg_col],
                label="Adjusted EMG",
                color="blue",
            )
            ax[row, 1].fill_between(
                post["index"],
                post[adjusted_emg_col] - ci_post[adjusted_emg_col],
                post[adjusted_emg_col] + ci_post[adjusted_emg_col],
                color="blue",
                alpha=0.2,
            )

            # Add RMSE and R² text
            textstr_post = f"RMSE: {rmse_post:.1f}%\n$R^2$: {r_squared_post:.3f}"
            ax[row, 1].text(
                0.95,
                0.95,
                textstr_post,
                transform=ax[row, 1].transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
            )

            ax[row, 1].grid()

        # Set consistent y-axis limits for all rows
        for row in range(3):
            max_y = 1
            # Set the same y-limit for both columns (0 to max_y)
            ax[row, 0].set_ylim(0, max_y)
            ax[row, 1].set_ylim(0, max_y)

        # Add a single legend at the top
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.98),
        )

        fig.supxlabel("Sprint Cycle (%)")

        return fig, ax

    def plot_moments_torques_single(self, subject_id: str):
        """
        Plot Inverse Dynamics Moments vs Simulated Torques for a single subject comparing Pre and Post trials.

        Args:
            subject_id: Subject ID to plot

        Returns:
            fig, ax: Figure and axes objects

        """
        torques_df = Analysis_utils().create_torques_results_df()

        # Filter data for this subject (keep as string since torques_df uses string IDs)
        subject_df = torques_df[torques_df["subject_id"] == subject_id]

        if subject_df.empty:
            raise Exception(
                f"Error: No data found for subject {subject_id} in torques results data. "
                f"Available Participants: {sorted(torques_df['subject_id'].unique())}",
            )

        # Get available trials for this subject
        subject_trials = subject_df["trial"].unique()

        # Determine which baseline trial to use
        if "Run_baselineA1" in subject_trials:
            pre_trial = "Run_baselineA1"
        elif "Run_baselineB1" in subject_trials:
            pre_trial = "Run_baselineB1"
        else:
            raise Exception(f"Error: No baseline trial found for subject {subject_id}")

        post_trial = "RunL1"

        if post_trial not in subject_trials:
            raise Exception(
                f"Error: Post trial {post_trial} not found for subject {subject_id}",
            )

        # Filter data for pre and post trials
        pre_df = subject_df[subject_df["trial"] == pre_trial]
        post_df = subject_df[subject_df["trial"] == post_trial]

        plot_configs = [
            (
                0,
                "hip_flexion_id_moment",
                "hip_flexion_torque",
                "Hip Flexion",
            ),
            (
                1,
                "knee_angle_id_moment",
                "knee_angle_torque",
                "Knee Angle",
            ),
            (
                2,
                "ankle_angle_id_moment",
                "ankle_angle_torque",
                "Ankle Angle",
            ),
        ]

        # Calculate RMSE and R² per trial
        analysis_utils = Analysis_utils()
        rmse_results = {}
        r_squared_results = {}

        for row, id_col, torque_col, joint_name in plot_configs:
            joint_key = id_col.replace("_id_moment", "")

            # Pre trial metrics
            rmse_pre = analysis_utils.get_rmse(
                pre_df[id_col].values,
                pre_df[torque_col].values,
            )
            r_squared_pre = analysis_utils.get_r_squared(
                pre_df[id_col].values,
                pre_df[torque_col].values,
            )
            # Calculate range for pre trial
            combined_data_pre = np.concatenate(
                [pre_df[id_col].to_numpy(), pre_df[torque_col].to_numpy()],
            )
            range_pre = np.max(combined_data_pre) - np.min(combined_data_pre)
            rmse_pct_pre = (rmse_pre / range_pre * 100) if range_pre > 0 else 0

            # Post trial metrics
            rmse_post = analysis_utils.get_rmse(
                post_df[id_col].values,
                post_df[torque_col].values,
            )
            r_squared_post = analysis_utils.get_r_squared(
                post_df[id_col].values,
                post_df[torque_col].values,
            )
            # Calculate range for post trial
            combined_data_post = np.concatenate(
                [post_df[id_col].to_numpy(), post_df[torque_col].to_numpy()],
            )
            range_post = np.max(combined_data_post) - np.min(combined_data_post)
            rmse_pct_post = (rmse_post / range_post * 100) if range_post > 0 else 0

            rmse_results[joint_key] = {"pre": rmse_pct_pre, "post": rmse_pct_post}
            r_squared_results[joint_key] = {
                "pre": r_squared_pre,
                "post": r_squared_post,
            }

        fig, ax = plt.subplots(3, 2, figsize=(16, 9), constrained_layout=True)

        fig.suptitle(
            f"Inverse Dynamics Moments vs. Simulated Torques for Subject {subject_id}",
        )

        # Set column titles
        ax[0, 0].set_title("Pre")
        ax[0, 1].set_title("Post")

        # Set row labels
        ax[0, 0].set_ylabel("Hip Flexion Moment (Nm/kg)")
        ax[1, 0].set_ylabel("Knee Angle Moment (Nm/kg)")
        ax[2, 0].set_ylabel("Ankle Angle Moment (Nm/kg)")

        for row, id_col, torque_col, joint_name in plot_configs:
            joint_key = id_col.replace("_id_moment", "")

            rmse_pre = rmse_results[joint_key]["pre"]
            r_squared_pre = r_squared_results[joint_key]["pre"]
            rmse_post = rmse_results[joint_key]["post"]
            r_squared_post = r_squared_results[joint_key]["post"]

            # PRE column (left)
            ax[row, 0].plot(
                pre_df["index"],
                pre_df[id_col],
                label="Inverse Dynamics",
                color="blue",
                linestyle="-",
            )

            ax[row, 0].plot(
                pre_df["index"],
                pre_df[torque_col],
                label="Simulated Torque",
                color="orange",
            )

            textstr_pre = f"RMSE: {rmse_pre:.1f}%\n$R^2$: {r_squared_pre:.3f}"
            ax[row, 0].text(
                0.95,
                0.95,
                textstr_pre,
                transform=ax[row, 0].transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
            )

            ax[row, 0].grid()

            # POST column (right)
            ax[row, 1].plot(
                post_df["index"],
                post_df[id_col],
                label="Inverse Dynamics",
                color="blue",
                linestyle="-",
            )

            ax[row, 1].plot(
                post_df["index"],
                post_df[torque_col],
                label="Simulated Torque",
                color="orange",
            )

            # Add RMSE and R² text
            textstr_post = f"RMSE: {rmse_post:.1f}%\n$R^2$: {r_squared_post:.3f}"
            ax[row, 1].text(
                0.95,
                0.95,
                textstr_post,
                transform=ax[row, 1].transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
            )

            ax[row, 1].grid()

        # Add single legend at the top
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.98),
        )

        fig.supxlabel("Sprint Cycle (%)")

        return fig, ax

    def plot_moments_torques(self):
        analysis_utils = Analysis_utils()
        torques_df = analysis_utils.create_torques_results_df()
        spm_sig_idx = analysis_utils.get_spm_sig_index_moments_vs_torques()

        # Consolidate pre-trial data
        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        pre_df = torques_df[torques_df["trial"].isin(pre_trials)].copy()
        pre_df["trial"] = "Run_baseline"  # Standardize pre-trial name

        cols_to_process = [
            col for col in torques_df.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]

        pre_consolidated = pre_df.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()
        pre_consolidated["trial"] = "Run_baseline"

        post_df = torques_df[torques_df["trial"].isin(post_trials)].copy()

        torques_df = pd.concat([pre_consolidated, post_df], ignore_index=True)

        fig, ax = plt.subplots(3, 2, figsize=(16, 9), constrained_layout=True)

        # Rmse and R2 per subject
        analysis_utils = Analysis_utils()
        subject_ids = torques_df["subject_id"].unique()
        rmse_results = []
        r_squared_results = []

        for subject_id in subject_ids:
            subject_data = torques_df[torques_df["subject_id"] == subject_id]

            for trial in ["Run_baseline", "RunL1"]:
                trial_data = subject_data[subject_data["trial"] == trial]

                if len(trial_data) == 0:
                    continue

                for id_col, torque_col in [
                    ("hip_flexion_id_moment", "hip_flexion_torque"),
                    ("knee_angle_id_moment", "knee_angle_torque"),
                    ("ankle_angle_id_moment", "ankle_angle_torque"),
                ]:
                    rmse = analysis_utils.get_rmse(
                        trial_data[id_col].values,
                        trial_data[torque_col].values,
                    )
                    r_squared = analysis_utils.get_r_squared(
                        trial_data[id_col].values,
                        trial_data[torque_col].values,
                    )

                    # Calculate range and RMSE percentage
                    combined_data = np.concatenate(
                        [
                            trial_data[id_col].to_numpy(),
                            trial_data[torque_col].to_numpy(),
                        ],
                    )
                    data_range = np.max(combined_data) - np.min(combined_data)
                    rmse_pct = (rmse / data_range * 100) if data_range > 0 else 0

                    rmse_results.append(
                        {
                            "subject_id": subject_id,
                            "trial": trial,
                            "joint": id_col.replace("_id_moment", ""),
                            "rmse": rmse_pct,
                        },
                    )

                    r_squared_results.append(
                        {
                            "subject_id": subject_id,
                            "trial": trial,
                            "joint": id_col.replace("_id_moment", ""),
                            "r_squared": r_squared,
                        },
                    )

        rmse_df = pd.DataFrame(rmse_results)
        r_squared_df = pd.DataFrame(r_squared_results)

        rmse_summary = rmse_df.groupby(["trial", "joint"])["rmse"].mean()
        r_squared_summary = r_squared_df.groupby(["trial", "joint"])["r_squared"].mean()

        df_mean = torques_df.groupby(["trial", "index"])[cols_to_process].mean().reset_index()

        df_std = torques_df.groupby(["trial", "index"])[cols_to_process].std().reset_index()

        df_count = torques_df.groupby(["trial", "index"])[cols_to_process].count().reset_index()

        # Calculate 95% confidence interval
        df_ci = self.calculate_confidence_intervals(df_std, df_count, cols_to_process)

        fig.suptitle("Inverse Dynamics Moments vs. Simulated Torques")

        # Set column titles
        ax[0, 0].set_title("Pre")
        ax[0, 1].set_title("Post")

        # Set row labels
        ax[0, 0].set_ylabel("Hip Flexion Moment (Nm/kg)")
        ax[1, 0].set_ylabel("Knee Angle Moment (Nm/kg)")
        ax[2, 0].set_ylabel("Ankle Angle Moment (Nm/kg)")

        baseline = df_mean[df_mean["trial"] == "Run_baseline"]
        ci_baseline = df_ci[df_ci["trial"] == "Run_baseline"]
        post = df_mean[df_mean["trial"] == "RunL1"]
        ci_post = df_ci[df_ci["trial"] == "RunL1"]

        # Map plot configs to ID and torque columns for proper processing
        joint_mapping = {
            0: ("hip_flexion_id_moment", "hip_flexion_torque", "hip_flexion"),
            1: ("knee_angle_id_moment", "knee_angle_torque", "knee_angle"),
            2: ("ankle_angle_id_moment", "ankle_angle_torque", "ankle_angle"),
        }

        # Process each row (joint) - plot both ID and torque on both Pre and Post columns
        for row in range(3):
            id_col, torque_col, joint_key = joint_mapping[row]

            rmse_pre = rmse_summary.loc[("Run_baseline", joint_key)]
            r_squared_pre = r_squared_summary.loc[("Run_baseline", joint_key)]
            rmse_post = rmse_summary.loc[("RunL1", joint_key)]
            r_squared_post = r_squared_summary.loc[("RunL1", joint_key)]

            # PRE column (left) - plot both ID moment and torque
            ax[row, 0].plot(
                baseline["index"],
                baseline[id_col],
                label="Inverse Dynamics",
                color="blue",
                linestyle="-",
            )
            ax[row, 0].fill_between(
                baseline["index"],
                baseline[id_col] - ci_baseline[id_col],
                baseline[id_col] + ci_baseline[id_col],
                color="blue",
                alpha=0.2,
            )

            ax[row, 0].plot(
                baseline["index"],
                baseline[torque_col],
                label="Simulated Torque",
                color="orange",
            )
            ax[row, 0].fill_between(
                baseline["index"],
                baseline[torque_col] - ci_baseline[torque_col],
                baseline[torque_col] + ci_baseline[torque_col],
                color="orange",
                alpha=0.2,
            )

            textstr_pre = f"RMSE: {rmse_pre:.1f}%\n$R^2$: {r_squared_pre:.3f}"
            ax[row, 0].text(
                0.95,
                0.95,
                textstr_pre,
                transform=ax[row, 0].transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
            )

            # Add SPM significance shading comparing ID moment vs torque (Pre)
            sig_idx_pre = spm_sig_idx.get(f"{joint_key}_pre")
            if sig_idx_pre is not None and len(sig_idx_pre) > 0:
                ax[row, 0].fill_between(
                    range(len(sig_idx_pre)),
                    ax[row, 0].get_ylim()[0],
                    ax[row, 0].get_ylim()[1],
                    where=sig_idx_pre,
                    color="red",
                    alpha=0.1,
                )

            ax[row, 0].grid(True, alpha=0.3)

            # POST column (right) - plot both ID moment and torque
            ax[row, 1].plot(
                post["index"],
                post[id_col],
                label="Inverse Dynamics",
                color="blue",
                linestyle="-",
            )
            ax[row, 1].fill_between(
                post["index"],
                post[id_col] - ci_post[id_col],
                post[id_col] + ci_post[id_col],
                color="blue",
                alpha=0.2,
            )

            ax[row, 1].plot(
                post["index"],
                post[torque_col],
                label="Simulated Torque",
                color="orange",
            )
            ax[row, 1].fill_between(
                post["index"],
                post[torque_col] - ci_post[torque_col],
                post[torque_col] + ci_post[torque_col],
                color="orange",
                alpha=0.2,
            )

            # Add RMSE and R² text
            textstr_post = f"RMSE: {rmse_post:.1f}%\n$R^2$: {r_squared_post:.3f}"
            ax[row, 1].text(
                0.95,
                0.95,
                textstr_post,
                transform=ax[row, 1].transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
            )

            # Add SPM significance shading comparing ID moment vs torque (Post)
            sig_idx_post = spm_sig_idx.get(f"{joint_key}_post")
            if sig_idx_post is not None and len(sig_idx_post) > 0:
                ax[row, 1].fill_between(
                    range(len(sig_idx_post)),
                    ax[row, 1].get_ylim()[0],
                    ax[row, 1].get_ylim()[1],
                    where=sig_idx_post,
                    color="red",
                    alpha=0.1,
                )

            ax[row, 1].grid(True, alpha=0.3)

            # Add single legend at the top
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.98),
        )

        fig.supxlabel("Sprint Cycle (%)")

        return fig, ax

    def plot_muscle_parameters(self, interactive=False):
        """
        Plot boxplots for muscle synergy evaluation data with optional interactivity.
        Creates a 3x3 grid where:
        - Rows represent different muscles (BFLH, Semiten, Semimem)
        - Columns represent different metrics (Optimal Fibre Length Diff, Tendon Slack Length Diff, Strength Coefficient)

        Args:
            interactive: If True and plotly is available, return interactive plotly figure

        Returns:
            If interactive=True: plotly figure object or None
            Otherwise: (matplotlib fig, ax)

        """
        data = Analysis_utils().get_muscle_parameters()

        # If plotly requested and available, create interactive plot
        if interactive:
            return self._plot_muscle_parameters_interactive(data)

        fig, ax = plt.subplots(3, 3, figsize=(16, 9), constrained_layout=True)

        fig.suptitle("Muscle Parameters: Calibrated vs Uncalibrated")

        # Define muscles and their display names
        muscles = [
            ("bflh", "M. Biceps Femoris Long Head"),
            ("semiten", "M. Semitendinosus"),
            ("semimem", "M. Semimembranosus"),
        ]

        # Define metrics and their display names
        metrics = [
            ("optimal_fibre_length", "Optimal Fibre Length\nDifference (m)"),
            ("tendon_slack_length", "Tendon Slack Length\nDifference (m)"),
            ("strength_coefficient", "Strength Coefficient"),
        ]

        # Create boxplots
        for row_idx, (muscle_key, muscle_name) in enumerate(muscles):
            for col_idx, (metric_key, metric_label) in enumerate(metrics):
                # Calculate difference or use strength coefficient directly
                if metric_key == "strength_coefficient":
                    # Use strength coefficient directly (no difference calculation)
                    col_name = f"{muscle_key}_{metric_key}_calibrated"
                    plot_data = data[col_name].dropna()
                else:
                    # Calculate difference: calibrated - uncalibrated
                    uncal_col = f"{muscle_key}_{metric_key}_uncalibrated"
                    cal_col = f"{muscle_key}_{metric_key}_calibrated"
                    plot_data = (data[cal_col] - data[uncal_col]).dropna()

                # Create boxplot
                bp = ax[row_idx, col_idx].boxplot(
                    [plot_data],
                    positions=[1],
                    widths=0.6,
                    patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
                )

                # Color the box
                for patch in bp["boxes"]:
                    patch.set_facecolor("lightblue")
                    patch.set_alpha(0.7)

                # Add individual data points with jitter
                y_data = plot_data.values
                x_data = np.ones(len(y_data)) + np.random.normal(0, 0.04, len(y_data))
                ax[row_idx, col_idx].scatter(
                    x_data,
                    y_data,
                    alpha=0.6,
                    s=30,
                    color="darkblue",
                    zorder=3,
                )

                # Set labels
                if row_idx == 0:
                    ax[row_idx, col_idx].set_title(metric_label, fontsize=10)

                if col_idx == 0:
                    ax[row_idx, col_idx].set_ylabel(muscle_name, fontsize=9)

                # Remove x-axis ticks
                ax[row_idx, col_idx].set_xticks([])
                ax[row_idx, col_idx].grid(axis="y", alpha=0.3)

                # Add sample size
                n_samples = len(plot_data)
                ax[row_idx, col_idx].text(
                    0.05,
                    0.95,
                    f"n={n_samples}",
                    transform=ax[row_idx, col_idx].transAxes,
                    verticalalignment="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        return fig, ax

    def _plot_muscle_parameters_interactive(self, data):
        """Create interactive plotly version of muscle parameters plot."""
        # Define muscles and metrics
        muscles = [
            ("bflh", "M. Biceps Femoris Long Head"),
            ("semiten", "M. Semitendinosus"),
            ("semimem", "M. Semimembranosus"),
        ]

        metrics = [
            ("optimal_fibre_length", "Optimal Fibre Length Difference (m)"),
            ("tendon_slack_length", "Tendon Slack Length Difference (m)"),
            ("strength_coefficient", "Strength Coefficient"),
        ]

        # Create subplot with 3 rows x 3 columns
        fig = make_subplots(  # type: ignore
            rows=3,
            cols=3,
            subplot_titles=[metrics[i][1] for i in range(3)] * 3,
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        for row_idx, (muscle_key, muscle_name) in enumerate(muscles):
            for col_idx, (metric_key, metric_label) in enumerate(metrics):
                # Calculate difference or use strength coefficient directly
                if metric_key == "strength_coefficient":
                    col_name = f"{muscle_key}_{metric_key}_calibrated"
                    plot_data = data[col_name].dropna()
                else:
                    uncal_col = f"{muscle_key}_{metric_key}_uncalibrated"
                    cal_col = f"{muscle_key}_{metric_key}_calibrated"
                    plot_data = (data[cal_col] - data[uncal_col]).dropna()

                # Get subject IDs from subject_id column, falling back to index if not available
                subject_ids = (
                    data.loc[plot_data.index, "subject_id"].values
                    if "subject_id" in data.columns
                    else plot_data.index.values
                )

                # Add box plot - position at x=0 to align with scatter points
                fig.add_trace(
                    go.Box(  # type: ignore
                        x0=0,  # Center box at x=0
                        y=plot_data.values,
                        name=muscle_name if col_idx == 0 else "",
                        marker=dict(color="lightblue", opacity=0.7),
                        showlegend=False,
                        boxmean="sd",  # Show mean and std dev
                        hoverinfo="skip",
                        width=0.6,  # Control box width
                    ),
                    row=row_idx + 1,
                    col=col_idx + 1,
                )

                # Add individual scatter points with subject IDs - center around x=0
                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=np.random.normal(0, 0.08, len(plot_data)),  # Center at 0
                        y=plot_data.values,
                        mode="markers",
                        marker=dict(size=6, color="darkblue", opacity=0.6),
                        text=[f"Subject {sid}" for sid in subject_ids],
                        hovertemplate="<b>%{text}</b><br>Value: %{y:.4f}<extra></extra>",
                        showlegend=False,
                    ),
                    row=row_idx + 1,
                    col=col_idx + 1,
                )

                # Update axes
                if col_idx == 0:
                    fig.update_yaxes(
                        title_text=muscle_name,
                        row=row_idx + 1,
                        col=col_idx + 1,
                    )

                # Hide x-axis ticks
                fig.update_xaxes(showticklabels=False, row=row_idx + 1, col=col_idx + 1)

        # Update layout
        fig.update_layout(
            title_text="Muscle Parameters: Calibrated vs Uncalibrated (Interactive)",
            height=900,
            width=1600,
            hovermode="closest",
        )

        return fig

    def plot_norm_fiber_length_pre_post(self):
        """
        Plot normalized fiber length for all muscles comparing Pre vs Post trials.
        Creates a large grid where each subplot shows one muscle with two lines (pre vs post).
        X-axis: index (sprint cycle %)
        Y-axis: normalized fiber length
        """
        analysis_utils = Analysis_utils()
        norm_fiber_length_df = analysis_utils.get_norm_fiber_length_all_muscles_df()
        pre_df = analysis_utils.prep_pre_df(df=norm_fiber_length_df)
        post_df = analysis_utils.prep_post_df(df=norm_fiber_length_df)

        # Consolidate pre trials
        pre_df["trial"] = "Run_baseline"

        # Extract muscle columns dynamically (exclude metadata columns)
        exclude_cols = ["subject_id", "trial", "leg", "index", "time"]
        muscle_cols = [col for col in pre_df.columns if col not in exclude_cols]

        # Calculate mean and confidence intervals
        df_filtered = pd.concat([pre_df, post_df], ignore_index=True)

        df_mean = df_filtered.groupby(["trial", "index"])[muscle_cols].mean().reset_index()

        df_std = df_filtered.groupby(["trial", "index"])[muscle_cols].std().reset_index()

        df_count = df_filtered.groupby(["trial", "index"])[muscle_cols].count().reset_index()

        # Calculate 95% confidence interval
        df_ci = self.calculate_confidence_intervals(df_std, df_count, muscle_cols)

        # Determine grid size (aim for roughly square layout)
        n_muscles = len(muscle_cols)
        n_cols = int(np.ceil(np.sqrt(n_muscles)))
        n_rows = int(np.ceil(n_muscles / n_cols))

        # Create figure
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 4, n_rows * 3),
            constrained_layout=True,
            sharex=True,
        )

        fig.suptitle(
            "Normalized Fiber Length: Pre vs Post Across All Muscles",
            fontsize=16,
        )

        # Flatten axes array for easier iteration
        axes_flat = axes.flatten() if n_muscles > 1 else [axes]

        # Get pre and post data
        baseline = df_mean[df_mean["trial"] == "Run_baseline"]
        ci_baseline = df_ci[df_ci["trial"] == "Run_baseline"]
        post = df_mean[df_mean["trial"] == "RunL1"]
        ci_post = df_ci[df_ci["trial"] == "RunL1"]

        # Plot each muscle
        for idx, muscle in enumerate(muscle_cols):
            ax = axes_flat[idx]

            # Plot Pre (baseline) data
            ax.plot(
                baseline["index"],
                baseline[muscle],
                label="Pre",
                color="blue",
                linestyle="-",
                linewidth=2,
            )
            ax.fill_between(
                baseline["index"],
                baseline[muscle] - ci_baseline[muscle],
                baseline[muscle] + ci_baseline[muscle],
                color="blue",
                alpha=0.2,
            )

            # Plot Post data
            ax.plot(
                post["index"],
                post[muscle],
                label="Post",
                color="orange",
                linestyle="-",
                linewidth=2,
            )
            ax.fill_between(
                post["index"],
                post[muscle] - ci_post[muscle],
                post[muscle] + ci_post[muscle],
                color="orange",
                alpha=0.2,
            )

            # Format muscle name for title
            muscle_name = muscle.replace("_", " ").title()
            ax.set_title(muscle_name, fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_ylabel("Norm. Fiber Length", fontsize=8)

            # Add legend to first plot only
            if idx == 0:
                ax.legend(loc="upper right", fontsize=8)

        # Hide unused subplots
        for idx in range(n_muscles, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Common x-label
        fig.supxlabel("Sprint Cycle (%)", fontsize=12)

        return fig, axes

    def plot_max_model_force_comp(self, interactive=False):
        """
        Plot max model force comparison with optional interactivity.

        Args:
            interactive: If True and plotly is available, return interactive plotly figure

        Returns:
            If interactive=True: plotly figure object or None
            Otherwise: (matplotlib fig, ax)

        """
        data = Analysis_utils().get_muscle_parameters()

        # If plotly requested and available, create interactive plot
        if interactive:
            return self._plot_max_model_force_comp_interactive(data)

        fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

        fig.suptitle(
            "Max Model Force Comparison: Opensim vs. Uncalibrated vs. Calibrated",
        )

        ax[0].set_title("M. Biceps Femoris Long Head")
        ax[1].set_title("M. Semitendinosus")
        ax[2].set_title("M. Semimembranosus")
        ax[0].set_ylabel("Max Model Force (N)")

        plot_configs = [
            (
                ax[0],
                "bflh_max_isometric_force_opensim",
                "bflh_max_isometric_force_uncalibrated",
                "bflh_max_isometric_force_calibrated",
            ),
            (
                ax[1],
                "semiten_max_isometric_force_opensim",
                "semiten_max_isometric_force_uncalibrated",
                "semiten_max_isometric_force_calibrated",
            ),
            (
                ax[2],
                "semimem_max_isometric_force_opensim",
                "semimem_max_isometric_force_uncalibrated",
                "semimem_max_isometric_force_calibrated",
            ),
        ]

        for axis, osim_col, uncal_col, cal_col in plot_configs:
            for idx in range(len(data)):
                axis.plot(
                    [1, 2, 3],
                    [
                        data[osim_col].iloc[idx],
                        data[uncal_col].iloc[idx],
                        data[cal_col].iloc[idx],
                    ],
                    marker="o",
                    color="lightblue",
                    alpha=0.6,
                )

            # Calculate means for bar plots
            mean_osim = data[osim_col].mean()
            mean_uncal = data[uncal_col].mean()
            mean_cal = data[cal_col].mean()

            axis.bar([1], mean_osim, label="Opensim", color="blue", alpha=0.3)
            axis.bar([2], mean_uncal, label="Uncalibrated", color="gray", alpha=0.3)
            axis.bar([3], mean_cal, label="Calibrated", color="orange", alpha=0.3)

            axis.set_xticks([1, 2, 3])
            axis.set_xticklabels(["Opensim", "Uncalibrated", "Calibrated"])
            axis.grid(axis="y", alpha=0.3)
        return fig, ax

    def _plot_max_model_force_comp_interactive(self, data):
        """Create interactive plotly version of max model force comparison plot."""
        # Create subplot
        fig = make_subplots(  # type: ignore
            rows=1,
            cols=3,
            subplot_titles=(
                "M. Biceps Femoris Long Head",
                "M. Semitendinosus",
                "M. Semimembranosus",
            ),
        )

        plot_configs = [
            (
                1,
                "bflh_max_isometric_force_opensim",
                "bflh_max_isometric_force_uncalibrated",
                "bflh_max_isometric_force_calibrated",
            ),
            (
                2,
                "semiten_max_isometric_force_opensim",
                "semiten_max_isometric_force_uncalibrated",
                "semiten_max_isometric_force_calibrated",
            ),
            (
                3,
                "semimem_max_isometric_force_opensim",
                "semimem_max_isometric_force_uncalibrated",
                "semimem_max_isometric_force_calibrated",
            ),
        ]

        x_labels = ["Opensim", "Uncalibrated", "Calibrated"]

        for col_idx, osim_col, uncal_col, cal_col in plot_configs:
            # Calculate means for bar plots
            mean_osim = data[osim_col].mean()
            mean_uncal = data[uncal_col].mean()
            mean_cal = data[cal_col].mean()

            # Add mean bars
            fig.add_trace(
                go.Bar(  # type: ignore
                    x=x_labels,
                    y=[mean_osim, mean_uncal, mean_cal],
                    marker=dict(color=["blue", "gray", "orange"], opacity=0.3),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )

            # Add individual subject lines
            subject_ids = data["subject_id"].values if "subject_id" in data.columns else range(len(data))
            for idx in range(len(data)):
                subject_id = subject_ids[idx]
                y_values = [
                    data[osim_col].iloc[idx],
                    data[uncal_col].iloc[idx],
                    data[cal_col].iloc[idx],
                ]

                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=x_labels,
                        y=y_values,
                        mode="lines+markers",
                        line=dict(color="lightblue", width=1),
                        marker=dict(size=6, color="lightblue"),
                        opacity=0.6,
                        name=f"Subject {subject_id}",
                        legendgroup=f"subject_{idx}",
                        showlegend=False,
                        hovertemplate=f"Subject {subject_id}<br>%{{y:.1f}} N<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

            # Update axes
            fig.update_xaxes(ticktext=x_labels, tickvals=x_labels, row=1, col=col_idx)

        # Update layout
        fig.update_yaxes(title_text="Max Model Force (N)", row=1, col=1)
        fig.update_layout(
            title_text="Max Model Force Comparison: Opensim vs. Uncalibrated vs. Calibrated (Interactive)",
            height=500,
            width=1600,
            hovermode="closest",
        )

        return fig

    def plot_muscle_work(self, return_plotly=False):
        """
        Plot muscle work with individual subject lines.

        Args:
            return_plotly: If True and plotly is available, return interactive plotly figure for HTML

        Returns:
            If return_plotly=True: plotly figure object or None
            Otherwise: (matplotlib fig, ax)

        """
        analysis_utils = Analysis_utils()
        muscle_work_df = analysis_utils.get_muscle_work_df()

        # If plotly requested and available, create interactive plot
        if return_plotly:
            return self._plot_muscle_work_interactive(muscle_work_df)

        fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

        fig.suptitle("Muscle Work: Pre vs Post")

        ax[0].set_title("M. Biceps Femoris Long Head")
        ax[1].set_title("M. Semitendinosus")
        ax[2].set_title("M. Semimembranosus")
        ax[0].set_ylabel("Muscle Work (J/kg)")

        # Define plot configurations: (axis, muscle_key, muscle_name)
        plot_configs = [
            (ax[0], "bflh", "M. Biceps Femoris Long Head"),
            (ax[1], "semiten", "M. Semitendinosus"),
            (ax[2], "semimem", "M. Semimembranosus"),
        ]

        # Bar positions and width
        bar_positions = [1, 2, 3, 4]
        bar_width = 0.6

        for axis, muscle_key, muscle_name in plot_configs:
            # Extract the relevant columns
            pos_work_pre_col = f"{muscle_key}_positive_work_pre"
            pos_work_post_col = f"{muscle_key}_positive_work_post"
            neg_work_pre_col = f"{muscle_key}_negative_work_pre"
            neg_work_post_col = f"{muscle_key}_negative_work_post"

            # Calculate means
            mean_pos_pre = muscle_work_df[pos_work_pre_col].mean()
            mean_pos_post = muscle_work_df[pos_work_post_col].mean()
            mean_neg_pre = muscle_work_df[neg_work_pre_col].mean()
            mean_neg_post = muscle_work_df[neg_work_post_col].mean()

            means = [mean_pos_pre, mean_pos_post, mean_neg_pre, mean_neg_post]

            # Plot bars for means
            colors = ["blue", "orange", "blue", "orange"]
            alphas = [0.3, 0.3, 0.3, 0.3]

            for pos, mean_val, color, alpha in zip(
                bar_positions,
                means,
                colors,
                alphas,
            ):
                axis.bar(pos, mean_val, bar_width, color=color, alpha=alpha)

            # Plot individual subject values with lines connecting pre to post for each work type
            for idx in range(len(muscle_work_df)):
                # Line connecting Positive Pre to Positive Post
                pos_pre_val = muscle_work_df[pos_work_pre_col].iloc[idx]
                pos_post_val = muscle_work_df[pos_work_post_col].iloc[idx]
                axis.plot(
                    [bar_positions[0], bar_positions[1]],
                    [pos_pre_val, pos_post_val],
                    marker="o",
                    color="lightblue",
                    alpha=0.6,
                    linewidth=1,
                    markersize=4,
                )

                # Line connecting Negative Pre to Negative Post
                neg_pre_val = muscle_work_df[neg_work_pre_col].iloc[idx]
                neg_post_val = muscle_work_df[neg_work_post_col].iloc[idx]
                axis.plot(
                    [bar_positions[2], bar_positions[3]],
                    [neg_pre_val, neg_post_val],
                    marker="o",
                    color="lightblue",
                    alpha=0.6,
                    linewidth=1,
                    markersize=4,
                )

            # Set x-axis labels
            axis.set_xticks(bar_positions)
            axis.set_xticklabels(
                ["Pos Pre", "Pos Post", "Neg Pre", "Neg Post"],
                rotation=0,
                ha="center",
            )
            axis.grid(axis="y", alpha=0.3)

            # Add horizontal line at y=0 for reference
            axis.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        return fig, ax

    def _plot_muscle_work_interactive(self, muscle_work_df):
        """Create interactive plotly version of muscle work plot."""
        # Create subplot
        fig = make_subplots(  # type: ignore
            rows=1,
            cols=3,
            subplot_titles=(
                "M. Biceps Femoris Long Head",
                "M. Semitendinosus",
                "M. Semimembranosus",
            ),
        )

        # Define configurations
        muscles = [("bflh", 1), ("semiten", 2), ("semimem", 3)]

        bar_positions = [1, 2, 3, 4]
        bar_labels = ["Pos Pre", "Pos Post", "Neg Pre", "Neg Post"]

        for muscle_key, col_idx in muscles:
            pos_work_pre_col = f"{muscle_key}_positive_work_pre"
            pos_work_post_col = f"{muscle_key}_positive_work_post"
            neg_work_pre_col = f"{muscle_key}_negative_work_pre"
            neg_work_post_col = f"{muscle_key}_negative_work_post"

            # Calculate means
            mean_pos_pre = muscle_work_df[pos_work_pre_col].mean()
            mean_pos_post = muscle_work_df[pos_work_post_col].mean()
            mean_neg_pre = muscle_work_df[neg_work_pre_col].mean()
            mean_neg_post = muscle_work_df[neg_work_post_col].mean()

            means = [mean_pos_pre, mean_pos_post, mean_neg_pre, mean_neg_post]
            colors_bar = ["blue", "orange", "blue", "orange"]

            # Add mean bars
            for pos, mean_val, color in zip(bar_positions, means, colors_bar):
                fig.add_trace(
                    go.Bar(  # type: ignore
                        x=[pos],
                        y=[mean_val],
                        marker_color=color,
                        opacity=0.3,
                        width=0.6,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col_idx,
                )

            # Add individual subject lines
            for idx in range(len(muscle_work_df)):
                subject_id = (
                    muscle_work_df.index[idx]
                    if "subject_id" not in muscle_work_df.columns
                    else muscle_work_df["subject_id"].iloc[idx]
                )

                # Positive work line
                pos_pre_val = muscle_work_df[pos_work_pre_col].iloc[idx]
                pos_post_val = muscle_work_df[pos_work_post_col].iloc[idx]

                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=[bar_positions[0], bar_positions[1]],
                        y=[pos_pre_val, pos_post_val],
                        mode="lines+markers",
                        line=dict(color="lightblue", width=1),
                        marker=dict(size=4, color="lightblue"),
                        opacity=0.6,
                        name=f"Subject {subject_id}",
                        legendgroup=f"subject_{idx}",
                        showlegend=(col_idx == 1),  # Only show in legend once
                        hovertemplate=f"Subject {subject_id}<br>%{{y:.2f}} J/kg<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

                # Negative work line
                neg_pre_val = muscle_work_df[neg_work_pre_col].iloc[idx]
                neg_post_val = muscle_work_df[neg_work_post_col].iloc[idx]

                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=[bar_positions[2], bar_positions[3]],
                        y=[neg_pre_val, neg_post_val],
                        mode="lines+markers",
                        line=dict(color="lightblue", width=1),
                        marker=dict(size=4, color="lightblue"),
                        opacity=0.6,
                        name=f"Subject {subject_id}",
                        legendgroup=f"subject_{idx}",
                        showlegend=False,
                        hovertemplate=f"Subject {subject_id}<br>%{{y:.2f}} J/kg<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

        # Update layout
        fig.update_xaxes(tickvals=bar_positions, ticktext=bar_labels)
        fig.update_yaxes(title_text="Muscle Work (J/kg)", row=1, col=1)
        fig.update_layout(
            title_text="Muscle Work: Pre vs Post (Interactive)",
            height=500,
            width=1600,
            showlegend=True,
            hovermode="closest",
        )

        # Add horizontal line at y=0 for each subplot
        for col_idx in [1, 2, 3]:
            fig.add_hline(y=0, line_width=0.5, line_color="black", row=1, col=col_idx)  # type: ignore[arg-type]

        return fig

    def plot_power_overview(self, return_plotly=False):
        """
        Plot power overview with individual subject lines.

        Args:
            return_plotly: If True and plotly is available, return interactive plotly figure for HTML

        Returns:
            If return_plotly=True: plotly figure object or None
            Otherwise: (matplotlib fig, ax)

        """
        analysis_utils = Analysis_utils()

        pre_df = analysis_utils.prep_pre_df()
        post_df = analysis_utils.prep_post_df()

        # Consolidate pre trials to a single "Run_baseline" label
        pre_df["trial"] = "Run_baseline"

        df_filtered = pd.concat([pre_df, post_df], ignore_index=True)

        # If plotly requested and available, create interactive plot
        if return_plotly:
            return self._plot_power_overview_interactive(df_filtered)

        cols_to_process = [
            col for col in df_filtered.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]

        df_mean = df_filtered.groupby(["trial", "index"])[cols_to_process].mean().reset_index()

        df_std = df_filtered.groupby(["trial", "index"])[cols_to_process].std().reset_index()

        df_count = df_filtered.groupby(["trial", "index"])[cols_to_process].count().reset_index()

        # Calculate 95% confidence interval
        df_ci = self.calculate_confidence_intervals(df_std, df_count, cols_to_process)

        fig, ax = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)

        fig.suptitle(f"n={df_filtered['subject_id'].nunique()} Participants")

        ax[0, 0].set_title("M. Biceps Femoris Long Head")
        ax[0, 1].set_title("M. Semitendinosus")
        ax[0, 2].set_title("M. Semimembranosus")
        ax[0, 0].set_ylabel("Fiber Velocity (OptFL/s)")
        ax[1, 0].set_ylabel("Fiber Active Force (N)")
        ax[2, 0].set_ylabel("Muscle Power (W)")

        # Define plot configurations for rows 0 and 1 (line plots)
        plot_configs = [
            # Row 0: Fiber Velocity
            (
                0,
                0,
                "bflh_fiber_velocity",
                "M. Biceps Femoris Long Head",
            ),
            (0, 1, "semiten_fiber_velocity", "M. Semitendinosus"),
            (0, 2, "semimem_fiber_velocity", "M. Semimembranosus"),
            # Row 1: Fiber Active Force
            (
                1,
                0,
                "bflh_fiber_force",
                "M. Biceps Femoris Long Head",
            ),
            (1, 1, "semiten_fiber_force", "M. Semitendinosus"),
            (1, 2, "semimem_fiber_force", "M. Semimembranosus"),
        ]

        # Plot velocity and force as line plots with confidence intervals
        for row, col, column, label in plot_configs:
            self.plot_trial_data(
                ax[row, col],
                "Run_baseline",
                column,
                "",
                "blue",
                df_mean,
                df_ci,
            )

            self.plot_trial_data(
                ax[row, col],
                "RunL1",
                column,
                "",
                "orange",
                df_mean,
                df_ci,
            )

            ax[row, col].grid(alpha=0.3)

        # Row 2: Muscle Power (bar plots)
        # Power = velocity × force
        muscles = [
            ("bflh", 0),
            ("semiten", 1),
            ("semimem", 2),
        ]

        baseline_data = df_mean[df_mean["trial"] == "Run_baseline"]
        post_data = df_mean[df_mean["trial"] == "RunL1"]

        for muscle_name, col_idx in muscles:
            velocity_col = f"{muscle_name}_fiber_velocity"
            force_col = f"{muscle_name}_fiber_force"

            # Calculate power = velocity × force for each time point
            baseline_power = (baseline_data[velocity_col] * baseline_data[force_col]).to_numpy()
            post_power = (post_data[velocity_col] * post_data[force_col]).to_numpy()

            # Calculate mean power across sprint cycle
            mean_baseline_power = np.mean(baseline_power)
            mean_post_power = np.mean(post_power)

            # Calculate power for each subject to show individual variation
            subject_powers_pre = []
            subject_powers_post = []

            for subject_id in df_filtered["subject_id"].unique():
                subject_pre = df_filtered[
                    (df_filtered["subject_id"] == subject_id) & (df_filtered["trial"] == "Run_baseline")
                ]
                subject_post = df_filtered[
                    (df_filtered["subject_id"] == subject_id) & (df_filtered["trial"] == "RunL1")
                ]

                if len(subject_pre) > 0:
                    power_pre = subject_pre[velocity_col].values * subject_pre[force_col].values
                    subject_powers_pre.append(np.mean(power_pre))

                if len(subject_post) > 0:
                    power_post = subject_post[velocity_col].values * subject_post[force_col].values
                    subject_powers_post.append(np.mean(power_post))

            # Create bar plot
            bar_positions = [1, 2]
            bar_width = 0.6

            ax[2, col_idx].bar(
                1,
                mean_baseline_power,
                bar_width,
                color="blue",
                alpha=0.3,
                label="Pre",
            )
            ax[2, col_idx].bar(
                2,
                mean_post_power,
                bar_width,
                color="orange",
                alpha=0.3,
                label="Post",
            )

            # Plot individual subject values
            for pre_val, post_val in zip(subject_powers_pre, subject_powers_post):
                ax[2, col_idx].plot(
                    [1, 2],
                    [pre_val, post_val],
                    marker="o",
                    color="lightblue",
                    alpha=0.6,
                    linewidth=1,
                    markersize=4,
                )

            ax[2, col_idx].set_xticks(bar_positions)
            ax[2, col_idx].set_xticklabels(["Pre", "Post"])
            ax[2, col_idx].grid(axis="y", alpha=0.3)

        # Add legend to the right like master plot
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="blue", linewidth=2),
            Line2D([0], [0], color="orange", linewidth=2),
        ]
        fig.legend(custom_lines, ["Pre", "Post"], loc="upper right", ncol=2)

        fig.supxlabel("Sprint Cycle (%) / Condition")

        return fig, ax

    def _plot_power_overview_interactive(self, df_filtered):
        """Create interactive plotly version of power overview plot."""
        cols_to_process = [
            col for col in df_filtered.columns if col not in ["subject_id", "trial", "index", "time", "leg"]
        ]

        df_mean = df_filtered.groupby(["trial", "index"])[cols_to_process].mean().reset_index()

        # Create subplots
        fig = make_subplots(  # type: ignore
            rows=3,
            cols=3,
            subplot_titles=(
                "M. Biceps Femoris Long Head",
                "M. Semitendinosus",
                "M. Semimembranosus",
                "M. Biceps Femoris Long Head",
                "M. Semitendinosus",
                "M. Semimembranosus",
                "M. Biceps Femoris Long Head",
                "M. Semitendinosus",
                "M. Semimembranosus",
            ),
            row_heights=[0.33, 0.33, 0.34],
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        muscles = [("bflh", 0), ("semiten", 1), ("semimem", 2)]

        # Row 0: Fiber Velocity (line plots)
        baseline_data = df_mean[df_mean["trial"] == "Run_baseline"]
        post_data = df_mean[df_mean["trial"] == "RunL1"]

        for muscle_name, col_idx in muscles:
            velocity_col = f"{muscle_name}_fiber_velocity"

            # Add baseline line
            fig.add_trace(
                go.Scatter(  # type: ignore
                    x=baseline_data["index"],
                    y=baseline_data[velocity_col],
                    mode="lines",
                    name="Pre",
                    line=dict(color="blue", width=2),
                    legendgroup="pre",
                    showlegend=(col_idx == 0),
                    hovertemplate="Pre<br>%{y:.3f} m/s<extra></extra>",
                ),
                row=1,
                col=col_idx + 1,
            )

            # Add post line
            fig.add_trace(
                go.Scatter(  # type: ignore
                    x=post_data["index"],
                    y=post_data[velocity_col],
                    mode="lines",
                    name="Post",
                    line=dict(color="orange", width=2),
                    legendgroup="post",
                    showlegend=(col_idx == 0),
                    hovertemplate="Post<br>%{y:.3f} m/s<extra></extra>",
                ),
                row=1,
                col=col_idx + 1,
            )

        # Row 1: Fiber Active Force (line plots)
        for muscle_name, col_idx in muscles:
            force_col = f"{muscle_name}_fiber_force"

            fig.add_trace(
                go.Scatter(  # type: ignore
                    x=baseline_data["index"],
                    y=baseline_data[force_col],
                    mode="lines",
                    line=dict(color="blue", width=2),
                    legendgroup="pre",
                    showlegend=False,
                    hovertemplate="Pre<br>%{y:.1f} N<extra></extra>",
                ),
                row=2,
                col=col_idx + 1,
            )

            fig.add_trace(
                go.Scatter(  # type: ignore
                    x=post_data["index"],
                    y=post_data[force_col],
                    mode="lines",
                    line=dict(color="orange", width=2),
                    legendgroup="post",
                    showlegend=False,
                    hovertemplate="Post<br>%{y:.1f} N<extra></extra>",
                ),
                row=2,
                col=col_idx + 1,
            )

        # Row 2: Muscle Power (bar plots with individual lines)
        for muscle_name, col_idx in muscles:
            velocity_col = f"{muscle_name}_fiber_velocity"
            force_col = f"{muscle_name}_fiber_force"

            # Calculate mean power
            baseline_power = (baseline_data[velocity_col] * baseline_data[force_col]).to_numpy()
            post_power = (post_data[velocity_col] * post_data[force_col]).to_numpy()
            mean_baseline_power = np.mean(baseline_power)
            mean_post_power = np.mean(post_power)

            # Add mean bars
            fig.add_trace(
                go.Bar(  # type: ignore
                    x=[1],
                    y=[mean_baseline_power],
                    marker_color="blue",
                    opacity=0.3,
                    width=0.6,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=3,
                col=col_idx + 1,
            )

            fig.add_trace(
                go.Bar(  # type: ignore
                    x=[2],
                    y=[mean_post_power],
                    marker_color="orange",
                    opacity=0.3,
                    width=0.6,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=3,
                col=col_idx + 1,
            )

            # Calculate and plot individual subject powers
            for subject_id in df_filtered["subject_id"].unique():
                subject_pre = df_filtered[
                    (df_filtered["subject_id"] == subject_id) & (df_filtered["trial"] == "Run_baseline")
                ]
                subject_post = df_filtered[
                    (df_filtered["subject_id"] == subject_id) & (df_filtered["trial"] == "RunL1")
                ]

                if len(subject_pre) > 0 and len(subject_post) > 0:
                    power_pre = subject_pre[velocity_col].values * subject_pre[force_col].values
                    power_post = subject_post[velocity_col].values * subject_post[force_col].values
                    mean_power_pre = np.mean(power_pre)
                    mean_power_post = np.mean(power_post)

                    fig.add_trace(
                        go.Scatter(  # type: ignore
                            x=[1, 2],
                            y=[mean_power_pre, mean_power_post],
                            mode="lines+markers",
                            line=dict(color="lightblue", width=1),
                            marker=dict(size=4, color="lightblue"),
                            opacity=0.6,
                            name=f"Subject {subject_id}",
                            legendgroup=f"subject_{subject_id}",
                            showlegend=False,
                            hovertemplate=f"Subject {subject_id}<br>%{{y:.1f}} W<extra></extra>",
                        ),
                        row=3,
                        col=col_idx + 1,
                    )

        # Update axes labels
        fig.update_yaxes(title_text="Fiber Velocity (OptFL/s)", row=1, col=1)
        fig.update_yaxes(title_text="Fiber Active Force (N)", row=2, col=1)
        fig.update_yaxes(title_text="Muscle Power (W)", row=3, col=1)

        for col_idx in [1, 2, 3]:
            fig.update_xaxes(title_text="Sprint Cycle (%)", row=1, col=col_idx)
            fig.update_xaxes(title_text="Sprint Cycle (%)", row=2, col=col_idx)
            fig.update_xaxes(
                tickvals=[1, 2],
                ticktext=["Pre", "Post"],
                row=3,
                col=col_idx,
            )

        n_subjects = df_filtered["subject_id"].nunique()
        fig.update_layout(
            title_text=f"Power Overview (n={n_subjects} Participants) - Interactive",
            height=1200,
            width=1600,
            showlegend=True,
            hovermode="closest",
        )

        return fig

    def velocity_strain_scatter_plot(self):
        analysis_utils = Analysis_utils()
        velocity_df = analysis_utils.get_speed_df()
        norm_fiber_length_df = analysis_utils.get_peak_norm_fiber_length_df(
            muscles=["biceps_femoris_long_head", "semitendinosus", "semimembranosus"],
        )

        merged_df = pd.merge(velocity_df, norm_fiber_length_df, on="subject_id")

        fig, ax = plt.subplots(3, 2, figsize=(10, 12))
        plt.subplots_adjust(wspace=0.35, left=0.10, right=0.95, top=0.96)

        fig.suptitle(f"n={merged_df['subject_id'].nunique()} Participants")
        ax[2, 0].set_xlabel("Running Speed (m/s)")
        ax[2, 1].set_xlabel("Change in Running Speed (m/s)")
        ax[0, 0].set_ylabel("M. Biceps Femoris Long Head")
        ax[1, 0].set_ylabel("M. Semitendinosus")
        ax[2, 0].set_ylabel("M. Semimembranosus")

        fig.text(
            0.015,
            0.5,
            "Peak Norm Fiber Length",
            va="center",
            ha="center",
            rotation="vertical",
            fontsize=13,
        )
        fig.text(
            0.515,
            0.5,
            "Change in Peak Norm Fiber Length",
            va="center",
            ha="center",
            rotation="vertical",
            fontsize=13,
        )

        plot_configs = [
            (0, 0, "pre_speed_m_s", "biceps_femoris_long_head_peak_norm_fiber_length_pre"),
            (1, 0, "pre_speed_m_s", "semitendinosus_peak_norm_fiber_length_pre"),
            (2, 0, "pre_speed_m_s", "semimembranosus_peak_norm_fiber_length_pre"),
            (0, 1, "speed_change_m_s", "biceps_femoris_long_head_peak_norm_fiber_length_change"),
            (1, 1, "speed_change_m_s", "semitendinosus_peak_norm_fiber_length_change"),
            (2, 1, "speed_change_m_s", "semimembranosus_peak_norm_fiber_length_change"),
        ]

        for row, col, x_col, y_col in plot_configs:
            ax[row, col].scatter(
                merged_df[x_col],
                merged_df[y_col],
                color="#0173B2",
                alpha=0.7,
                s=50,
            )

            # Fit and plot regression line
            if len(merged_df) >= 2:
                m, b = np.polyfit(merged_df[x_col], merged_df[y_col], 1)
                x_vals = np.array(ax[row, col].get_xlim())
                y_vals = m * x_vals + b
                ax[row, col].plot(
                    x_vals,
                    y_vals,
                    color="#DE8F05",
                    linestyle="--",
                    linewidth=2,
                )

                # Calculate correlation coefficient and R²
                r = np.corrcoef(
                    np.asarray(merged_df[x_col]),
                    np.asarray(merged_df[y_col]),
                )[0, 1]
                r_squared = r**2

                # Add text with r and R²
                textstr = f"r = {r:.3f}\n$R^2$ = {r_squared:.3f}"
                ax[row, col].text(
                    0.05,
                    0.95,
                    textstr,
                    transform=ax[row, col].transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            ax[row, col].grid(alpha=0.3)

        return fig, ax

    def plot_muscle_group_activations(self, muscle_groups: list | None = None):
        """
        Plot pre and post activations for selected muscle groups.

        Args:
            muscle_groups: List of muscle group names. If None, defaults to hamstring muscles.
                          Available groups (anatomical): "Gluteus Maximus", "Gluteus Medius", "Gluteus Minimus",
                          "Adductor Magnus", "Biceps Femoris", "Semimembranosus", "Semitendinosus",
                          "Rectus Femoris", "Vasti", "Triceps Surae"
                          Available groups (functional): "hip_extensors", "hip_flexors", "knee_extensors",
                          "ankle_plantarflexors", "hip_abductors", "hip_adductors", "hamstrings"

        Returns:
            fig, ax: Figure and axes objects

        """
        # Get muscle group dictionary from Analysis_utils
        analysis_utils = Analysis_utils()
        muscle_groups_dict = analysis_utils.MUSCLE_GROUPS_BY_LEG

        # Map functional group names to anatomical group names
        functional_groups = {
            "hip_extensors": [
                "Gluteus Maximus",
                "Biceps Femoris",
                "Semimembranosus",
                "Semitendinosus",
            ],
            "hip_flexors": ["Rectus Femoris"],
            "knee_extensors": ["Rectus Femoris", "Vasti"],
            "knee_flexors": ["Biceps Femoris", "Semimembranosus", "Semitendinosus"],
            "ankle_plantarflexors": ["Triceps Surae"],
            "hip_abductors": ["Gluteus Medius", "Gluteus Minimus"],
            "hip_adductors": ["Adductor Magnus"],
            "hamstrings": ["Biceps Femoris", "Semimembranosus", "Semitendinosus"],
        }

        # Default to hamstrings if not specified
        if muscle_groups is None:
            muscle_groups = ["hamstrings"]

        # Expand functional groups to anatomical groups
        expanded_groups = []
        for group in muscle_groups:
            if group in functional_groups:
                expanded_groups.extend(functional_groups[group])
            elif group in muscle_groups_dict["R"]:
                expanded_groups.append(group)
            else:
                available = list(muscle_groups_dict["R"].keys()) + list(
                    functional_groups.keys(),
                )
                raise ValueError(
                    f"Muscle group '{group}' not found. Available groups: {available}",
                )

        # Remove duplicates while preserving order
        muscle_groups = list(dict.fromkeys(expanded_groups))

        # Prepare data
        master_df = self.master_df
        p_value = 0.05 / len(muscle_groups)  # Bonferroni correction
        spm_sig_idx = Analysis_utils().get_spm_sig_index(p=p_value)

        # Define pre and post trials
        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        # Filter data for pre and post
        pre_df = master_df[master_df["trial"].isin(pre_trials)].copy()
        post_df = master_df[master_df["trial"].isin(post_trials)].copy()

        # Calculate layout: max 4 columns
        n_groups = len(muscle_groups)
        n_cols = min(4, n_groups)
        n_rows = int(np.ceil(n_groups / n_cols))

        # Create figure
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            squeeze=False,
        )
        fig.suptitle("Muscle Group Activations (Pre vs Post)", fontsize=16)

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        # Plot each muscle group
        for idx, group_name in enumerate(muscle_groups):
            ax = axes_flat[idx]

            # Get muscles in this group from one leg (just to get the base names)
            muscles_in_group = muscle_groups_dict["R"][group_name]

            # Convert to activation column names
            # The master_results columns use raw muscle names like "bflh_activation", not "bflh_activation"
            activation_cols = []
            for muscle in muscles_in_group:
                # Remove _r or _l suffix to get base muscle name (e.g., "bflh")
                base_muscle = muscle.replace("_r", "").replace("_l", "")
                # Create column name by adding "_activation" suffix
                activation_cols.append(f"{base_muscle}_activation")

            # Filter to only existing columns
            existing_cols = [col for col in activation_cols if col in master_df.columns]

            if not existing_cols:
                ax.text(
                    0.5,
                    0.5,
                    f"No activation data\nfor {group_name}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(group_name)
                continue

            # Calculate mean activation across muscles in the group for pre
            if not pre_df.empty and any(col in pre_df.columns for col in existing_cols):
                # For each subject/trial, calculate mean across muscles in group
                pre_df_copy = pre_df.copy()
                pre_df_copy["group_activation"] = pre_df_copy[existing_cols].mean(
                    axis=1,
                )

                # Now calculate mean and std across subjects at each index
                pre_group_mean = pre_df_copy.groupby("index")["group_activation"].mean().reset_index()
                pre_group_mean.columns = ["index", "activation"]

                # Calculate 95% CI across subjects
                pre_group_std = pre_df_copy.groupby("index")["group_activation"].std().reset_index()
                pre_group_count = pre_df_copy.groupby("index").size().reset_index(name="count")

                # Prepare DataFrames for CI calculation
                pre_std_df = pre_group_std.set_index("index")[["group_activation"]]
                pre_count_df = pre_group_count.set_index("index")[["count"]]
                pre_count_df.columns = ["group_activation"]

                pre_ci_df = self.calculate_confidence_intervals(
                    pre_std_df,
                    pre_count_df,
                    ["group_activation"],
                )
                pre_ci = pre_ci_df["group_activation"].values

                # Plot pre
                ax.plot(
                    pre_group_mean["index"],
                    pre_group_mean["activation"],
                    label="Pre",
                    color="blue",
                    linewidth=2,
                )
                ax.fill_between(
                    pre_group_mean["index"],
                    pre_group_mean["activation"] - pre_ci,
                    pre_group_mean["activation"] + pre_ci,
                    color="blue",
                    alpha=0.2,
                )

            # Calculate mean activation across muscles in the group for post
            if not post_df.empty and any(col in post_df.columns for col in existing_cols):
                # For each subject/trial, calculate mean across muscles in group
                post_df_copy = post_df.copy()
                post_df_copy["group_activation"] = post_df_copy[existing_cols].mean(
                    axis=1,
                )

                # Now calculate mean and std across subjects at each index
                post_group_mean = post_df_copy.groupby("index")["group_activation"].mean().reset_index()
                post_group_mean.columns = ["index", "activation"]

                # Calculate 95% CI across subjects
                post_group_std = post_df_copy.groupby("index")["group_activation"].std().reset_index()
                post_group_count = post_df_copy.groupby("index").size().reset_index(name="count")

                # Prepare DataFrames for CI calculation
                post_std_df = post_group_std.set_index("index")[["group_activation"]]
                post_count_df = post_group_count.set_index("index")[["count"]]
                post_count_df.columns = ["group_activation"]

                post_ci_df = self.calculate_confidence_intervals(
                    post_std_df,
                    post_count_df,
                    ["group_activation"],
                )
                post_ci = post_ci_df["group_activation"].values

                # Plot post
                ax.plot(
                    post_group_mean["index"],
                    post_group_mean["activation"],
                    label="Post",
                    color="orange",
                    linewidth=2,
                )
                ax.fill_between(
                    post_group_mean["index"],
                    post_group_mean["activation"] - post_ci,
                    post_group_mean["activation"] + post_ci,
                    color="orange",
                    alpha=0.2,
                )

            # Add significance shading if available
            # Check if any muscle in the group has significance data
            for col in existing_cols:
                sig_idx = spm_sig_idx.get(col)
                if sig_idx is not None:
                    ax.fill_between(
                        range(len(sig_idx)),
                        ax.get_ylim()[0],
                        ax.get_ylim()[1],
                        where=sig_idx,
                        color="red",
                        alpha=0.1,
                        label="Significant Difference",
                    )
                    break  # Only add shading once per subplot

            # Set labels and title
            ax.set_title(group_name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Sprint Cycle (%)")
            ax.set_ylabel("Activation (0-1)")
            ax.legend(loc="best")
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)

        # Hide unused subplots
        for idx in range(n_groups, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def plot_muscle_group_forces(self, muscle_groups: list | None = None):
        """
        Plot pre and post normalized forces for selected muscle groups.
        Forces are summed across muscles in the group and normalized by the summed max isometric force.

        Args:
            muscle_groups: List of muscle group names. If None, defaults to all muscle groups.
                          Available groups: "Gluteus Maximus", "Gluteus Medius", "Gluteus Minimus",
                          "Adductor Magnus", "Biceps Femoris", "Semimembranosus", "Semitendinosus",
                          "Rectus Femoris", "Vasti", "Triceps Surae"

        Returns:
            fig, ax: Figure and axes objects

        """
        # Get muscle group dictionary from Analysis_utils
        analysis_utils = Analysis_utils()
        muscle_groups_dict = analysis_utils.MUSCLE_GROUPS_BY_LEG
        file_utils = File_utils()
        subject_utils = Subject_utils()

        # Default to all muscle groups if not specified
        if muscle_groups is None:
            muscle_groups = list(muscle_groups_dict["R"].keys())

        # Validate if groups exist
        for group in muscle_groups:
            if group not in muscle_groups_dict["R"]:
                raise ValueError(
                    f"Muscle group '{group}' not found in available muscle groups: {list(muscle_groups_dict['R'].keys())}",
                )

        # Prepare data
        master_df = self.master_df

        # Define pre and post trials
        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        # Filter data for pre and post
        pre_df = master_df[master_df["trial"].isin(pre_trials)].copy()
        post_df = master_df[master_df["trial"].isin(post_trials)].copy()

        # Get all unique subjects
        all_subjects = master_df["subject_id"].unique()

        # Build dictionaries to store max isometric forces and legs for each subject
        max_forces_dict = {}
        subject_legs = {}

        print("Loading max isometric forces for all subjects...")
        for subject_id in all_subjects:
            subject_id_str = str(int(subject_id)).zfill(2)

            # Get and store measured leg
            leg = subject_utils.get_measured_leg(subject_id_str)
            subject_legs[subject_id] = leg

            session_path = file_utils.get_session_dir("0" + subject_id_str)
            subject_calibrated_file_path = session_path / "subjectCalibrated_hybrid.xml"

            if not subject_calibrated_file_path.exists():
                print(
                    f"Warning: Calibrated file not found for subject {subject_id_str}",
                )
                continue

            # Parse XML to get max isometric forces
            tree = ET.parse(subject_calibrated_file_path)
            root = tree.getroot()
            mtu_set = root.find("mtuSet")

            if mtu_set is None:
                print(f"Warning: mtuSet not found in XML for subject {subject_id_str}")
                continue

            subject_max_forces = {}
            for mtu in mtu_set.findall("mtu"):
                muscle_name = None
                max_model_iso_force = None
                strength_coeff = None

                for element in mtu:
                    if element.tag == "name":
                        muscle_name = element.text
                    if element.tag == "maxIsometricForce":
                        max_model_iso_force = element.text
                    if element.tag == "strengthCoefficient":
                        strength_coeff = element.text

                if muscle_name is not None and max_model_iso_force is not None and strength_coeff is not None:
                    max_isom_force = float(max_model_iso_force) * float(strength_coeff)
                    subject_max_forces[muscle_name] = max_isom_force

            max_forces_dict[subject_id] = subject_max_forces

        # Calculate layout: max 4 columns
        n_groups = len(muscle_groups)
        n_cols = min(4, n_groups)
        n_rows = int(np.ceil(n_groups / n_cols))

        # Create figure
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            squeeze=False,
        )
        fig.suptitle("Muscle Group Normalized Forces (Pre vs Post)", fontsize=16)

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        # Plot each muscle group
        for idx, group_name in enumerate(muscle_groups):
            ax = axes_flat[idx]

            # Get muscles in this group from one leg (just to get the base names)
            muscles_in_group = muscle_groups_dict["R"][group_name]

            # Convert to fiber force column names
            force_cols = []
            for muscle in muscles_in_group:
                # Remove _r or _l suffix to get base muscle name (e.g., "bflh")
                base_muscle = muscle.replace("_r", "").replace("_l", "")
                # Create column name by adding "_fiber_force" suffix
                force_cols.append(f"{base_muscle}_fiber_force")

            # Filter to only existing columns
            existing_cols = [col for col in force_cols if col in master_df.columns]

            if not existing_cols:
                ax.text(
                    0.5,
                    0.5,
                    f"No force data\nfor {group_name}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(group_name)
                continue

            # Calculate normalized force for pre
            if not pre_df.empty and any(col in pre_df.columns for col in existing_cols):
                pre_df_copy = pre_df.copy()

                # Vectorized calculation: sum(fiber_forces) / sum(max_isometric_forces)
                # Sum fiber forces across muscle columns
                pre_df_copy["sum_fiber_force"] = pre_df_copy[existing_cols].sum(axis=1)

                # Calculate sum of max forces for each subject
                sum_max_forces = []
                for _, row in pre_df_copy.iterrows():
                    subject_id = row["subject_id"]
                    if subject_id not in max_forces_dict or subject_id not in subject_legs:
                        sum_max_forces.append(np.nan)
                        continue

                    leg = subject_legs[subject_id]
                    leg_suffix = "_r" if leg == "R" else "_l"

                    sum_max = 0
                    for force_col in existing_cols:
                        muscle_name = force_col.replace("_fiber_force", "")
                        full_muscle_name = muscle_name + leg_suffix

                        if full_muscle_name in max_forces_dict[subject_id]:
                            sum_max += max_forces_dict[subject_id][full_muscle_name]

                    sum_max_forces.append(sum_max if sum_max > 0 else np.nan)

                pre_df_copy["sum_max_force"] = sum_max_forces
                pre_df_copy["group_norm_force"] = pre_df_copy["sum_fiber_force"] / pre_df_copy["sum_max_force"]

                # Now calculate mean and std across subjects at each index
                pre_group_mean = pre_df_copy.groupby("index")["group_norm_force"].mean().reset_index()
                pre_group_mean.columns = ["index", "force"]

                # Calculate 95% CI across subjects
                pre_group_std = pre_df_copy.groupby("index")["group_norm_force"].std().reset_index()
                pre_group_count = pre_df_copy.groupby("index").size().reset_index(name="count")

                # Prepare DataFrames for CI calculation
                pre_std_df = pre_group_std.set_index("index")[["group_norm_force"]]
                pre_count_df = pre_group_count.set_index("index")[["count"]]
                pre_count_df.columns = ["group_norm_force"]

                pre_ci_df = self.calculate_confidence_intervals(
                    pre_std_df,
                    pre_count_df,
                    ["group_norm_force"],
                )
                pre_ci = pre_ci_df["group_norm_force"].values

                # Plot pre
                ax.plot(
                    pre_group_mean["index"],
                    pre_group_mean["force"],
                    label="Pre",
                    color="blue",
                    linewidth=2,
                )
                ax.fill_between(
                    pre_group_mean["index"],
                    pre_group_mean["force"] - pre_ci,
                    pre_group_mean["force"] + pre_ci,
                    color="blue",
                    alpha=0.2,
                )

            # Calculate normalized force for post
            if not post_df.empty and any(col in post_df.columns for col in existing_cols):
                post_df_copy = post_df.copy()

                # Vectorized calculation: sum(fiber_forces) / sum(max_isometric_forces)
                # Sum fiber forces across muscle columns
                post_df_copy["sum_fiber_force"] = post_df_copy[existing_cols].sum(
                    axis=1,
                )

                # Calculate sum of max forces for each subject
                sum_max_forces = []
                for _, row in post_df_copy.iterrows():
                    subject_id = row["subject_id"]
                    if subject_id not in max_forces_dict or subject_id not in subject_legs:
                        sum_max_forces.append(np.nan)
                        continue

                    leg = subject_legs[subject_id]
                    leg_suffix = "_r" if leg == "R" else "_l"

                    sum_max = 0
                    for force_col in existing_cols:
                        muscle_name = force_col.replace("_fiber_force", "")
                        full_muscle_name = muscle_name + leg_suffix

                        if full_muscle_name in max_forces_dict[subject_id]:
                            sum_max += max_forces_dict[subject_id][full_muscle_name]

                    sum_max_forces.append(sum_max if sum_max > 0 else np.nan)

                post_df_copy["sum_max_force"] = sum_max_forces
                post_df_copy["group_norm_force"] = post_df_copy["sum_fiber_force"] / post_df_copy["sum_max_force"]

                # Now calculate mean and std across subjects at each index
                post_group_mean = post_df_copy.groupby("index")["group_norm_force"].mean().reset_index()
                post_group_mean.columns = ["index", "force"]

                # Calculate 95% CI across subjects
                post_group_std = post_df_copy.groupby("index")["group_norm_force"].std().reset_index()
                post_group_count = post_df_copy.groupby("index").size().reset_index(name="count")

                # Prepare DataFrames for CI calculation
                post_std_df = post_group_std.set_index("index")[["group_norm_force"]]
                post_count_df = post_group_count.set_index("index")[["count"]]
                post_count_df.columns = ["group_norm_force"]

                post_ci_df = self.calculate_confidence_intervals(
                    post_std_df,
                    post_count_df,
                    ["group_norm_force"],
                )
                post_ci = post_ci_df["group_norm_force"].values

                # Plot post
                ax.plot(
                    post_group_mean["index"],
                    post_group_mean["force"],
                    label="Post",
                    color="orange",
                    linewidth=2,
                )
                ax.fill_between(
                    post_group_mean["index"],
                    post_group_mean["force"] - post_ci,
                    post_group_mean["force"] + post_ci,
                    color="orange",
                    alpha=0.2,
                )

            # Add significance shading if available
            # Check if any muscle in the group has significance data
            p = 0.05 / len(muscle_groups)  # Bonferroni correction
            spm_sig_idx = Analysis_utils().get_spm_sig_index(p=p)
            for col in existing_cols:
                # Convert force column name to the corresponding norm_force column name in spm_sig_idx
                norm_force_col = col.replace("_fiber_force", "_norm_force")
                sig_idx = spm_sig_idx.get(norm_force_col)
                if sig_idx is not None:
                    ax.fill_between(
                        range(len(sig_idx)),
                        ax.get_ylim()[0],
                        ax.get_ylim()[1],
                        where=sig_idx,
                        color="red",
                        alpha=0.1,
                        label="Significant Difference",
                    )
                    break  # Only add shading once per subplot

            # Set labels and title
            ax.set_title(group_name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Sprint Cycle (%)")
            ax.set_ylabel("Normalized Force (F/Fmax)")
            ax.legend(loc="best")
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)

        # Hide unused subplots
        for idx in range(n_groups, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def _load_ceinms_mtu_set(self, modelPath) -> pd.DataFrame:
        """
        Load muscle-tendon unit (MTU) parameters from a CEINMS model XML file.

        Args:
            modelPath: Path to the CEINMS model XML file (str or Path)

        Returns:
            DataFrame with muscle names as index and parameters as columns

        """
        import xml.etree.ElementTree as ET

        root = ET.parse(str(modelPath)).getroot()
        mtu_set = root.find("mtuSet")
        if mtu_set is None:
            raise ValueError("mtuSet element not found in the model file")
        mtus = mtu_set.findall("mtu")

        # Collect all data first
        data = []
        for mtu in mtus:
            name_element = mtu.find("name")
            if name_element is None or name_element.text is None:
                continue

            row_data: dict[str, str | float] = {"name": name_element.text}
            for element in mtu.findall("*"):
                if element.tag != "name" and element.text is not None:
                    row_data[element.tag] = float(element.text)
            data.append(row_data)

        # Create DataFrame from list of dictionaries
        df = pd.DataFrame(data)
        df = df.set_index("name")

        return df

    def plot_ceinms_model_parameters(self, subject_id: str):
        """
        Plot optimized muscle parameters from a CEINMS model XML file.
        Creates a grid of bar plots showing different parameters for each muscle.

        Args:
            subject_id: Identifier for the subject whose CEINMS model parameters are to be plotted

        Returns:
            fig, axs: Figure and axes objects

        """
        file_utils = File_utils()

        mtuSet = self._load_ceinms_mtu_set(
            file_utils.get_ceinms_cal_model_file_path(subject_id),
        )
        muscle_names = mtuSet.index.tolist()
        parameters = mtuSet.columns.tolist()

        if len(parameters) == 10:
            n_cols = 5
        else:
            n_cols = 4

        n_rows = (len(parameters) + n_cols - 1) // n_cols
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(18, n_rows * 3),
            constrained_layout=True,
        )
        fig.suptitle("Optimised Muscle Parameters", fontsize=16)
        axs = axs.flatten()

        for i, param in enumerate(parameters):
            # convert to numeric
            mtuSet[param] = pd.to_numeric(mtuSet[param], errors="coerce")

            # plot bars left leg in red and right leg in blue
            colors = ["red" if name.endswith("_l") else "blue" for name in muscle_names]
            axs[i].bar(muscle_names, mtuSet[param], color=colors)
            axs[i].set_title(param)
            axs[i].set_xticks(range(len(muscle_names)))
            axs[i].set_xticklabels(muscle_names, rotation=90, ha="right")
            axs[i].grid(axis="y", alpha=0.3)

        # Hide unused subplots
        for i in range(len(parameters), len(axs)):
            axs[i].set_visible(False)

        # set legend left leg on top right leg on bottom
        from matplotlib.lines import Line2D

        red_patch = Line2D([0], [0], color="red", lw=4, label="Left Leg")
        blue_patch = Line2D([0], [0], color="blue", lw=4, label="Right Leg")
        fig.legend(handles=[red_patch, blue_patch], loc="upper right")

        return fig, axs

    def plot_compare_ceinms_models(self, subject_id: str):
        """
        Plot comparison between uncalibrated and calibrated CEINMS model parameters.
        Creates a grid of bar plots showing differences for certain parameters and
        calibrated values for others.

        Args:
            subject_id: Identifier for the subject whose CEINMS models are to be compared

        Returns:
            fig, axs: Figure and axes objects

        """
        file_utils = File_utils()

        # Load both models
        mtuSet_uncalibrated = self._load_ceinms_mtu_set(
            file_utils.get_ceinms_uncal_model_file_path(subject_id),
        )
        mtuSet_calibrated = self._load_ceinms_mtu_set(
            file_utils.get_ceinms_cal_model_file_path(subject_id),
        )

        # Only use parameters that exist in both models
        common_params = set(mtuSet_uncalibrated.columns) & set(
            mtuSet_calibrated.columns,
        )
        parameters = [col for col in mtuSet_calibrated.columns if col in common_params]

        # Only use muscles that exist in both models
        common_muscles = set(mtuSet_uncalibrated.index) & set(mtuSet_calibrated.index)
        muscle_names = [idx for idx in mtuSet_calibrated.index if idx in common_muscles]

        # Filter both dataframes to only include common muscles and parameters
        mtuSet_uncalibrated = mtuSet_uncalibrated.loc[muscle_names, parameters]
        mtuSet_calibrated = mtuSet_calibrated.loc[muscle_names, parameters]

        if len(parameters) == 10:
            n_cols = 5
        else:
            n_cols = 4

        n_rows = (len(parameters) + n_cols - 1) // n_cols
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(18, n_rows * 3),
            constrained_layout=True,
        )

        fig.suptitle("Compare: Uncalibrated vs Calibrated", fontsize=16)
        axs = axs.flatten()

        for i, param in enumerate(parameters):
            # convert to numeric
            mtuSet_uncalibrated[param] = pd.to_numeric(
                mtuSet_uncalibrated[param],
                errors="coerce",
            )
            mtuSet_calibrated[param] = pd.to_numeric(
                mtuSet_calibrated[param],
                errors="coerce",
            )

            # plot bars left leg in red and right leg in blue
            colors = ["red" if name.endswith("_l") else "blue" for name in muscle_names]

            if param in ["optimalFibreLength", "pennationAngle", "tendonSlackLength"]:
                diff = mtuSet_calibrated[param] - mtuSet_uncalibrated[param]
                axs[i].bar(muscle_names, diff, color=colors)
                axs[i].set_title(f"Difference in {param}\n(Calibrated - Uncalibrated)")
                axs[i].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
            else:
                axs[i].bar(muscle_names, mtuSet_calibrated[param], color=colors)
                axs[i].set_title(param)

            axs[i].set_xticks(range(len(muscle_names)))
            axs[i].set_xticklabels(muscle_names, rotation=90, ha="right")
            axs[i].grid(axis="y", alpha=0.3)

        # Hide unused subplots
        for i in range(len(parameters), len(axs)):
            axs[i].set_visible(False)

        # set legend left leg on top right leg on bottom
        from matplotlib.lines import Line2D

        red_patch = Line2D([0], [0], color="red", lw=4, label="Left Leg")
        blue_patch = Line2D([0], [0], color="blue", lw=4, label="Right Leg")
        fig.legend(handles=[red_patch, blue_patch], loc="upper right")

        return fig, axs

    def plot_moments_calibration_results(self, subject_id: str, trial_name: str):
        """
        Plot comparison between CEINMS torques and inverse dynamics moments.
        Loads CEINMS Torques.sto file and compares with ID moments from master_results.
        Creates a grid of subplots comparing each DOF.

        Args:
            subject_id: Subject ID
            trial_name: Trial name (e.g., 'Run_baselineA1', 'RunL1')

        Returns:
            fig, axes: Figure and axes objects

        """
        file_utils = File_utils()
        data_utils = Data_utils()

        # Load CEINMS torques from .sto file
        torques_file = file_utils.get_ceinms_torque_file_path(subject_id, trial_name)
        ceinms_df = data_utils.create_df(torques_file)

        # Load inverse dynamics moments from id.sto file
        id_file = file_utils.get_id_file_path(subject_id, trial_name)
        id_df = data_utils.create_df(id_file)

        if id_df.empty:
            raise ValueError(
                f"No ID data found for subject {subject_id}, trial {trial_name}",
            )

        # Detect which leg is in the CEINMS data (look for _l or _r suffix)
        ceinms_columns = ceinms_df.columns.tolist()
        leg_suffix = None
        if any("_l" in col for col in ceinms_columns):
            leg_suffix = "_l"
        elif any("_r" in col for col in ceinms_columns):
            leg_suffix = "_r"
        else:
            raise ValueError("Could not detect leg suffix (_l or _r) in CEINMS data")

        # Find common DOFs between CEINMS and ID
        # CEINMS columns: hip_flexion_l/r, knee_angle_l/r, ankle_angle_l/r
        # ID columns: hip_flexion_l/r_moment, knee_angle_l/r_moment, ankle_angle_l/r_moment
        dof_mapping = {
            f"hip_flexion{leg_suffix}": f"hip_flexion{leg_suffix}_moment",
            f"knee_angle{leg_suffix}": f"knee_angle{leg_suffix}_moment",
            f"ankle_angle{leg_suffix}": f"ankle_angle{leg_suffix}_moment",
        }

        # Filter to only DOFs that exist in both datasets
        available_dofs = []
        for ceinms_col, id_col in dof_mapping.items():
            if ceinms_col in ceinms_df.columns and id_col in id_df.columns:
                available_dofs.append((ceinms_col, id_col))

        if not available_dofs:
            raise ValueError(
                f"No common DOFs found between CEINMS and inverse dynamics data.\n"
                f"CEINMS columns: {ceinms_columns}\n"
                f"ID columns: {id_df.columns.tolist()}",
            )

        n_dofs = len(available_dofs)
        ncols = min(3, n_dofs)  # Max 3 columns
        nrows = int(np.ceil(n_dofs / ncols))

        # Create figure
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4 * nrows),
            constrained_layout=True,
        )

        if n_dofs == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        fig.suptitle(
            f"CEINMS vs Inverse Dynamics Moments\nSubject {subject_id}, Trial {trial_name}",
            fontsize=14,
        )

        # Plot each DOF
        for i, (ceinms_col, id_col) in enumerate(available_dofs):
            ax = axes[i]

            # Plot CEINMS torques
            ax.plot(
                ceinms_df["time"],
                ceinms_df[ceinms_col],
                "b-",
                linewidth=2,
                label="CEINMS",
            )

            # Plot ID moments
            ax.plot(
                id_df["time"],
                id_df[id_col],
                "r-",
                linewidth=2,
                label="Inverse Dynamics",
            )

            # Format
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Moment (Nm)")
            # Remove leg suffix for title display
            title = ceinms_col.replace("_l", "").replace("_r", "").replace("_", " ").title()
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        # Hide unused subplots
        for i in range(n_dofs, len(axes) if n_dofs > 1 else 1):
            if n_dofs > 1:
                axes[i].set_visible(False)

        return fig, axes

    def plot_ceinms_calibration_results(self, subject_id: str):
        """
        Plot CEINMS calibration results grouped by muscle groups.
        Reads all CSV files from the calibration output directory and plots
        muscle activations, excitations, or other calibration outputs.

        Args:
            subject_id: Subject ID

        Returns:
            List of (fig, axs, result_file) tuples for each CSV file found

        """
        import xml.etree.ElementTree as ET

        file_utils = File_utils()
        analysis_utils = Analysis_utils()

        # Get the calibration setup XML file path
        session_dir = file_utils.get_session_dir(subject_id)
        setup_xml_path = session_dir / "calibrationSetup_ceinms-nn_hybrid.xml"

        if not setup_xml_path.exists():
            raise FileNotFoundError(
                f"Calibration setup XML not found at {setup_xml_path}",
            )

        # Parse XML to get output directory
        setupXML = ET.parse(setup_xml_path).getroot()
        output_dir_element = setupXML.find("outputDirectory")
        if output_dir_element is None or output_dir_element.text is None:
            raise ValueError(
                f"outputDirectory element not found or empty in {setup_xml_path}",
            )
        calibration_output_dir_text = output_dir_element.text

        # Clean up path (remove .\ or ./ prefix if present)
        calibration_output_dir_text = calibration_output_dir_text.lstrip("./\\")
        calibration_output_dir = session_dir / calibration_output_dir_text

        if not calibration_output_dir.exists():
            raise FileNotFoundError(
                f"Calibration output directory not found: {calibration_output_dir}",
            )

        # Find all CSV files in calibration output directory
        result_files = list(calibration_output_dir.glob("*.csv"))

        if not result_files:
            raise FileNotFoundError(
                f"No CSV files found in calibration directory: {calibration_output_dir}",
            )

        # Get muscle groups (without leg suffix for calibration data)
        muscle_groups = analysis_utils.MUSCLE_GROUPS

        plots = []

        for result_file in result_files:
            # Load data
            data = pd.read_csv(result_file)
            muscle_names = [col for col in data.columns if col != "time"]

            # Check if this file has any muscle-level data by checking if any
            # muscles from MUSCLE_GROUPS exist (with or without leg suffixes)
            has_muscle_data = False
            for muscles in muscle_groups.values():
                for muscle in muscles:
                    if muscle in muscle_names or any(f"{muscle}{suffix}" in muscle_names for suffix in ["_r", "_l"]):
                        has_muscle_data = True
                        break
                if has_muscle_data:
                    break

            # Skip files that don't contain muscle-level data
            if not has_muscle_data:
                continue

            # Create subplots for each muscle group
            n_muscle_groups = len(muscle_groups)
            fig, axs = plt.subplots(
                n_muscle_groups,
                1,
                figsize=(12, n_muscle_groups * 3),
                constrained_layout=True,
            )

            fig.suptitle(
                f"Calibration Results: {result_file.stem}\nSubject {subject_id}",
                fontsize=14,
            )

            # Handle single subplot case
            if n_muscle_groups == 1:
                axs = [axs]

            for i, (muscle_group, muscles) in enumerate(muscle_groups.items()):
                ax = axs[i]

                # Plot each muscle in the group that exists in the data
                # Match with and without leg suffixes (_r, _l)
                muscles_plotted = 0
                for muscle in muscles:
                    # Try exact match first
                    if muscle in muscle_names:
                        ax.plot(data["time"], data[muscle], label=muscle, linewidth=1.5)
                        muscles_plotted += 1
                    else:
                        # Try with leg suffixes
                        for suffix in ["_r", "_l"]:
                            muscle_with_suffix = muscle + suffix
                            if muscle_with_suffix in muscle_names:
                                ax.plot(
                                    data["time"],
                                    data[muscle_with_suffix],
                                    label=muscle_with_suffix,
                                    linewidth=1.5,
                                )
                                muscles_plotted += 1

                ax.set_title(f"{muscle_group}", fontweight="bold")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)

                # Only add legend if muscles were plotted
                if muscles_plotted > 0:
                    ax.legend(loc="upper right", fontsize=8, ncol=2)
                else:
                    # Add text to indicate no matching data
                    ax.text(
                        0.5,
                        0.5,
                        "No matching muscles in data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        color="gray",
                    )

                # Only show x-label on bottom subplot
                if i < n_muscle_groups - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Time (s)")

            plots.append((fig, axs, result_file.name))

        return plots

    def plot_optimisation_results(self, subject_id: str, trial_name: str):
        """
        Plot CEINMS optimisation results for a specific trial.

        Loads all .sto files from the Optimised directory and creates plots
        grouped by muscle groups for each file.

        Args:
            subject_id: Subject identifier (e.g., "019")
            trial_name: Trial name (e.g., "RunL1")

        Returns:
            list: List of tuples (fig, axs, filename) for each result file

        """
        file_utils = File_utils()
        analysis_utils = Analysis_utils()
        data_utils = Data_utils()

        # Get optimised output directory
        session_dir = file_utils.get_session_dir(subject_id)
        trial_dir = session_dir / trial_name
        optimised_dir = trial_dir / "Optimised"

        if not optimised_dir.exists():
            raise FileNotFoundError(
                f"Optimised output directory not found: {optimised_dir}",
            )

        # Find all .sto files in optimised directory
        result_files = list(optimised_dir.glob("*.sto"))

        if not result_files:
            raise FileNotFoundError(
                f"No .sto files found in optimised directory: {optimised_dir}",
            )

        # Get muscle groups
        muscle_groups = analysis_utils.MUSCLE_GROUPS

        plots = []

        for result_file in result_files:
            # Skip files that are not muscle-level time-series data
            # Files like ObjectiveFunctionComponentsAndWeightings contain aggregate data
            if "ObjectiveFunction" in result_file.name or "Weightings" in result_file.name:
                continue

            # Load data
            data = data_utils.create_df(result_file)

            # Skip if data is empty or has very few columns
            if data.empty or len(data.columns) < 5:
                continue

            muscle_names = [col for col in data.columns if col != "time"]

            # Check if this file has any muscle-level data
            has_muscle_data = False
            for muscles in muscle_groups.values():
                for muscle in muscles:
                    if muscle in muscle_names or any(f"{muscle}{suffix}" in muscle_names for suffix in ["_r", "_l"]):
                        has_muscle_data = True
                        break
                if has_muscle_data:
                    break

            # Skip files that don't contain muscle-level data
            if not has_muscle_data:
                continue

            # Create subplots for each muscle group
            n_muscle_groups = len(muscle_groups)
            fig, axs = plt.subplots(
                n_muscle_groups,
                1,
                figsize=(12, n_muscle_groups * 3),
                constrained_layout=True,
            )

            fig.suptitle(
                f"Optimisation Results: {result_file.stem}\nSubject {subject_id} - {trial_name}",
                fontsize=14,
            )

            # Handle single subplot case
            if n_muscle_groups == 1:
                axs = [axs]

            for i, (muscle_group, muscles) in enumerate(muscle_groups.items()):
                ax = axs[i]

                # Plot each muscle in the group that exists in the data
                # Match with and without leg suffixes (_r, _l)
                muscles_plotted = 0
                for muscle in muscles:
                    # Try exact match first
                    if muscle in muscle_names:
                        ax.plot(data["time"], data[muscle], label=muscle, linewidth=1.5)
                        muscles_plotted += 1
                    else:
                        # Try with leg suffixes
                        for suffix in ["_r", "_l"]:
                            muscle_with_suffix = muscle + suffix
                            if muscle_with_suffix in muscle_names:
                                ax.plot(
                                    data["time"],
                                    data[muscle_with_suffix],
                                    label=muscle_with_suffix,
                                    linewidth=1.5,
                                )
                                muscles_plotted += 1

                ax.set_title(f"{muscle_group}", fontweight="bold")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)

                # Only add legend if muscles were plotted
                if muscles_plotted > 0:
                    ax.legend(loc="upper right", fontsize=8, ncol=2)
                else:
                    # Add text to indicate no matching data
                    ax.text(
                        0.5,
                        0.5,
                        "No matching muscles in data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        color="gray",
                    )

                # Only show x-label on bottom subplot
                if i < n_muscle_groups - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Time (s)")

            plots.append((fig, axs, result_file.name))

        return plots

    def plot_experimental_vs_ceinms(self, subject_id: str, trial_name: str):
        """
        Plot comparison between experimental data and CEINMS outputs.
        Creates two figures:
        1. EMG signals vs CEINMS activations (with R² and RMSE)
        2. Inverse dynamics moments vs CEINMS torques (with R² and RMSE)

        Args:
            subject_id: Subject identifier (e.g., "019")
            trial_name: Trial name (e.g., "RunL1")

        Returns:
            tuple: (emg_fig, emg_axes, moments_fig, moments_axes)

        """
        analysis_utils = Analysis_utils()
        data_utils = Data_utils()

        # Get file paths
        session_dir = self.file_utils.get_session_dir(subject_id)
        trial_dir = session_dir / trial_name

        emg_file = self.file_utils.get_emg_file_path(subject_id, trial_name)
        ceinms_activations_file = trial_dir / "Optimised" / "Activations.sto"
        excitation_generator_file = session_dir / "excitationGenerator.xml"
        id_file = self.file_utils.get_id_file_path(subject_id, trial_name)
        ceinms_torques_file = self.file_utils.get_ceinms_torque_file_path(
            subject_id,
            trial_name,
        )

        # Load data
        emg_data = data_utils.create_df(emg_file)
        ceinms_activations = data_utils.create_df(ceinms_activations_file)

        # Strip whitespace from EMG column names
        emg_data.columns = emg_data.columns.str.strip()

        # Check if data has enough rows
        if len(emg_data) < 2 or len(ceinms_activations) < 2:
            raise ValueError(
                f"Insufficient data: EMG has {len(emg_data)} rows, "
                f"CEINMS has {len(ceinms_activations)} rows. Need at least 2 rows each.",
            )

        # Time normalize both datasets to 0-100% BEFORE aligning
        # (OpenSim files are already aligned in time)
        emg_data = analysis_utils.time_normalise_df(emg_data)
        ceinms_activations = analysis_utils.time_normalise_df(ceinms_activations)

        # Parse excitation generator XML to get EMG to muscle mapping
        emg_mapping_xml = ET.parse(excitation_generator_file).getroot().find("mapping")
        muscle_mapping = {}

        if emg_mapping_xml is None:
            raise ValueError(
                f"No 'mapping' element found in {excitation_generator_file}",
            )

        for excitation in emg_mapping_xml.findall("excitation"):
            muscle_id = excitation.get("id")
            input_elems = excitation.findall("input")
            if len(input_elems) > 0:
                for input_elem in input_elems:
                    signal = input_elem.text
                    if signal not in muscle_mapping:
                        muscle_mapping[signal] = []
                    muscle_mapping[signal].append(muscle_id)

        # Filter out signals that don't exist in the EMG data
        available_muscle_mapping = {
            signal: muscles for signal, muscles in muscle_mapping.items() if signal in emg_data.columns
        }

        # ===== Plot 1: EMG vs CEINMS Activations =====
        n_signals = len(available_muscle_mapping)

        if n_signals == 0:
            raise ValueError(
                f"No EMG signals from excitation generator found in EMG data. "
                f"Expected signals: {list(muscle_mapping.keys())}, "
                f"Available columns: {list(emg_data.columns)}",
            )

        emg_fig, emg_axes = plt.subplots(
            n_signals,
            1,
            figsize=(12, n_signals * 3),
            constrained_layout=True,
        )

        if n_signals == 1:
            emg_axes = [emg_axes]

        emg_fig.suptitle(
            f"EMG vs CEINMS Activations\nSubject {subject_id} - {trial_name}",
            fontsize=14,
        )

        line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

        for i, (signal, muscles) in enumerate(available_muscle_mapping.items()):
            ax = emg_axes[i]

            # Plot EMG signal
            ax.plot(
                emg_data["time"],
                emg_data[signal],
                label=f"EMG: {signal}",
                color="blue",
                linewidth=2,
            )

            # Plot each muscle's CEINMS activation
            for j, muscle in enumerate(muscles):
                if muscle in ceinms_activations.columns:
                    # Calculate R² and RMSE
                    r2 = analysis_utils.get_r_squared(
                        emg_data[signal].values,
                        ceinms_activations[muscle].values,
                    )
                    rmse = analysis_utils.get_rmse(
                        emg_data[signal].values,
                        ceinms_activations[muscle].values,
                    )

                    # Calculate RMSE as percentage of signal range
                    signal_range = emg_data[signal].max() - emg_data[signal].min()
                    rmse_percent = (rmse / signal_range) * 100 if signal_range != 0 else 0

                    ax.plot(
                        ceinms_activations["time"],
                        ceinms_activations[muscle],
                        linestyle=line_styles[j % len(line_styles)],
                        label=f"{muscle} (R²={r2:.2f}, RMSE={rmse_percent:.0f}%)",
                        color="red",
                        linewidth=1.5,
                    )

            ax.set_ylabel("Activation (0-1)")
            ax.set_title(f"EMG Signal: {signal}", fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=9)

            if i < n_signals - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Sprint Cycle (%)")

        # ===== Plot 2: ID Moments vs CEINMS Torques =====
        id_data = data_utils.create_df(id_file)
        ceinms_torques = data_utils.create_df(ceinms_torques_file)

        # Check if data has enough rows
        if len(id_data) < 2 or len(ceinms_torques) < 2:
            raise ValueError(
                f"Insufficient torque data: ID has {len(id_data)} rows, "
                f"CEINMS has {len(ceinms_torques)} rows. Need at least 2 rows each.",
            )

        # Time normalize both datasets to 0-100%
        id_data = analysis_utils.time_normalise_df(id_data)
        ceinms_torques = analysis_utils.time_normalise_df(ceinms_torques)

        # Get DOF names (columns excluding time)
        dof_names = [col for col in ceinms_torques.columns if col != "time"]

        n_dofs = len(dof_names)
        moments_fig, moments_axes = plt.subplots(
            n_dofs,
            1,
            figsize=(12, n_dofs * 3),
            constrained_layout=True,
        )

        if n_dofs == 1:
            moments_axes = [moments_axes]

        moments_fig.suptitle(
            f"Inverse Dynamics vs CEINMS Torques\nSubject {subject_id} - {trial_name}",
            fontsize=14,
        )

        for i, dof in enumerate(dof_names):
            ax = moments_axes[i]

            # Find corresponding ID moment column
            id_col = dof + "_moment"

            if id_col in id_data.columns:
                # Calculate R² and RMSE
                r2 = analysis_utils.get_r_squared(
                    id_data[id_col].values,
                    ceinms_torques[dof].values,
                )
                rmse = analysis_utils.get_rmse(
                    id_data[id_col].values,
                    ceinms_torques[dof].values,
                )

                # Calculate RMSE as percentage of moment range
                moment_range = id_data[id_col].max() - id_data[id_col].min()
                rmse_percent = (rmse / moment_range) * 100 if moment_range != 0 else 0

                # Plot ID moment
                ax.plot(
                    id_data["time"],
                    id_data[id_col],
                    label="Inverse Dynamics",
                    color="blue",
                    linewidth=2,
                )

                # Plot CEINMS torque
                ax.plot(
                    ceinms_torques["time"],
                    ceinms_torques[dof],
                    label=f"CEINMS (R²={r2:.2f}, RMSE={rmse_percent:.0f}%)",
                    color="red",
                    linewidth=2,
                )

                # Format title (remove leg suffix and clean up)
                title = dof.replace("_l", "").replace("_r", "").replace("_", " ").title()
                ax.set_title(title, fontweight="bold")
                ax.set_ylabel("Moment (Nm)")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right")

                if i < n_dofs - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Sprint Cycle (%)")

        return emg_fig, emg_axes, moments_fig, moments_axes

    def plot_ceinms_muscle_forces(self, subject_id: str, trial_name: str):
        """
        Plot CEINMS muscle forces with activations overlay on secondary y-axis.
        Creates a grid of subplots with one subplot per muscle, showing:
        - Force (N) on primary y-axis
        - Activation (0-1) on secondary y-axis

        Args:
            subject_id: Subject identifier (e.g., "019")
            trial_name: Trial name (e.g., "RunL1")

        Returns:
            tuple: (fig, axes) - Figure and flattened array of axes

        """
        analysis_utils = Analysis_utils()
        data_utils = Data_utils()

        # Get file paths
        forces_file = self.file_utils.get_ceinms_muscle_forces_file_path(
            subject_id,
            trial_name,
        )

        # Load CEINMS outputs
        ceinms_forces = data_utils.create_df(forces_file)

        # Get related files in same directory
        activations_file = forces_file.parent / "Activations.sto"
        ceinms_activations = data_utils.create_df(activations_file)

        # Time normalize both datasets to 0-100%
        ceinms_forces = analysis_utils.time_normalise_df(ceinms_forces)
        ceinms_activations = analysis_utils.time_normalise_df(ceinms_activations)

        # Get muscle names
        muscle_names = [col for col in ceinms_forces.columns if col != "time"]

        n_muscles = len(muscle_names)
        if n_muscles == 0:
            raise ValueError("No muscle columns found in CEINMS forces file.")

        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(n_muscles)))
        n_rows = int(np.ceil(n_muscles / n_cols))

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 4, n_rows * 3),
            constrained_layout=True,
        )
        fig.suptitle(
            f"CEINMS Muscle Forces\\nSubject {subject_id} - {trial_name}",
            fontsize=16,
        )

        # Normalize axs to 1D array for easy indexing
        if isinstance(axs, np.ndarray):
            axs_flat = axs.flatten()
        else:
            axs_flat = np.array([axs])

        for i, muscle in enumerate(muscle_names):
            ax = axs_flat[i]

            # Plot force on primary y-axis
            ax.plot(
                ceinms_forces["time"],
                ceinms_forces[muscle],
                label="Force",
                color="green",
                linewidth=1.5,
            )

            # Add activations on secondary y-axis
            ax2 = ax.twinx()
            if muscle in ceinms_activations.columns:
                ax2.plot(
                    ceinms_activations["time"],
                    ceinms_activations[muscle],
                    label="Activation",
                    color="gray",
                    alpha=0.6,
                    linewidth=3,
                )
                ax2.set_ylabel("Activation (0-1)", fontsize=9)
                ax2.set_ylim([0, 1])

            # Format muscle name for title
            title = muscle.replace("_r", "").replace("_l", "")
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("Force (N)", fontsize=9)
            ax.grid(True, alpha=0.3)

            # Only the bottom row gets x labels
            row = i // n_cols
            if row < n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Sprint Cycle (%)")

            # Create simple legend
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            h_force = h1[0] if h1 else None
            h_act = h2[0] if h2 else None
            legend_handles = [h for h in (h_force, h_act) if h is not None]
            legend_labels = ["Force", "Activation"][: len(legend_handles)]
            if legend_handles:
                ax.legend(
                    legend_handles,
                    legend_labels,
                    fontsize="small",
                    loc="upper left",
                )

        # Hide any unused subplots
        for j in range(n_muscles, n_rows * n_cols):
            axs_flat[j].set_visible(False)

        return fig, axs_flat

    @staticmethod
    def save_figure(
        fig: Figure,
        base_dir: Path,
        filename: str,
        subfolder: str | None = None,
        dpi: int = 300,
        format: str = "png",
        bbox_inches: str = "tight",
        facecolor: str = "white",
    ) -> Path:
        """
        Save a figure with automatic organization into subdirectories.

        Organizes plots using hybrid structure:
        - overview/ : Cross-subject summary plots
        - per_subject/<plot_type>/ : Individual subject plots grouped by type
        - regression/ : Regression diagnostics
        - muscle_groups/ : Muscle-specific analyses

        Args:
            fig: Matplotlib figure to save
            base_dir: Base directory (e.g., data/plot_results/fatigue_analysis)
            filename: Filename (e.g., "master_plot_012.png")
            subfolder: Explicit subfolder path (overrides auto-detection)
            dpi: Resolution
            format: File format (png, pdf, svg)
            bbox_inches: Bounding box setting
            facecolor: Background color

        Returns:
            Path to saved file

        Example:
            >>> plots = Plot_utils()
            >>> fig, ax = plt.subplots()
            >>> # Auto-organize per_subject plots
            >>> plots.save_figure(fig, OUTPUT_DIR, "master_plot_012.png")
            >>> # Saves to: OUTPUT_DIR/per_subject/master_plots/012.png

        """
        if subfolder is None:
            # Extract subject ID if present (3 digits before .png)
            subject_id = None
            if filename.endswith(".png") and "_" in filename:
                # Get the part between last underscore and .png
                parts = filename[:-4].split("_")  # Remove .png and split by _
                last_part = parts[-1]
                # Check if it's exactly 3 digits
                if len(last_part) == 3 and last_part.isdigit():
                    subject_id = last_part

            if subject_id:
                # Determine plot type and organize into per_subject subdirectories
                if "master_plot" in filename.lower():
                    subfolder = "per_subject/master_plots"
                    filename = f"{subject_id}.png"
                elif "emg" in filename.lower() and "adjusted" not in filename.lower():
                    subfolder = "per_subject/emg"
                    filename = f"{subject_id}.png"
                elif "emg_adjusted" in filename.lower():
                    subfolder = "per_subject/emg_adjusted"
                    filename = f"{subject_id}.png"
                elif "moments_torques" in filename.lower():
                    subfolder = "per_subject/moments_torques"
                    filename = f"{subject_id}.png"
                elif "muscle_force" in filename.lower():
                    subfolder = "per_subject/muscle_forces"
                    filename = f"{subject_id}.png"
                else:
                    subfolder = "per_subject/other"
                    filename = f"{subject_id}.png"

            # Overview plots (all subjects, group comparisons)
            elif any(
                kw in filename.lower()
                for kw in [
                    "_all",
                    "overview",
                    "before_after",
                    "group_split",
                    "individual_groups",
                ]
            ):
                subfolder = "overview"

            # Regression diagnostics
            elif "regression" in filename.lower():
                if "delta_strain" in filename.lower():
                    subfolder = "regression/delta_strain"
                elif "strain" in filename.lower():
                    subfolder = "regression/strain"
                else:
                    subfolder = "regression"

            # Muscle group plots
            elif any(kw in filename.lower() for kw in ["muscle_group", "hip_muscle", "knee_muscle", "ankle_muscle"]):
                subfolder = "muscle_groups"

            # Default: save to root
            else:
                subfolder = ""

        # Create save directory
        if subfolder:
            save_dir = base_dir / subfolder
        else:
            save_dir = base_dir

        save_dir.mkdir(parents=True, exist_ok=True)

        # Ensure correct extension
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        save_path = save_dir / filename

        # Save figure
        fig.savefig(
            save_path,
            dpi=dpi,
            format=format,
            bbox_inches=bbox_inches,
            facecolor=facecolor,
        )

        return save_path

    @staticmethod
    def save_plotly_figure(
        fig_plotly,
        base_dir: Path,
        filename: str,
        subfolder: str | None = None,
    ) -> Path | None:
        """
        Save a plotly figure as interactive HTML.

        Args:
            fig_plotly: Plotly figure object
            base_dir: Base directory (e.g., data/plot_results/publication)
            filename: Filename (e.g., "figure_6_muscle_work_interactive.html")
            subfolder: Subfolder within base_dir (e.g., "interactive")

        Returns:
            Path to saved HTML file or None if plotly not available

        """
        if fig_plotly is None:
            return None

        # Create save directory
        if subfolder:
            save_dir = base_dir / subfolder
        else:
            save_dir = base_dir / "interactive"

        save_dir.mkdir(parents=True, exist_ok=True)

        # Ensure .html extension
        if not filename.endswith(".html"):
            filename = filename.replace(".png", ".html").replace(".pdf", ".html")
            if not filename.endswith(".html"):
                filename = f"{filename}.html"

        save_path = save_dir / filename

        # Save as standalone HTML with plotly
        fig_plotly.write_html(
            save_path,
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "responsive": True},
        )

        return save_path

    @staticmethod
    def generate_html_index(
        plot_dir: Path,
        output_name: str = "index.html",
        title: str | None = None,
        columns: int = 3,
        analysis_type: str | None = None,
    ) -> Path:
        """
        Generate an HTML index page for easy plot browsing.

        Creates an interactive HTML page with thumbnails organized by
        subdirectory with collapsible sections and lightbox viewing.

        Args:
            plot_dir: Directory containing plots
            output_name: HTML filename (ignored if analysis_type is provided)
            title: Page title (defaults to directory name)
            columns: Grid columns
            analysis_type: If provided, saves as {analysis_type}_index.html in plot_results/html

        Returns:
            Path to generated HTML file

        Example:
            >>> plots = Plot_utils()
            >>> plots.generate_html_index(
            ...     Path("data/plot_results/fatigue_analysis"),
            ...     title="Fatigue Analysis Plots",
            ...     analysis_type="fatigue_analysis"
            ... )

        """
        if title is None:
            title = f"Plot Index: {plot_dir.name.replace('_', ' ').title()}"

        # Collect all image files organized by subdirectory
        plot_structure = {}
        image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
        interactive_extensions = {".html"}

        def collect_plots(directory: Path, prefix: str = "") -> None:
            """Recursively collect plots from directories."""
            plots = []
            subdirs = []

            for item in sorted(directory.iterdir()):
                if item.is_file() and (
                    item.suffix.lower() in image_extensions or item.suffix.lower() in interactive_extensions
                ):
                    # Skip the index.html files themselves
                    if item.name != "index.html" and not item.name.endswith(
                        "_index.html",
                    ):
                        plots.append(item)
                elif item.is_dir():
                    subdirs.append(item)

            if plots:
                section_name = prefix or "Root"
                plot_structure[section_name] = plots

            # Recursively process subdirectories
            for subdir in subdirs:
                subdir_prefix = f"{prefix}/{subdir.name}" if prefix else subdir.name
                collect_plots(subdir, subdir_prefix)

        collect_plots(plot_dir)

        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2c3e50;
            margin: 0 0 10px 0;
            font-size: 2.5em;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
        }}
        .stats {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0 30px 0;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }}
        .stat-item {{
            text-align: center;
            padding: 10px;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .search-container {{
            margin: 20px 0;
            position: relative;
        }}
        .search-box {{
            width: 100%;
            padding: 15px 50px 15px 20px;
            font-size: 16px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            outline: none;
            transition: all 0.3s;
            background: #f8f9fa;
        }}
        .search-box:focus {{
            border-color: #667eea;
            background: white;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }}
        .search-icon {{
            position: absolute;
            right: 20px;
            top: 50%;
            transform: translateY(-50%);
            color: #667eea;
            font-size: 20px;
            pointer-events: none;
        }}
        .search-results {{
            margin-top: 15px;
            padding: 10px;
            background: #e7f3ff;
            border-radius: 6px;
            color: #0066cc;
            font-size: 14px;
            display: none;
        }}
        .no-results {{
            text-align: center;
            padding: 40px;
            color: #6c757d;
            font-size: 18px;
            display: none;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            cursor: pointer;
            user-select: none;
            background: linear-gradient(to right, #f8f9fa, white);
            padding: 15px 20px;
            border-radius: 8px;
            border-left: 5px solid #667eea;
            transition: all 0.3s;
            font-size: 1.5em;
        }}
        h2:hover {{
            background: linear-gradient(to right, #e9ecef, #f8f9fa);
            transform: translateX(5px);
        }}
        h2::before {{
            content: '▼ ';
            display: inline-block;
            transition: transform 0.3s;
            color: #667eea;
        }}
        h2.collapsed::before {{
            transform: rotate(-90deg);
        }}
        .section-count {{
            float: right;
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat({columns}, 1fr);
            gap: 25px;
            margin: 25px 0;
        }}
        .plot-item {{
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            background: #fafafa;
            transition: all 0.3s;
            cursor: pointer;
        }}
        .plot-item:hover {{
            transform: translateY(-8px);
            box-shadow: 0 12px 24px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
        }}
        .plot-item img {{
            width: 100%;
            height: auto;
            border-radius: 6px;
            display: block;
        }}
        .plot-name {{
            margin-top: 12px;
            font-size: 14px;
            color: #495057;
            text-align: center;
            font-weight: 500;
        }}
        .interactive-link {{
            display: inline-block;
            margin-top: 8px;
            padding: 6px 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.3s;
        }}
        .interactive-link:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        .interactive-badge {{
            display: inline-block;
            margin-left: 8px;
            padding: 2px 8px;
            background: #28a745;
            color: white;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
        }}
        .section-content {{
            overflow: hidden;
            transition: max-height 0.4s ease-out, opacity 0.3s;
        }}
        .section-content.collapsed {{
            max-height: 0 !important;
            opacity: 0;
        }}
        .lightbox {{
            display: none;
            position: fixed;
            z-index: 9999;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.95);
            cursor: pointer;
            animation: fadeIn 0.3s;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .lightbox img {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 95%;
            max-height: 95%;
            border-radius: 8px;
            box-shadow: 0 0 50px rgba(255,255,255,0.2);
        }}
        .lightbox-close {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        @media (max-width: 1200px) {{
            .plot-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        @media (max-width: 768px) {{
            .plot-grid {{ grid-template-columns: 1fr; }}
            h1 {{ font-size: 1.8em; }}
            .container {{ padding: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {title}</h1>
        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{len(plot_structure)}</div>
                <div class="stat-label">Sections</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{sum(len(plots) for plots in plot_structure.values())}</div>
                <div class="stat-label">Total Plots</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Generated</div>
                <div class="stat-label">{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</div>
            </div>
        </div>
        
        <div class="search-container">
            <input type="text" id="searchBox" class="search-box" placeholder="🔍 Search plots by name..." onkeyup="searchPlots()">
            <span class="search-icon">⌕</span>
            <div id="searchResults" class="search-results"></div>
        </div>
        
        <div id="noResults" class="no-results">
            <p>No plots found matching your search.</p>
        </div>
"""

        # Add each section
        for section_name, plots in plot_structure.items():
            section_id = section_name.lower().replace("/", "_").replace(" ", "_")
            display_name = section_name.replace("/", " / ").replace("_", " ").title()

            html_content += f"""
        <h2 onclick="toggleSection('{section_id}')" id="header_{section_id}">
            {display_name}
            <span class="section-count">{len(plots)} plots</span>
        </h2>
        <div class="section-content" id="section_{section_id}">
            <div class="plot-grid">
"""
            for plot_path in plots:
                rel_path = plot_path.relative_to(plot_dir)
                plot_name = plot_path.stem.replace("_", " ").title()

                # Check if this is an interactive HTML file
                is_interactive_html = plot_path.suffix.lower() == ".html"

                # Adjust path if saving to separate html directory
                if analysis_type:
                    # HTML is in data/plot_results/html/, images in data/plot_results/{analysis_type}/
                    img_path = f"../{plot_dir.name}/{rel_path}"
                else:
                    img_path = str(rel_path)

                # Check for interactive HTML version alongside image files
                interactive_path = None
                interactive_rel_path = None

                if not is_interactive_html:
                    # Look for interactive version in same directory
                    interactive_file = plot_path.parent / f"{plot_path.stem}_interactive.html"
                    if interactive_file.exists():
                        interactive_path = interactive_file
                        if analysis_type:
                            interactive_rel_path = f"../{plot_dir.name}/{interactive_file.relative_to(plot_dir)}"
                        else:
                            interactive_rel_path = str(
                                interactive_file.relative_to(plot_dir),
                            )

                    # Also look for interactive version in interactive/ subdirectory
                    interactive_dir = plot_dir / "interactive"
                    if not interactive_path and interactive_dir.exists():
                        # Try exact name with _interactive.html
                        interactive_file = interactive_dir / f"{plot_path.stem}_interactive.html"
                        if interactive_file.exists():
                            interactive_path = interactive_file
                            if analysis_type:
                                interactive_rel_path = f"../{plot_dir.name}/interactive/{interactive_file.name}"
                            else:
                                interactive_rel_path = f"interactive/{interactive_file.name}"

                # If this is a standalone interactive HTML file, display it as an iframe
                if is_interactive_html:
                    html_content += f"""
                <div class="plot-item">
                    <div class="interactive-container">
                        <iframe src="{img_path}" style="width: 100%; height: 400px; border: 1px solid #ddd; border-radius: 8px;" loading="lazy"></iframe>
                    </div>
                    <div class="plot-name">
                        {plot_name}
                        <span class="interactive-badge">🎯 Interactive</span>
                    </div>
                    <a href="{img_path}" target="_blank" class="interactive-link">
                        🔍 Open Full Screen
                    </a>
                </div>
"""
                else:
                    # Regular image file
                    html_content += f"""
                <div class="plot-item" onclick="openLightbox('{img_path}')">
                    <img src="{img_path}" alt="{plot_name}" loading="lazy">
                    <div class="plot-name">
                        {plot_name}
                        {'<span class="interactive-badge">🎯 Interactive</span>' if interactive_path else ""}
                    </div>
"""

                    if interactive_rel_path:
                        html_content += f"""
                    <a href="{interactive_rel_path}" target="_blank" class="interactive-link" onclick="event.stopPropagation();">
                        🔍 Open Interactive Version
                    </a>
"""

                    html_content += """
                </div>
"""

            html_content += """
            </div>
        </div>
"""

        # Close HTML and add JavaScript
        html_content += """
    </div>
    
    <div class="lightbox" id="lightbox" onclick="event.target === this && closeLightbox()">
        <span class="lightbox-close" onclick="closeLightbox(); event.stopPropagation();">&times;</span>
        <img id="lightbox-img" src="" alt="" onclick="nextImage(); event.stopPropagation();">
    </div>

    <script>
        let allImages = [];
        let currentImageIndex = 0;

        // Initialize sections as collapsed and collect all images
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('.section-content').forEach(section => {
                section.classList.add('collapsed');
                section.style.maxHeight = '0';
            });
            document.querySelectorAll('h2').forEach(header => {
                header.classList.add('collapsed');
            });

            // Collect all image sources in order
            document.querySelectorAll('.plot-item').forEach(item => {
                const onclick = item.getAttribute('onclick');
                if (onclick && onclick.includes('openLightbox')) {
                    const imgSrc = onclick.match(/'([^']+)'/)[1];
                    allImages.push(imgSrc);
                }
            });
        });

        function toggleSection(sectionId) {
            const section = document.getElementById('section_' + sectionId);
            const header = document.getElementById('header_' + sectionId);
            
            if (section.classList.contains('collapsed')) {
                section.style.maxHeight = section.scrollHeight + 'px';
                section.classList.remove('collapsed');
                header.classList.remove('collapsed');
            } else {
                section.style.maxHeight = '0';
                section.classList.add('collapsed');
                header.classList.add('collapsed');
            }
        }

        function openLightbox(imgSrc) {
            currentImageIndex = allImages.indexOf(imgSrc);
            const lightbox = document.getElementById('lightbox');
            document.getElementById('lightbox-img').src = imgSrc;
            lightbox.style.display = 'block';
            document.body.style.overflow = 'hidden';
        }

        function closeLightbox() {
            document.getElementById('lightbox').style.display = 'none';
            document.body.style.overflow = 'auto';
        }

        function nextImage() {
            if (allImages.length === 0) return;
            currentImageIndex = (currentImageIndex + 1) % allImages.length;
            document.getElementById('lightbox-img').src = allImages[currentImageIndex];
        }

        function prevImage() {
            if (allImages.length === 0) return;
            currentImageIndex = (currentImageIndex - 1 + allImages.length) % allImages.length;
            document.getElementById('lightbox-img').src = allImages[currentImageIndex];
        }

        // Keyboard navigation
        document.addEventListener('keydown', function(event) {
            const lightbox = document.getElementById('lightbox');
            if (lightbox.style.display === 'block') {
                if (event.key === 'Escape') {
                    closeLightbox();
                } else if (event.key === 'ArrowRight') {
                    nextImage();
                } else if (event.key === 'ArrowLeft') {
                    prevImage();
                }
            }
        });

        // Search functionality
        function searchPlots() {
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            const sections = document.querySelectorAll('.section-content');
            const headers = document.querySelectorAll('h2');
            const noResults = document.getElementById('noResults');
            const searchResults = document.getElementById('searchResults');
            
            let totalVisible = 0;
            let matchCount = 0;
            
            if (searchTerm === '') {
                // Reset to default collapsed state
                sections.forEach(section => {
                    section.classList.add('collapsed');
                    section.style.maxHeight = '0';
                });
                headers.forEach(header => {
                    header.classList.add('collapsed');
                    header.style.display = 'block';
                });
                document.querySelectorAll('.plot-item').forEach(item => {
                    item.style.display = 'block';
                });
                noResults.style.display = 'none';
                searchResults.style.display = 'none';
                return;
            }
            
            // Search through all plot items
            sections.forEach((section, index) => {
                const plotItems = section.querySelectorAll('.plot-item');
                let sectionHasMatch = false;
                
                plotItems.forEach(item => {
                    const plotName = item.querySelector('.plot-name').textContent.toLowerCase();
                    if (plotName.includes(searchTerm)) {
                        item.style.display = 'block';
                        sectionHasMatch = true;
                        matchCount++;
                    } else {
                        item.style.display = 'none';
                    }
                });
                
                // Show/hide section based on matches
                const header = headers[index];
                if (sectionHasMatch) {
                    section.classList.remove('collapsed');
                    section.style.maxHeight = section.scrollHeight + 'px';
                    header.classList.remove('collapsed');
                    header.style.display = 'block';
                    totalVisible++;
                } else {
                    section.classList.add('collapsed');
                    section.style.maxHeight = '0';
                    header.style.display = 'none';
                }
            });
            
            // Show/hide no results message
            if (matchCount === 0) {
                noResults.style.display = 'block';
                searchResults.style.display = 'none';
            } else {
                noResults.style.display = 'none';
                searchResults.style.display = 'block';
                searchResults.textContent = `Found ${matchCount} plot${matchCount !== 1 ? 's' : ''} matching "${searchTerm}"`;
            }
        }
    </script>
</body>
</html>
"""

        # Write HTML file
        if analysis_type:
            # Save in centralized html directory with analysis-specific name
            html_dir = Path("data/plot_results/html")
            html_dir.mkdir(parents=True, exist_ok=True)
            html_path = html_dir / f"{analysis_type}_index.html"
        else:
            # Use original behavior for backward compatibility
            html_path = plot_dir / output_name

        html_path.write_text(html_content)

        return html_path

    @staticmethod
    def generate_hub_index(
        html_dir: Path = Path("data/plot_results/html"),
        plot_results_dir: Path = Path("data/plot_results"),
    ) -> Path:
        """
        Generate a central hub page linking to all analysis index pages.

        Scans the html directory for analysis index files and creates
        a landing page with cards for each analysis type.

        Args:
            html_dir: Directory containing analysis HTML index files
            plot_results_dir: Base directory containing plot subdirectories

        Returns:
            Path to generated hub HTML file

        Example:
            >>> Plot_utils.generate_hub_index()

        """
        html_dir = Path(html_dir)
        plot_results_dir = Path(plot_results_dir)

        # Scan for analysis index files
        analysis_info = []

        if html_dir.exists():
            for html_file in sorted(html_dir.glob("*_index.html")):
                analysis_name = html_file.stem.replace("_index", "")

                # Map analysis names to actual directory names
                dir_mapping = {
                    "publication_figures": "publication",
                }
                actual_dir_name = dir_mapping.get(analysis_name, analysis_name)

                # Try to find corresponding plot directory
                plot_dir = plot_results_dir / actual_dir_name

                # Count plots if directory exists
                plot_count = 0
                subdir_count = 0
                if plot_dir.exists():
                    image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
                    plot_count = sum(
                        1 for f in plot_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_extensions
                    )
                    subdir_count = sum(1 for d in plot_dir.iterdir() if d.is_dir())

                # Create display name and description
                display_name = analysis_name.replace("_", " ").title()
                descriptions = {
                    "fatigue_analysis": "Core thesis analyses: muscle activation, forces, work, and coordination patterns during fatigue",
                    "ceinms_analysis": "CEINMS model calibration, validation, and muscle force estimation",
                    "opensim_validation": "OpenSim inverse kinematics and inverse dynamics validation",
                    "publication_figures": "High-quality figures formatted for publication",
                }
                description = descriptions.get(
                    analysis_name,
                    "Analysis plots and visualizations",
                )

                # Determine icon/emoji based on analysis type
                icons = {
                    "fatigue_analysis": "⚡",
                    "ceinms_analysis": "🔬",
                    "opensim_validation": "✓",
                    "publication_figures": "📊",
                }
                icon = icons.get(analysis_name, "📈")

                analysis_info.append(
                    {
                        "name": analysis_name,
                        "display_name": display_name,
                        "description": description,
                        "icon": icon,
                        "html_file": html_file.name,
                        "plot_count": plot_count,
                        "subdir_count": subdir_count,
                    },
                )

        # Generate hub HTML
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Hub - MSC Thesis</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        header {
            text-align: center;
            color: white;
            margin-bottom: 50px;
        }
        h1 {
            font-size: 3em;
            margin: 0 0 10px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            font-size: 1.2em;
            opacity: 0.95;
            margin: 10px 0;
        }
        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .stat-item {
            background: rgba(255,255,255,0.2);
            padding: 15px 25px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-top: 40px;
        }
        .analysis-card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
            text-decoration: none;
            color: inherit;
            display: block;
        }
        .analysis-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .card-icon {
            font-size: 3em;
            margin-right: 15px;
        }
        .card-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin: 0;
        }
        .card-description {
            color: #555;
            line-height: 1.6;
            margin: 15px 0;
            min-height: 60px;
        }
        .card-stats {
            display: flex;
            gap: 20px;
            padding-top: 15px;
            border-top: 2px solid #f0f0f0;
            margin-top: 15px;
        }
        .card-stat {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #667eea;
            font-weight: 500;
        }
        .card-stat-number {
            font-size: 1.3em;
            font-weight: bold;
        }
        .view-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-weight: 600;
            margin-top: 15px;
            width: 100%;
            cursor: pointer;
            transition: opacity 0.3s ease;
        }
        .view-button:hover {
            opacity: 0.9;
        }
        footer {
            text-align: center;
            color: white;
            margin-top: 60px;
            padding: 20px;
            opacity: 0.9;
        }
        .no-analyses {
            background: white;
            padding: 40px;
            border-radius: 12px;
            text-align: center;
            color: #666;
        }
        @media (max-width: 768px) {
            h1 { font-size: 2em; }
            .analysis-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 Analysis Hub</h1>
            <p class="subtitle">MSC Thesis: Fatigue Prediction in Gait Analysis</p>
            <div class="stats-bar">
"""

        # Calculate total stats
        total_plots = sum(a["plot_count"] for a in analysis_info)
        total_analyses = len(analysis_info)

        html_content += f"""
                <div class="stat-item">
                    <span class="stat-number">{total_analyses}</span>
                    <span class="stat-label">Analyses</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{total_plots}</span>
                    <span class="stat-label">Total Plots</span>
                </div>
            </div>
        </header>

        <div class="analysis-grid">
"""

        # Add cards for each analysis
        if analysis_info:
            for analysis in analysis_info:
                html_content += f"""
            <a href="{analysis["html_file"]}" class="analysis-card">
                <div class="card-header">
                    <div class="card-icon">{analysis["icon"]}</div>
                    <h2 class="card-title">{analysis["display_name"]}</h2>
                </div>
                <p class="card-description">{analysis["description"]}</p>
                <div class="card-stats">
                    <div class="card-stat">
                        <span class="card-stat-number">{analysis["plot_count"]}</span>
                        <span>plots</span>
                    </div>
"""
                if analysis["subdir_count"] > 0:
                    html_content += f"""
                    <div class="card-stat">
                        <span class="card-stat-number">{analysis["subdir_count"]}</span>
                        <span>categories</span>
                    </div>
"""
                html_content += """
                </div>
                <button class="view-button">View Plots →</button>
            </a>
"""
        else:
            html_content += """
            <div class="no-analyses">
                <h2>No Analyses Found</h2>
                <p>Run analysis scripts to generate plot indices.</p>
            </div>
"""

        html_content += (
            """
        </div>

        <footer>
            <p>Generated: """
            f"{Path.cwd().name}"
            """ | Last updated: """
            f"{html_dir.stat().st_mtime if html_dir.exists() else 'N/A'}"
            """</p>
        </footer>
    </div>

    <script>
        // Prevent default link behavior for button clicks
        document.querySelectorAll('.view-button').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                window.location.href = e.target.closest('.analysis-card').href;
            });
        });
    </script>
</body>
</html>
"""
        )

        # Write hub HTML
        html_dir.mkdir(parents=True, exist_ok=True)
        hub_path = html_dir / "index.html"
        hub_path.write_text(html_content)

        return hub_path

    def create_regression_scatter_plots(
        self,
        type: str = "strain",
        interactive: bool = True,
    ) -> list:
        """
        Create scatter plots showing relationships between all input variables and output variables.

        Args:
            type: Type of regression - "strain" (pre values) or "delta_strain" (changes).
            interactive: If True and plotly is available, create interactive plots (default: True).

        Returns:
            List of tuples containing (figure, muscle_name, variable_name) for saving.

        """
        base_vars = [
            "body_mass",
            "running_speed",
            "knee_peak_angle",
            "hip_peak_angle",
            "knee_peak_velocity",
            "hip_peak_velocity",
            "knee_range_of_motion",
            "hip_range_of_motion",
            "contact_time",
            "step_length",
        ]

        # Define input variables based on type
        if type == "strain":
            input_vars_dict = {"Body Mass (kg)": "body_mass"}
            for var in base_vars:
                if var != "body_mass":
                    # Create readable names
                    readable_name = var.replace("_", " ").title().replace("Of", "of")
                    input_vars_dict[f"{readable_name} (Pre)"] = f"{var}_pre"
        elif type == "delta_strain":
            input_vars_dict = {"Body Mass (kg)": "body_mass"}
            for var in base_vars:
                if var != "body_mass":
                    readable_name = var.replace("_", " ").title().replace("Of", "of")
                    input_vars_dict[f"{readable_name} (Change)"] = f"{var}_change"
        else:
            raise ValueError("type must be 'strain' or 'delta_strain'")

        # Define output variables
        muscles = {
            "BFLH": "bflh",
            "Semitendinosus": "semiten",
            "Semimembranosus": "semimem",
        }

        if type == "strain":
            output_suffix = "_peak_norm_fiber_length_pre"
            y_label_suffix = "Pre-Fatigue Peak Norm Fiber Length"
        else:
            output_suffix = "_peak_norm_fiber_length_change"
            y_label_suffix = "Change in Peak Norm Fiber Length"

        # Get regression data
        analysis_utils = Analysis_utils()
        regression_df = analysis_utils.get_regression_df()

        # Create scatter plots
        figures = []

        for muscle_readable, muscle_code in muscles.items():
            output_var = f"{muscle_code}{output_suffix}"

            if not interactive:
                # Create matplotlib figures
                for readable_name, input_var in input_vars_dict.items():
                    fig, ax = plt.subplots(figsize=(8, 6))

                    x_data = np.asarray(regression_df[input_var].values)
                    y_data = np.asarray(regression_df[output_var].values)

                    # Scatter plot
                    ax.scatter(x_data, y_data, alpha=0.6, s=60, label="Data")

                    # Calculate regression line
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_data.min(), x_data.max(), 100)
                    y_line = p(x_line)

                    # Calculate r²
                    r_squared = np.corrcoef(np.asarray(x_data), np.asarray(y_data))[0, 1] ** 2

                    # Plot regression line
                    ax.plot(
                        x_line,
                        y_line,
                        "r--",
                        alpha=0.8,
                        linewidth=2,
                        label=f"Fit: y={z[0]:.4f}x+{z[1]:.4f}",
                    )

                    ax.set_xlabel(readable_name, fontsize=11)
                    ax.set_ylabel(f"{muscle_readable} {y_label_suffix}", fontsize=11)
                    ax.set_title(
                        f"{muscle_readable}: {readable_name} vs {y_label_suffix}",
                        fontsize=12,
                    )
                    ax.grid(True, alpha=0.3)

                    # Add r² text box
                    ax.text(
                        0.05,
                        0.95,
                        f"R² = {r_squared:.4f}",
                        transform=ax.transAxes,
                        fontsize=11,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    )
                    ax.legend(loc="lower right", fontsize=9)

                    plt.tight_layout()

                    figures.append((fig, muscle_code, input_var))
            else:
                # Create interactive plotly figure with all input variables for this muscle
                n_vars = len(input_vars_dict)
                n_cols = 3
                n_rows = (n_vars + n_cols - 1) // n_cols

                fig = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    subplot_titles=list(input_vars_dict.keys()),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.10,
                )

                for idx, (readable_name, input_var) in enumerate(
                    input_vars_dict.items(),
                    1,
                ):
                    row = (idx - 1) // n_cols + 1
                    col = (idx - 1) % n_cols + 1

                    x_data = np.asarray(regression_df[input_var].values)
                    y_data = np.asarray(regression_df[output_var].values)
                    subject_ids = regression_df["subject_id"].values

                    # Add scatter points
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode="markers",
                            marker=dict(size=8, opacity=0.7),
                            text=[f"Subject {sid}" for sid in subject_ids],
                            hovertemplate="<b>%{text}</b><br>"
                            f"{readable_name}: "
                            "%{x:.2f}<br>"
                            f"{y_label_suffix}: "
                            "%{y:.4f}<extra></extra>",
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

                    # Calculate and add regression line
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_data.min(), x_data.max(), 100)
                    y_line = p(x_line)

                    # Calculate r²
                    r_squared = np.corrcoef(np.asarray(x_data), np.asarray(y_data))[0, 1] ** 2

                    # Add regression line
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line=dict(color="red", dash="dash", width=2),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

                    # Add r² annotation
                    # Plotly uses 'x' and 'y' for first subplot in each row, then 'x2', 'x3', etc.
                    subplot_num = (row - 1) * n_cols + col
                    xref = "x domain" if subplot_num == 1 else f"x{subplot_num} domain"
                    yref = "y domain" if subplot_num == 1 else f"y{subplot_num} domain"

                    fig.add_annotation(
                        x=0.05,
                        y=0.95,
                        xref=xref,
                        yref=yref,
                        text=f"R² = {r_squared:.4f}",
                        showarrow=False,
                        bgcolor="wheat",
                        opacity=0.8,
                        font=dict(size=10),
                    )

                    # Update axes labels
                    fig.update_xaxes(title_text=readable_name, row=row, col=col)
                    fig.update_yaxes(title_text=y_label_suffix, row=row, col=col)

                # Update layout
                fig.update_layout(
                    title_text=f"{muscle_readable}: Input Variables vs {y_label_suffix}",
                    title_font_size=16,
                    height=300 * n_rows,
                    showlegend=False,
                    hovermode="closest",
                )

                figures.append((fig, muscle_code, "all_variables"))

        return figures

    def create_output_correlation_plots(self, interactive: bool = True) -> list:
        """
        Create scatter plots showing relationships between output variables (muscle strains).

        Creates plots comparing:
        - Pre-fatigue strain values between muscles
        - Change in strain values between muscles

        Args:
            interactive: If True and plotly is available, create interactive plots (default: True).

        Returns:
            List of tuples containing (figure, plot_description) for saving.

        """
        # Get regression data
        analysis_utils = Analysis_utils()
        regression_df = analysis_utils.get_regression_df()

        muscles = {
            "BFLH": "bflh",
            "Semitendinosus": "semiten",
            "Semimembranosus": "semimem",
        }

        figures = []

        # Create plots for both pre-fatigue and change values
        for analysis_type in ["pre", "change"]:
            suffix = "_peak_norm_fiber_length_pre" if analysis_type == "pre" else "_peak_norm_fiber_length_change"
            title_suffix = "Pre-Fatigue" if analysis_type == "pre" else "Change in"

            if not interactive:
                # Create matplotlib figures for each pair of muscles
                muscle_pairs = [
                    ("BFLH", "Semitendinosus"),
                    ("BFLH", "Semimembranosus"),
                    ("Semitendinosus", "Semimembranosus"),
                ]

                for muscle1_name, muscle2_name in muscle_pairs:
                    muscle1_code = muscles[muscle1_name]
                    muscle2_code = muscles[muscle2_name]

                    var1 = f"{muscle1_code}{suffix}"
                    var2 = f"{muscle2_code}{suffix}"

                    fig, ax = plt.subplots(figsize=(8, 8))

                    x_data = np.asarray(regression_df[var1].values)
                    y_data = np.asarray(regression_df[var2].values)

                    ax.scatter(x_data, y_data, alpha=0.6, s=60)
                    ax.set_xlabel(
                        f"{muscle1_name} {title_suffix} Peak Norm Fiber Length",
                        fontsize=11,
                    )
                    ax.set_ylabel(
                        f"{muscle2_name} {title_suffix} Peak Norm Fiber Length",
                        fontsize=11,
                    )
                    ax.set_title(
                        f"{title_suffix} Peak Strain: {muscle1_name} vs {muscle2_name}",
                        fontsize=12,
                    )
                    ax.grid(True, alpha=0.3)

                    # Add diagonal line for reference
                    min_val = min(x_data.min(), y_data.min())
                    max_val = max(x_data.max(), y_data.max())
                    ax.plot(
                        [min_val, max_val],
                        [min_val, max_val],
                        "r--",
                        alpha=0.5,
                        label="x=y",
                    )
                    ax.legend()

                    # Calculate correlation
                    corr = np.corrcoef(np.asarray(x_data), np.asarray(y_data))[0, 1]
                    ax.text(
                        0.05,
                        0.95,
                        f"r = {corr:.3f}",
                        transform=ax.transAxes,
                        fontsize=11,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    )

                    plt.tight_layout()

                    figures.append(
                        (fig, f"{analysis_type}_{muscle1_code}_vs_{muscle2_code}"),
                    )
            else:
                # Create one interactive plotly figure with all three comparisons
                muscle_pairs = [
                    ("BFLH", "Semitendinosus"),
                    ("BFLH", "Semimembranosus"),
                    ("Semitendinosus", "Semimembranosus"),
                ]

                fig = make_subplots(
                    rows=1,
                    cols=3,
                    subplot_titles=[f"{m1} vs {m2}" for m1, m2 in muscle_pairs],
                    horizontal_spacing=0.12,
                )

                for idx, (muscle1_name, muscle2_name) in enumerate(muscle_pairs, 1):
                    muscle1_code = muscles[muscle1_name]
                    muscle2_code = muscles[muscle2_name]

                    var1 = f"{muscle1_code}{suffix}"
                    var2 = f"{muscle2_code}{suffix}"

                    x_data = np.asarray(regression_df[var1].values)
                    y_data = np.asarray(regression_df[var2].values)
                    subject_ids = regression_df["subject_id"].values

                    # Calculate correlation
                    corr = np.corrcoef(np.asarray(x_data), np.asarray(y_data))[0, 1]

                    # Add scatter points
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode="markers",
                            marker=dict(size=8, opacity=0.7),
                            text=[f"Subject {sid}" for sid in subject_ids],
                            hovertemplate="<b>%{text}</b><br>"
                            f"{muscle1_name}: "
                            "%{x:.4f}<br>"
                            f"{muscle2_name}: "
                            "%{y:.4f}<extra></extra>",
                            showlegend=False,
                            name=f"r={corr:.3f}",
                        ),
                        row=1,
                        col=idx,
                    )

                    # Add diagonal line
                    min_val = min(x_data.min(), y_data.min())
                    max_val = max(x_data.max(), y_data.max())
                    fig.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode="lines",
                            line=dict(color="red", dash="dash"),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=idx,
                    )

                    # Update axes
                    fig.update_xaxes(
                        title_text=f"{muscle1_name}<br>({title_suffix} Strain)",
                        row=1,
                        col=idx,
                    )
                    fig.update_yaxes(
                        title_text=f"{muscle2_name}<br>({title_suffix} Strain)",
                        row=1,
                        col=idx,
                    )

                    # Add correlation annotation
                    # Plotly uses 'x' and 'y' for first subplot, 'x2', 'x3' etc for others
                    xref = "x domain" if idx == 1 else f"x{idx} domain"
                    yref = "y domain" if idx == 1 else f"y{idx} domain"
                    fig.add_annotation(
                        x=0.05,
                        y=0.95,
                        xref=xref,
                        yref=yref,
                        text=f"r = {corr:.3f}",
                        showarrow=False,
                        bgcolor="wheat",
                        opacity=0.8,
                        font=dict(size=11),
                    )

                # Update layout
                fig.update_layout(
                    title_text=f"{title_suffix} Peak Strain: Muscle Comparisons",
                    title_font_size=16,
                    height=500,
                    width=1500,
                    showlegend=False,
                    hovermode="closest",
                )

                figures.append((fig, f"{analysis_type}_muscle_comparisons"))

        return figures

    def plot_peak_velocity_strain(
        self,
        pre_or_change,
        muscle,
        interactive: bool = True,
        responders_dict=None,
        non_responders_dict=None,
        speed_df=None,
        peak_norm_fiber_length_df=None,
    ):
        """
        Plot relationship between running speed and peak muscle strain.

        Args:
            pre_or_change: "pre" for baseline values or "change" for fatigue-induced changes
            muscle: Muscle code - "bflh", "semiten", or "semimem"
            responders_dict: Optional dict of responder IDs by muscle
            non_responders_dict: Optional dict of non-responder IDs by muscle
            speed_df: Optional pre-loaded speed dataframe
            peak_norm_fiber_length_df: Optional pre-loaded peak normalized fiber length dataframe

        Returns:
            matplotlib Figure object

        """
        analysis_utils = Analysis_utils()

        if responders_dict is None:
            responders_dict = analysis_utils.get_norm_fiber_length_responders_ids()
        if non_responders_dict is None:
            non_responders_dict = analysis_utils.get_norm_fiber_length_non_responders_ids()
        if speed_df is None:
            speed_df = analysis_utils.get_speed_df()
        if peak_norm_fiber_length_df is None:
            peak_norm_fiber_length_df = analysis_utils.get_peak_norm_fiber_length_df()

        if muscle not in ["bflh", "semiten", "semimem"]:
            raise ValueError("muscle must be 'bflh', 'semiten', or 'semimem'")

        if pre_or_change not in ["pre", "change"]:
            raise ValueError("pre_or_change must be 'pre' or 'change'")

        # Dynamically construct column name based on muscle and pre_or_change
        y_suffix = "_peak_norm_fiber_length_pre" if pre_or_change == "pre" else "_peak_norm_fiber_length_change"
        y_column = f"{muscle}{y_suffix}"

        # Select x column and label based on pre_or_change (always use m/s)
        if pre_or_change == "pre":
            x_column = "pre_speed_m_s"
            x_label = "Pre-Fatigue Speed (m/s)"
        else:
            x_column = "speed_change_m_s"
            x_label = "Change in Speed (m/s)"

        # Merge speed data with peak normalized fiber length data
        merged_df = pd.merge(peak_norm_fiber_length_df, speed_df, on="subject_id", how="inner")

        # Filter for responders and non-responders
        responders_ids = responders_dict.get(muscle, [])
        non_responders_ids = non_responders_dict.get(muscle, [])
        responders_df = merged_df[merged_df["subject_id"].isin(responders_ids)]
        non_responders_df = merged_df[merged_df["subject_id"].isin(non_responders_ids)]

        # Calculate regression line for all data
        all_x = merged_df[x_column].dropna()
        all_y = merged_df[y_column].dropna()
        valid_mask = all_x.index.isin(all_y.index)
        all_x = all_x[valid_mask]
        all_y = all_y[all_x.index]

        slope, intercept, r_value, p_value, std_err = stats.linregress(all_x, all_y)
        r_squared = r_value**2  # type: ignore

        # Generate points for regression line
        x_range = np.linspace(all_x.min(), all_x.max(), 100)
        y_pred = slope * x_range + intercept

        # Create scatter plot
        if interactive:
            # Create interactive Plotly plot
            fig = go.Figure()

            # Add responders trace
            fig.add_trace(
                go.Scatter(
                    x=responders_df[x_column],
                    y=responders_df[y_column],
                    mode="markers",
                    name="Responders",
                    marker=dict(size=10, color="blue", opacity=0.7),
                    text=responders_df["subject_id"],
                    hovertemplate="<b>Subject:</b> %{text}<br>"
                    f"<b>{x_label}:</b> %{{x:.3f}}<br>"
                    "<b>Peak Strain:</b> %{y:.4f}<br>"
                    "<extra></extra>",
                ),
            )

            # Add non-responders trace
            fig.add_trace(
                go.Scatter(
                    x=non_responders_df[x_column],
                    y=non_responders_df[y_column],
                    mode="markers",
                    name="Non-Responders",
                    marker=dict(size=10, color="orange", opacity=0.7),
                    text=non_responders_df["subject_id"],
                    hovertemplate="<b>Subject:</b> %{text}<br>"
                    f"<b>{x_label}:</b> %{{x:.3f}}<br>"
                    "<b>Peak Strain:</b> %{y:.4f}<br>"
                    "<extra></extra>",
                ),
            )

            # Add regression line
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode="lines",
                    name="Regression Line",
                    line=dict(color="red", dash="dash", width=2),
                    hoverinfo="skip",
                ),
            )

            # Add R² annotation
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref="x domain",
                yref="y domain",
                text=f"R² = {r_squared:.4f}<br>p = {p_value:.4f}",
                showarrow=False,
                bgcolor="wheat",
                opacity=0.8,
                font=dict(size=10),
                align="left",
            )

            # Update layout
            fig.update_layout(
                title=f"{muscle.upper()} {pre_or_change.capitalize()} Peak Strain vs {x_label}",
                xaxis_title=x_label,
                yaxis_title=f"{muscle.upper()} {pre_or_change.capitalize()} Peak Strain (Δfl/ofl)",
                hovermode="closest",
                template="plotly_white",
                width=800,
                height=600,
                font=dict(size=11),
            )

            return fig
        # Create static matplotlib plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            responders_df[x_column],
            responders_df[y_column],
            color="blue",
            label="Responders",
            s=80,
            alpha=0.7,
        )
        ax.scatter(
            non_responders_df[x_column],
            non_responders_df[y_column],
            color="orange",
            label="Non-Responders",
            s=80,
            alpha=0.7,
        )
        # Add regression line
        ax.plot(
            x_range,
            y_pred,
            "r--",
            alpha=0.8,
            linewidth=2,
            label=f"Fit: y={slope:.4f}x+{intercept:.4f} (R²={r_squared:.4f})",
        )
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(
            f"{muscle.upper()} {pre_or_change.capitalize()} Peak Strain (Δfl/ofl)",
            fontsize=11,
        )
        ax.set_title(
            f"{muscle.upper()} {pre_or_change.capitalize()} Peak Strain vs {x_label}",
            fontsize=12,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
