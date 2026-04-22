import warnings
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import spm1d as spm
from sklearn import linear_model
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.model_manager import OpensimModelManager
from utils.data_utils import Data_utils
from utils.file_utils import File_utils
from utils.subject_utils import Subject_utils

# Suppress specific runtime warnings from SPM1D that occur with zero variance data
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Mean of empty slice",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in scalar divide",
)


class Analysis_utils:
    """
    A class to assist with analysis tasks.
    """

    # Muscle groups mapping - leg-unspecific base names
    MUSCLE_GROUPS = {
        "Gluteus Maximus": ["glmax1", "glmax2", "glmax3"],
        "Gluteus Medius": ["glmed1", "glmed2", "glmed3"],
        "Gluteus Minimus": ["glmin1", "glmin2", "glmin3"],
        "Adductor Magnus": ["addmagDist", "addmagIsch", "addmagMid", "addmagProx"],
        "Biceps Femoris": ["bflh", "bfsh"],
        "Semimembranosus": ["semimem"],
        "Semitendinosus": ["semiten"],
        "Rectus Femoris": ["recfem"],
        "Vasti": ["vasint", "vaslat", "vasmed"],
        "Triceps Surae": ["soleus", "gaslat", "gasmed"],
    }

    # Muscle groups by leg - useful for leg-specific analysis
    MUSCLE_GROUPS_BY_LEG = {
        "R": {
            "Gluteus Maximus": ["glmax1_r", "glmax2_r", "glmax3_r"],
            "Gluteus Medius": ["glmed1_r", "glmed2_r", "glmed3_r"],
            "Gluteus Minimus": ["glmin1_r", "glmin2_r", "glmin3_r"],
            "Adductor Magnus": [
                "addmagDist_r",
                "addmagIsch_r",
                "addmagMid_r",
                "addmagProx_r",
            ],
            "Biceps Femoris": ["bflh_r", "bfsh_r"],
            "Semimembranosus": ["semimem_r"],
            "Semitendinosus": ["semiten_r"],
            "Rectus Femoris": ["recfem_r"],
            "Vasti": ["vasint_r", "vaslat_r", "vasmed_r"],
            "Triceps Surae": ["soleus_r", "gaslat_r", "gasmed_r"],
        },
        "L": {
            "Gluteus Maximus": ["glmax1_l", "glmax2_l", "glmax3_l"],
            "Gluteus Medius": ["glmed1_l", "glmed2_l", "glmed3_l"],
            "Gluteus Minimus": ["glmin1_l", "glmin2_l", "glmin3_l"],
            "Adductor Magnus": [
                "addmagDist_l",
                "addmagIsch_l",
                "addmagMid_l",
                "addmagProx_l",
            ],
            "Biceps Femoris": ["bflh_l", "bfsh_l"],
            "Semimembranosus": ["semimem_l"],
            "Semitendinosus": ["semiten_l"],
            "Rectus Femoris": ["recfem_l"],
            "Vasti": ["vasint_l", "vaslat_l", "vasmed_l"],
            "Triceps Surae": ["soleus_l", "gaslat_l", "gasmed_l"],
        },
    }

    # Get all individual muscles across all groups (leg-unspecific base names)
    ALL_MUSCLES = sorted(
        list(set(muscle for muscles in MUSCLE_GROUPS.values() for muscle in muscles)),
    )

    # Get all leg-specific muscles from MUSCLE_GROUPS_BY_LEG
    ALL_MUSCLES_WITH_LEG = sorted(
        list(
            set(
                muscle
                for leg_dict in MUSCLE_GROUPS_BY_LEG.values()
                for muscles in leg_dict.values()
                for muscle in muscles
            ),
        ),
    )

    def __init__(self):
        self._subject_mtu_cache: dict[str, pd.DataFrame] = {}
        self._initial_fiber_lengths_df: pd.DataFrame | None = None
        self._subject_initial_fiber_length_cache: dict[str, pd.Series] = {}

    def _get_subject_mtu_parameters(self, subject_id: str) -> pd.DataFrame:
        """
        Parse and cache per-subject MTU parameters from subjectCalibrated_hybrid.xml.

        Returns:
            DataFrame indexed by muscle name with numeric columns:
            - max_isometric_force
            - optimal_fiber_length

        """
        cache_key = str(subject_id)
        cached_df = self._subject_mtu_cache.get(cache_key)
        if cached_df is not None:
            return cached_df

        file_utils = File_utils()
        session_path = file_utils.get_session_dir(subject_id)
        subject_calibrated_file_path = session_path / "subjectCalibrated_hybrid.xml"

        tree = ET.parse(subject_calibrated_file_path)
        root = tree.getroot()
        mtu_set = root.find("mtuSet")

        if mtu_set is None:
            raise ValueError("mtuSet not found in the XML.")

        processing_data = []
        for mtu in mtu_set.findall("mtu"):
            muscle_name = None
            max_model_iso_force = None
            strength_coeff = None
            optimal_fiber_length = None

            for elements in mtu:
                if elements.tag == "name":
                    muscle_name = elements.text
                if elements.tag == "maxIsometricForce":
                    max_model_iso_force = elements.text
                if elements.tag == "strengthCoefficient":
                    strength_coeff = elements.text
                if elements.tag == "optimalFibreLength":
                    optimal_fiber_length = elements.text

            if muscle_name is not None:
                processing_data.append(
                    {
                        "muscle_name": muscle_name,
                        "max_model_iso_force": max_model_iso_force,
                        "strengthCoefficient": strength_coeff,
                        "optimal_fiber_length": optimal_fiber_length,
                    },
                )

        processing_df = pd.DataFrame(processing_data)
        processing_df.set_index("muscle_name", inplace=True)
        processing_df["max_isometric_force"] = processing_df["max_model_iso_force"].astype(float) * processing_df[
            "strengthCoefficient"
        ].astype(float)
        processing_df["optimal_fiber_length"] = processing_df["optimal_fiber_length"].astype(float)

        self._subject_mtu_cache[cache_key] = processing_df
        return processing_df

    def _get_initial_fiber_lengths_df(self) -> pd.DataFrame:
        """Load and cache initial fiber length lookup CSV once per Analysis_utils instance."""
        if self._initial_fiber_lengths_df is None:
            self._initial_fiber_lengths_df = pd.read_csv(
                "data/helper_results/true_initial_fiber_lengths.csv",
            )
        return self._initial_fiber_lengths_df

    def _get_subject_initial_fiber_lengths(self, subject_id: str) -> pd.Series:
        """Get and cache subject-specific initial fiber lengths indexed by muscle name."""
        cache_key = str(int(subject_id))
        cached_series = self._subject_initial_fiber_length_cache.get(cache_key)
        if cached_series is not None:
            return cached_series

        initial_fl_df = self._get_initial_fiber_lengths_df()
        subject_initial_fl = initial_fl_df[initial_fl_df["subject_id"] == int(subject_id)]
        subject_lookup = subject_initial_fl.set_index("muscle")["calibrated_optimal_fiber_length_diff_from_default"]

        self._subject_initial_fiber_length_cache[cache_key] = subject_lookup
        return subject_lookup

    @classmethod
    def get_muscle_group(cls, group_name: str, leg: str | None = None) -> list:
        """
        Get muscles for a specific group, optionally filtered by leg.

        Args:
            group_name: Name of the muscle group (e.g., 'Gluteus Maximus')
            leg: Optional leg filter - 'R', 'L', or None for both legs

        Returns:
            List of muscle names in the group

        """
        if leg is None:
            return cls.MUSCLE_GROUPS.get(group_name, [])
        return cls.MUSCLE_GROUPS_BY_LEG.get(leg, {}).get(group_name, [])

    def _get_all_muscles_config(self, leg: str) -> dict:
        """
        Get configuration for all muscles, mapping generic muscle names to leg-specific columns.

        Args:
            leg: "R" or "L" to specify which leg

        Returns:
            Dictionary mapping generic muscle names (without _r/_l suffix) to their leg-specific config

        """
        leg_suffix = "_r" if leg == "R" else "_l"

        # EMG mapping for muscles that have EMG data
        emg_mapping = {
            "vasmed": "VM",
            "vaslat": "VL",
            "recfem": "RF",
            "grac": "GRA",
            "tibant": "TA",
            "addlong": "ADDLONG",
            "semimem": "SEMIMEM",
            "bflh": "BFLH",
            "gasmed": "GM",
            "gaslat": "GL",
            "tfl": "TFL",
            "glmax1": "GLUTMAX",
            "glmax2": "GLUTMAX",
            "glmax3": "GLUTMAX",
            "glmed1": "GLUTMED",
            "glmed2": "GLUTMED",
            "glmed3": "GLUTMED",
            "piri": "PIRI",
            "obtint": "OBTINT",
            "quafem": "QF",
        }

        # Get unique muscle base names (without _r or _l suffix)
        muscle_base_names = set()
        for muscle in self.ALL_MUSCLES_WITH_LEG:
            if muscle.endswith("_r") or muscle.endswith("_l"):
                base_name = muscle[:-2]  # Remove last 2 characters (_r or _l)
                muscle_base_names.add(base_name)

        # Create config mapping base names to leg-specific columns
        config = {}
        for base_name in sorted(muscle_base_names):
            config[base_name] = {
                "col": f"{base_name}{leg_suffix}",
                "emg_col": emg_mapping.get(base_name),  # Map to EMG column if available
            }
        return config

    def _get_muscle_config(self, leg: str) -> dict:
        """
        Get muscle configuration mapping muscle names to their column names.

        Args:
            leg: "R" or "L" to specify which leg

        Returns:
            Dictionary mapping standardized muscle names to their config

        """
        leg_suffix = "_r" if leg == "R" else "_l"

        return {
            "biceps_femoris_long_head": {
                "col": f"bflh{leg_suffix}",
                "emg_col": "BFLH",
            },
            "semitendinosus": {
                "col": f"semiten{leg_suffix}",
                "emg_col": "SEMIMEM",  # Uses SEMIMEM raw EMG as input (per excitationGenerator.xml)
            },
            "semimembranosus": {
                "col": f"semimem{leg_suffix}",
                "emg_col": "SEMIMEM",
            },
            "gastrocnemius_medialis": {
                "col": f"gasmed{leg_suffix}",
                "emg_col": "GM",
            },
            "vastus_medialis": {
                "col": f"vasmed{leg_suffix}",
                "emg_col": "VM",
            },
            "vastus_lateralis": {
                "col": f"vaslat{leg_suffix}",
                "emg_col": "VL",
            },
            "rectus_femoris": {
                "col": f"recfem{leg_suffix}",
                "emg_col": "RF",
            },
        }

    def _apply_lowpass_filter_to_muscles(
        self,
        df: pd.DataFrame,
        muscles_config: dict,
        cutoff: float = 10,
        order: int = 2,
    ) -> pd.DataFrame:
        """
        Apply low-pass filter to all muscles in the configuration.

        Args:
            df: DataFrame containing muscle data with 'time' column
            muscles_config: Muscle configuration from _get_muscle_config()
            cutoff: Cutoff frequency in Hz (default: 10)
            order: Filter order (default: 2)

        Returns:
            DataFrame with filtered muscle columns

        """
        df_filtered = df.copy()

        # Calculate sampling frequency
        total_time = df_filtered["time"].iloc[-1] - df_filtered["time"].iloc[0]
        fs = len(df_filtered) / total_time if len(df_filtered) > 1 else 100

        # Filter all muscles in the config
        for muscle_key, muscle_info in muscles_config.items():
            col_name = muscle_info["col"]
            if col_name in df_filtered.columns:
                df_filtered[col_name] = self.butter_lowpass_filter(  # type: ignore
                    df_filtered[col_name],
                    cutoff,
                    fs,
                    order,
                )

        return df_filtered

    def _build_trial_dataframe(
        self,
        subject: str,
        trial: str,
        leg: str,
        data_dict: dict,
        muscles_config: dict,
        num_rows: int,
    ) -> pd.DataFrame:
        """
        Build a trial DataFrame dynamically from provided data sources.

        Args:
            subject: Subject ID
            trial: Trial name
            leg: Measured leg ("R" or "L")
            data_dict: Dictionary containing all data sources with keys:
                - "ik_df": Inverse kinematics data
                - "id_df": Inverse dynamics data
                - "norm_force_df": Normalized muscle forces
                - "strain_df": Muscle strain rate data
                - "emg_df": Raw EMG data (optional)
                - "emg_adjusted_df": CEINMS adjusted EMG (optional)
                - "fiber_lengths_df": Fiber length data (optional)
                - "tendon_lengths_df": Tendon length data (optional)
                - "fiber_velocity_df": Fiber velocity data (optional)
                - "fiber_force_df": Fiber force data (optional)
                - "active_force_df": CEINMS active force data (optional)
                - "passive_force_df": CEINMS passive force data (optional)
                - "activation_df": Muscle activation data (optional)
            muscles_config: Muscle configuration from _get_muscle_config()
            num_rows: Number of rows for the output DataFrame

        Returns:
            DataFrame with trial data

        """
        # Extract column names based on leg
        hip_flexion_col = "hip_flexion_r" if leg == "R" else "hip_flexion_l"
        hip_adduction_col = "hip_adduction_r" if leg == "R" else "hip_adduction_l"
        hip_rotation_col = "hip_rotation_r" if leg == "R" else "hip_rotation_l"
        knee_angle_col = "knee_angle_r" if leg == "R" else "knee_angle_l"
        ankle_angle_col = "ankle_angle_r" if leg == "R" else "ankle_angle_l"
        pelvis_tx_col = "pelvis_tx"
        hip_flexion_moment_col = "hip_flexion_r_moment" if leg == "R" else "hip_flexion_l_moment"
        hip_adduction_moment_col = "hip_adduction_r_moment" if leg == "R" else "hip_adduction_l_moment"
        hip_rotation_moment_col = "hip_rotation_r_moment" if leg == "R" else "hip_rotation_l_moment"
        knee_angle_moment_col = "knee_angle_r_moment" if leg == "R" else "knee_angle_l_moment"
        ankle_angle_moment_col = "ankle_angle_r_moment" if leg == "R" else "ankle_angle_l_moment"

        # Get data sources
        ik_df = data_dict["ik_df"]
        id_df = data_dict["id_df"]

        # Build base trial data
        trial_data = {
            "subject_id": np.full(num_rows, subject, dtype=object),
            "trial": np.full(num_rows, trial, dtype=object),
            "leg": np.full(num_rows, leg.lower(), dtype=object),
        }

        # Add index and time if available
        if "index" in ik_df.columns:
            trial_data["index"] = ik_df["index"].values[:num_rows]
        else:
            trial_data["index"] = np.arange(num_rows)
        trial_data["time"] = ik_df["time"].values[:num_rows]

        # Add kinematic data
        trial_data["hip_flexion"] = ik_df[hip_flexion_col].values[:num_rows]
        trial_data["knee_angle"] = ik_df[knee_angle_col].values[:num_rows]
        trial_data["ankle_angle"] = ik_df[ankle_angle_col].values[:num_rows]
        trial_data["pelvis_tx"] = ik_df[pelvis_tx_col].values[:num_rows]
        trial_data["hip_adduction"] = ik_df[hip_adduction_col].values[:num_rows]
        trial_data["hip_rotation"] = ik_df[hip_rotation_col].values[:num_rows]

        # Add kinetic data
        trial_data["hip_flexion_moment"] = id_df[hip_flexion_moment_col].values[:num_rows]
        trial_data["hip_adduction_moment"] = id_df[hip_adduction_moment_col].values[:num_rows]
        trial_data["hip_rotation_moment"] = id_df[hip_rotation_moment_col].values[:num_rows]
        trial_data["knee_angle_moment"] = id_df[knee_angle_moment_col].values[:num_rows]
        trial_data["ankle_angle_moment"] = id_df[ankle_angle_moment_col].values[:num_rows]

        # Define data sources for muscles
        muscle_data_sources = []

        # Always include force, strain, and strain rate
        if "norm_force_df" in data_dict:
            muscle_data_sources.append(("norm_force", data_dict["norm_force_df"]))
        if "strain_df" in data_dict:
            muscle_data_sources.append(("strain", data_dict["strain_df"]))
        if "strain_rate_df" in data_dict:
            muscle_data_sources.append(("strain_rate", data_dict["strain_rate_df"]))

        # Add optional data sources
        optional_sources = [
            ("adjusted_emg", "emg_adjusted_df"),
            ("fiber_lengths", "fiber_lengths_df"),
            ("tendon_lengths", "tendon_lengths_df"),
            ("fiber_velocity", "fiber_velocity_df"),
            ("fiber_force", "fiber_force_df"),
            ("active_force", "active_force_df"),
            ("passive_force", "passive_force_df"),
            ("activation", "activation_df"),
        ]

        for suffix, key in optional_sources:
            if key in data_dict and data_dict[key] is not None:
                muscle_data_sources.append((suffix, data_dict[key]))

        # Add muscle data dynamically
        for muscle_name, muscle_info in muscles_config.items():
            # Use abbreviated muscle name (without leg suffix) for column naming
            muscle_abbrev = muscle_info["col"].rstrip("_rl")  # Remove _r or _l suffix

            # Add data for each available source
            for suffix, df in muscle_data_sources:
                col_name = f"{muscle_abbrev}_{suffix}"
                if muscle_info["col"] in df.columns:
                    trial_data[col_name] = df[muscle_info["col"]].values[:num_rows]
                else:
                    # Add NaN values if muscle column doesn't exist in source
                    trial_data[col_name] = np.full(num_rows, np.nan)

            # Add RAW EMG data from emg.mot file (uses original EMG column names like BFLH, SEMIMEM)
            if "emg_df" in data_dict and data_dict["emg_df"] is not None:
                emg_col_name = muscle_info.get("emg_col")
                if emg_col_name is not None and emg_col_name in data_dict["emg_df"].columns:
                    # Map from raw EMG column names (BFLH, SEMIMEM, etc.) to muscle columns
                    trial_data[f"{muscle_abbrev}_emg"] = data_dict["emg_df"][emg_col_name].values[:num_rows]
                else:
                    # No raw EMG available for this muscle (e.g., semitendinosus)
                    trial_data[f"{muscle_abbrev}_emg"] = np.full(num_rows, np.nan)
            else:
                trial_data[f"{muscle_abbrev}_emg"] = np.full(num_rows, np.nan)

            # Add power calculation if both velocity and force are available
            if "fiber_velocity_df" in data_dict and "fiber_force_df" in data_dict:
                velocity_df = data_dict["fiber_velocity_df"]
                force_df = data_dict["fiber_force_df"]
                if muscle_info["col"] in velocity_df.columns and muscle_info["col"] in force_df.columns:
                    power = velocity_df[muscle_info["col"]] * force_df[muscle_info["col"]]
                    trial_data[f"{muscle_abbrev}_power"] = power.values[:num_rows]

        return pd.DataFrame(trial_data)

    def time_normalise_df(self, df: pd.DataFrame, sampling_frequency=None):
        if not isinstance(df, pd.DataFrame):
            raise Exception("Input must be a pandas DataFrame")
        if sampling_frequency is None:
            try:
                sampling_frequency = 1 / (df["time"][1] - df["time"][0])
            except KeyError:
                raise Exception('Input DataFrame must contain a column named "time"')

        # Remove any unnamed/empty columns
        df_cleaned = df.copy()
        # Drop columns that are all NaN or unnamed columns
        df_cleaned = df_cleaned.dropna(axis=1, how="all")
        df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.contains("^Unnamed")]

        normalised_df = pd.DataFrame(columns=df_cleaned.columns)

        for column in df_cleaned.columns:
            normalised_df[column] = np.zeros(101)

            currentData = df_cleaned[column]
            currentData = currentData[~np.isnan(currentData)]

            if len(currentData) == 0:
                print(
                    f"Warning: Column '{column}' contains only NaN values. Filling with zeros.",
                )
                continue

            timeTrial = np.arange(
                0,
                len(currentData) / sampling_frequency,
                1 / sampling_frequency,
            )

            if len(timeTrial) == 0:
                print(
                    f"Warning: Column '{column}' resulted in empty time array. Filling with zeros.",
                )
                continue

            Tnorm = np.arange(0, timeTrial[-1], timeTrial[-1] / 101)
            if len(Tnorm) == 102:
                Tnorm = Tnorm[:-1]
            normalised_df[column] = np.interp(Tnorm, timeTrial, currentData)

        return normalised_df

    def get_muscle_group_mean(self, df: pd.DataFrame, id: str):
        mean_df = pd.DataFrame({"time": df["time"]})

        if not isinstance(df, pd.DataFrame):
            raise Exception("Input must be a pandas DataFrame")
        model_manager = OpensimModelManager(id)
        muscle_groups = model_manager.get_muscle_groups()

        for group_name, muscles in muscle_groups.items():
            valid_muscles = [m for m in muscles if m in df.columns]
            if not valid_muscles:
                continue

            muscles_df = df[valid_muscles]
            column_name = f"{group_name}"

            mean_df[f"{column_name}_mean"] = muscles_df.mean(axis=1)
        return mean_df

    def get_muscle_group_sum(self, df: pd.DataFrame, id: str):
        sum_df = pd.DataFrame({"time": df["time"]})

        if not isinstance(df, pd.DataFrame):
            raise Exception("Input must be a pandas DataFrame")
        model_manager = OpensimModelManager(id)
        muscle_groups = model_manager.get_muscle_groups()

        for group_name, muscles in muscle_groups.items():
            valid_muscles = [m for m in muscles if m in df.columns]
            if not valid_muscles:
                continue

            muscles_df = df[valid_muscles]
            column_name = f"{group_name}"

            sum_df[f"{column_name}_sum"] = muscles_df.sum(axis=1)
        return sum_df

    def get_muscle_work_df(self, muscles: list | None = None):
        """
        Calculate positive and negative work for muscles across subjects for both pre and post conditions.

        Args:
            muscles: List of muscle names to include. If None, uses hamstring muscles.

        Returns:
            DataFrame with subjects as rows and columns:
            - {muscle}_positive_work_pre
            - {muscle}_negative_work_pre
            - {muscle}_positive_work_post
            - {muscle}_negative_work_post

        """
        file_utils = File_utils()
        subject_utils = Subject_utils()

        # Default to hamstring muscles if not specified
        if muscles is None:
            muscles = ["bflh", "semiten", "semimem"]

        all_subjects = subject_utils.get_all_subjects()
        all_results = []

        for subject_id in all_subjects:
            leg = subject_utils.get_measured_leg(subject_id)
            body_mass = subject_utils.get_body_mass(subject_id)
            available_trials = subject_utils.get_all_relevant_trials(subject_id)

            subject_row = {"subject_id": subject_id}

            # Process PRE trials (baseline A1 and B1)
            pre_trials = ["Run_baselineA1", "Run_baselineB1"]
            pre_trials = [t for t in pre_trials if t in available_trials]

            if pre_trials:
                pre_work_data = []

                for trial in pre_trials:
                    # Force
                    fiber_force_file = file_utils.get_ceinms_active_muscle_forces_file_path(
                        subject_id,
                        trial,
                    )
                    if not fiber_force_file.exists():
                        print(
                            f"Warning: Force file not found for {subject_id}, {trial}",
                        )
                        continue

                    fiber_force_df = Data_utils().create_df(fiber_force_file)

                    # Velocity
                    fiber_velocity_file = file_utils.get_ceinms_fiber_velocities_file_path(
                        subject_id,
                        trial,
                    )
                    if not fiber_velocity_file.exists():
                        print(
                            f"Warning: Velocity file not found for {subject_id}, {trial}",
                        )
                        continue

                    fiber_velocity_df = Data_utils().create_df(fiber_velocity_file)

                    time = fiber_force_df["time"]

                    # Calculate work for each specified muscle
                    trial_work = {}

                    for muscle_base in muscles:
                        # Add leg suffix
                        muscle_name = f"{muscle_base}_{leg.lower()}"

                        if muscle_name not in fiber_force_df.columns:
                            print(
                                f"Warning: {muscle_name} not found in force data for {subject_id}",
                            )
                            continue

                        if muscle_name not in fiber_velocity_df.columns:
                            print(
                                f"Warning: {muscle_name} not found in velocity data for {subject_id}",
                            )
                            continue

                        force = fiber_force_df[muscle_name].values
                        velocity = fiber_velocity_df[muscle_name].values

                        # Calculate power (force * velocity)
                        power = force * velocity  # type: ignore[operator]

                        # Separate positive and negative power
                        positive_power = np.where(power > 0, power, 0)
                        negative_power = np.where(power < 0, power, 0)

                        # Integrate to get work
                        positive_work = np.trapz(positive_power, time)  # type: ignore
                        negative_work = np.trapz(negative_power, time)  # type: ignore

                        # Normalize by body mass to get J/kg
                        trial_work[f"{muscle_base}_positive_work"] = positive_work / body_mass
                        trial_work[f"{muscle_base}_negative_work"] = negative_work / body_mass

                    if trial_work:
                        pre_work_data.append(trial_work)

                # Average across pre trials
                if pre_work_data:
                    pre_df = pd.DataFrame(pre_work_data)
                    for col in pre_df.columns:
                        # Rename to add _pre suffix
                        subject_row[f"{col}_pre"] = pre_df[col].mean()

            # Process POST trial (RunL1)
            post_trials = ["RunL1"]
            post_trials = [t for t in post_trials if t in available_trials]

            if post_trials:
                trial = post_trials[0]

                # Force
                fiber_force_file = file_utils.get_ceinms_active_muscle_forces_file_path(
                    subject_id,
                    trial,
                )
                if fiber_force_file.exists():
                    fiber_force_df = Data_utils().create_df(fiber_force_file)

                    # Velocity
                    fiber_velocity_file = file_utils.get_ceinms_fiber_velocities_file_path(
                        subject_id,
                        trial,
                    )
                    if fiber_velocity_file.exists():
                        fiber_velocity_df = Data_utils().create_df(fiber_velocity_file)

                        time = fiber_force_df["time"]

                        # Calculate work for each specified muscle
                        for muscle_base in muscles:
                            # Add leg suffix
                            muscle_name = f"{muscle_base}_{leg.lower()}"

                            if muscle_name not in fiber_force_df.columns:
                                print(
                                    f"Warning: {muscle_name} not found in force data for {subject_id}",
                                )
                                continue

                            if muscle_name not in fiber_velocity_df.columns:
                                print(
                                    f"Warning: {muscle_name} not found in velocity data for {subject_id}",
                                )
                                continue

                            force = fiber_force_df[muscle_name].values
                            velocity = fiber_velocity_df[muscle_name].values

                            # Calculate power (force * velocity)
                            power = force * velocity  # type: ignore[operator]

                            # Separate positive and negative power
                            positive_power = np.where(power > 0, power, 0)
                            negative_power = np.where(power < 0, power, 0)

                            # Integrate to get work
                            positive_work = np.trapz(positive_power, time)  # type: ignore
                            negative_work = np.trapz(negative_power, time)  # type: ignore

                            # Normalize by body mass to get J/kg
                            subject_row[f"{muscle_base}_positive_work_post"] = positive_work / body_mass
                            subject_row[f"{muscle_base}_negative_work_post"] = negative_work / body_mass

            all_results.append(subject_row)

        # Create final DataFrame
        work_df = pd.DataFrame(all_results)

        # Set subject_id as index for cleaner display
        if not work_df.empty:
            work_df = work_df.set_index("subject_id")

        return work_df

    def get_subjects_mean(self, subject_ids: list, variable: str):
        """
        Get the mean of a specific variable across multiple subjects.
        :param subject_ids: List of subject IDs.
        :param variable: The variable to calculate the mean for.
        :return: DataFrame with the mean values.
        """
        if not isinstance(subject_ids, list):
            raise ValueError("subject_ids must be a list")
        ###tbd###

    def calculate_marker_errors(self, subject_id, trial):
        marker_experimental_path = File_utils().get_trial_dir(subject_id, trial) / "markers.trc"

        marker_virtual_path = File_utils().get_trial_dir(subject_id, trial) / "_ik_model_marker_locations.sto"

        exp_path = marker_experimental_path
        virtual_path = marker_virtual_path

        if not exp_path.exists():
            raise FileNotFoundError(f"Experimental marker file not found: {exp_path}")
        if not virtual_path.exists():
            raise FileNotFoundError(f"Virtual marker file not found: {virtual_path}")

        # Initialize file utils and load data
        file_utils = File_utils()
        virtual_markers_df = file_utils.load_sto_file(virtual_path)
        experimental_markers_df = file_utils.load_trc_file(exp_path)

        # Extract marker names (remove coordinate suffixes)
        marker_names = [
            col.split("_")[0] for col in experimental_markers_df.columns if col not in ["Time", "Frame#"] and "_" in col
        ]
        marker_names = list(
            dict.fromkeys(marker_names),
        )  # Remove duplicates, preserve order

        # Find time alignment between datasets
        virtual_time = virtual_markers_df["Time"]
        exp_time = experimental_markers_df["Time"]

        # Find closest indices for time alignment - get integer positions
        initial_idx_label = (exp_time - virtual_time.iloc[0]).abs().idxmin()
        final_idx_label = (exp_time - virtual_time.iloc[-1]).abs().idxmin()

        # Convert index labels to integer positions
        initial_index_raw = experimental_markers_df.index.get_loc(initial_idx_label)
        final_index_raw = experimental_markers_df.index.get_loc(final_idx_label)

        # Ensure indices are integers (handle slice/array returns from get_loc)
        initial_index = (
            int(initial_index_raw)
            if isinstance(initial_index_raw, (int, np.integer))
            else int(initial_index_raw.start)
            if isinstance(initial_index_raw, slice)
            else int(initial_index_raw[0])
        )
        final_index = (
            int(final_index_raw)
            if isinstance(final_index_raw, (int, np.integer))
            else int(final_index_raw.start)
            if isinstance(final_index_raw, slice)
            else int(final_index_raw[0])
        )

        # Initialize results
        distances_df = pd.DataFrame({"Time": virtual_time.values})
        mean_errors = {}

        print("Calculating marker errors for all markers...")

        for marker_name in marker_names:
            try:
                # Find columns for this marker in both datasets - use natural order like compare_marker_locations.py
                exp_cols = [col for col in experimental_markers_df.columns if col.split("_")[0] == marker_name]
                virtual_cols = [col for col in virtual_markers_df.columns if col.split("_")[0] == marker_name]

                if len(exp_cols) < 3 or len(virtual_cols) < 3:
                    print(
                        f"Warning: Insufficient coordinate columns for marker '{marker_name}'. Skipping.",
                    )
                    continue

                # Get experimental data for the time range and convert mm to m (following compare_marker_locations.py approach)
                exp_slice = experimental_markers_df.iloc[initial_index : final_index + 1]
                x1 = np.array(exp_slice[exp_cols[0]].values, dtype=float) / 1000.0  # Convert mm to m
                y1 = np.array(exp_slice[exp_cols[1]].values, dtype=float) / 1000.0
                z1 = np.array(exp_slice[exp_cols[2]].values, dtype=float) / 1000.0

                # Get virtual data (already in meters)
                x2 = np.array(virtual_markers_df[virtual_cols[0]].values, dtype=float)
                y2 = np.array(virtual_markers_df[virtual_cols[1]].values, dtype=float)
                z2 = np.array(virtual_markers_df[virtual_cols[2]].values, dtype=float)

                # Ensure arrays are the same length
                min_len = min(len(x1), len(x2))
                x1, y1, z1 = x1[:min_len], y1[:min_len], z1[:min_len]
                x2, y2, z2 = x2[:min_len], y2[:min_len], z2[:min_len]

                # Calculate 3D Euclidean distance
                distances = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

                # Store results
                distances_df[marker_name] = pd.Series(distances)
                mean_errors[marker_name] = np.mean(distances)

                print(
                    f"Processed marker '{marker_name}': mean error = {mean_errors[marker_name]:.4f} m",
                )

            except (KeyError, IndexError, ValueError) as e:
                print(f"Could not process marker '{marker_name}': {e}")

        # Save results

        output_dir = file_utils.get_trial_dir(subject_id, trial)

        # Save mean errors
        mean_errors_file = output_dir / "_ik_marker_errors_mean.txt"
        with open(mean_errors_file, "w") as f:
            f.write("Mean errors for each marker (m)\n\n")
            f.writelines(f"{marker} = {error:.4f} m\n" for marker, error in mean_errors.items())

        # Save all distance data
        all_errors_file = output_dir / "_ik_marker_errors_all.sto"
        distances_clean = distances_df.dropna(axis=1, how="all")
        file_utils.write_sto_file(distances_clean, all_errors_file)

        return distances_df, mean_errors

    def get_norm_forces_df(self, id: str, trial: str):
        file_utils = File_utils()
        force_file = file_utils.get_ceinms_muscle_forces_file_path(id, trial)
        force_df = Data_utils().create_df(force_file)
        processing_df = self._get_subject_mtu_parameters(id)

        norm_force_df = pd.DataFrame({"time": force_df["time"]})

        for muscle in force_df.columns:
            if muscle == "time" or "Unnamed" in muscle or muscle.strip() == "":
                continue

            if muscle not in processing_df.index:
                raise KeyError(
                    f"Muscle '{muscle}' not found in subjectCalibrated_hybrid.xml for subject {id}.",
                )

            denominator = float(processing_df.at[muscle, "max_isometric_force"])  # type: ignore[arg-type]
            muscle_values = force_df[muscle].to_numpy(dtype=float)
            norm_force_df[muscle] = muscle_values / denominator

        return norm_force_df

    def get_norm_fiber_velocity_df(self, id: str, trial: str):
        """
        Get fiber velocity normalized by optimal fiber length.

        Args:
            id: Subject ID
            trial: Trial name

        Returns:
            DataFrame with normalized fiber velocities (in optimal fiber lengths per second)

        """
        file_utils = File_utils()
        fiber_velocity_file = file_utils.get_ceinms_fiber_velocities_file_path(
            id,
            trial,
        )
        fiber_velocity_df = Data_utils().create_df(fiber_velocity_file)
        processing_df = self._get_subject_mtu_parameters(id)

        norm_fiber_velocity_df = pd.DataFrame({"time": fiber_velocity_df["time"]})

        for muscle in fiber_velocity_df.columns:
            if muscle == "time" or "Unnamed" in muscle or muscle.strip() == "":
                continue

            if muscle not in processing_df.index:
                raise KeyError(
                    f"Muscle '{muscle}' not found in subjectCalibrated_hybrid.xml for subject {id}.",
                )

            denominator = float(processing_df.at[muscle, "optimal_fiber_length"])  # type: ignore[arg-type]
            muscle_values = fiber_velocity_df[muscle].to_numpy(dtype=float)
            norm_fiber_velocity_df[muscle] = muscle_values / denominator

        return norm_fiber_velocity_df

    def get_optimal_fiber_length_df(self, calibrated=True):
        subject_utils = Subject_utils()
        file_utils = File_utils()

        all_subjects = subject_utils.get_all_subjects()

        rows = []

        for subject in all_subjects:
            ceinms_model = (
                file_utils.get_ceinms_cal_model_file_path(subject)
                if calibrated
                else file_utils.get_ceinms_uncal_model_file_path(subject)
            )

            tree = ET.parse(ceinms_model)
            root = tree.getroot()
            mtu_set = root.find("mtuSet")

            if mtu_set is not None:
                for mtu in mtu_set:
                    muscle_name = None
                    optimal_fiber_length = None

                    for element in mtu:
                        if element.tag == "name":
                            muscle_name = element.text
                        if element.tag == "optimalFibreLength":
                            optimal_fiber_length = element.text

                    if muscle_name and optimal_fiber_length:
                        rows.append(
                            {
                                "subject_id": subject,
                                "muscle_name": muscle_name,
                                "optimal_fiber_length": float(optimal_fiber_length),
                            },
                        )

        df = pd.DataFrame(
            rows,
            columns=["subject_id", "muscle_name", "optimal_fiber_length"],
        )

        return df

    def get_initial_fiber_lengths_df(self):
        import math

        import opensim as osim

        subject_utils = Subject_utils()
        fl_records = []

        for subject in subject_utils.get_all_subjects():
            model_path = OpensimModelManager(subject).get_model_path()
            osim_model = osim.Model(str(model_path))
            state = osim_model.initSystem()

            static_mot_path = f"data/simulations/{subject}/pre/static/ik.mot"
            mot = osim.Storage(str(static_mot_path))
            labels = mot.getColumnLabels()
            first_row = mot.getStateVector(0)
            data = first_row.getData()

            for i in range(labels.getSize()):
                coord_name = labels.get(i)
                if osim_model.getCoordinateSet().contains(coord_name):
                    coord = osim_model.getCoordinateSet().get(coord_name)
                    val = data.get(i - 1)
                    if coord.getMotionType() != osim.Coordinate.Translational:
                        val = math.radians(val)
                    coord.setValue(state, val)

            osim_model.realizePosition(state)
            osim_model.equilibrateMuscles(state)

            muscles = osim_model.getMuscles()
            for i in range(muscles.getSize()):
                musc = muscles.get(i)
                muscle_name = musc.getName()
                length = musc.getFiberLength(state)
                fl_records.append(
                    {
                        "subject_id": subject,
                        "muscle": muscle_name,
                        "initial_fiber_length": length,
                    }
                )

        return pd.DataFrame(fl_records)

    def get_optimal_fiber_length_diff_df(self):
        uncal_ofl_df = self.get_optimal_fiber_length_df(calibrated=False)
        cal_ofl_df = self.get_optimal_fiber_length_df(calibrated=True)

        diff_df = pd.DataFrame()
        diff_df["subject_id"] = uncal_ofl_df["subject_id"]
        diff_df["muscle"] = uncal_ofl_df["muscle_name"]
        diff_df["optimal_fiber_length_diff"] = cal_ofl_df["optimal_fiber_length"] - uncal_ofl_df["optimal_fiber_length"]
        return diff_df

    # def get_muscle_strain_df(self, subject_id: str, trial: str):
    #     file_utils = File_utils()

    #     fiber_length_file = file_utils.get_ceinms_fiber_lengths_file_path(
    #         subject_id,
    #         trial,
    #     )
    #     fiber_length_df = Data_utils().create_df(fiber_length_file)

    #     muscle_abs_strain_df = pd.DataFrame({"time": fiber_length_df["time"]})

    #     optimal_fiber_length_df = self.get_optimal_fiber_length_df()
    #     subject_ofl = optimal_fiber_length_df[optimal_fiber_length_df["subject_id"] == subject_id]

    #     for col in fiber_length_df.columns:
    #         if col == "time" or "Unnamed" in col or col.strip() == "":
    #             continue

    #         ofl_row = subject_ofl[subject_ofl["muscle_name"] == col]

    #         if ofl_row.empty:
    #             print(
    #                 f"Warning: Optimal fiber length not found for muscle '{col}' in subject {subject_id}. Skipping strain calculation for this muscle.",  # noqa: E501
    #             )
    #             continue

    #         optimal_fiber_length = ofl_row["optimal_fiber_length"].to_numpy()[0]

    #         # Strain = (current length - optimal length) / optimal length
    #         muscle_abs_strain_df[col] = (fiber_length_df[col] - optimal_fiber_length) / optimal_fiber_length

    #     return muscle_abs_strain_df

    def get_muscle_strain_df(self, subject_id: str, trial: str):
        file_utils = File_utils()
        fiber_length_file = file_utils.get_ceinms_fiber_lengths_file_path(
            subject_id,
            trial,
        )
        fiber_length_df = Data_utils().create_df(fiber_length_file)
        subject_initial_fl = self._get_subject_initial_fiber_lengths(subject_id)

        muscle_strain_df = pd.DataFrame({"time": fiber_length_df["time"]})

        for col in fiber_length_df.columns:
            if col == "time" or "Unnamed" in col or col.strip() == "":
                continue

            if col not in subject_initial_fl.index:
                print(
                    f"Warning: Initial fiber length not found for muscle '{col}' in subject {subject_id}. Skipping strain calculation for this muscle.",
                )
                continue

            initial_fiber_length = float(subject_initial_fl.at[col])  # type: ignore[arg-type]
            fiber_length_values = fiber_length_df[col].to_numpy(dtype=float)
            muscle_strain_df[col] = (fiber_length_values - initial_fiber_length) / initial_fiber_length

        return muscle_strain_df

    def get_tendon_strain_df(self, subject_id: str, trial: str):
        file_utils = File_utils()
        tendon_length_file = file_utils.get_ceinms_tendon_lengths_file_path(
            subject_id,
            trial,
        )

        tendon_length_df = Data_utils().create_df(tendon_length_file)

        tendon_strain_df = pd.DataFrame({"time": tendon_length_df["time"]})

        for col in tendon_length_df.columns:
            if col == "time" or "Unnamed" in col or col.strip() == "":
                continue
            # current length - last length
            numerator = tendon_length_df[col] - tendon_length_df[col].shift(1)
            # Delta length / last length
            tendon_strain_df[col] = numerator / tendon_length_df[col].shift(1)

        tendon_strain_df.fillna(0, inplace=True)  # Replace NaN values with 0

        return tendon_strain_df

    def create_master_results_df(self):
        all_results = []  # Use a list to collect DataFrames
        subject_utils = Subject_utils()
        all_subjects = subject_utils.get_all_subjects()
        for subject in all_subjects:
            trials = []
            leg = subject_utils.get_measured_leg(subject)

            trials = subject_utils.get_all_relevant_trials(subject)

            file_utils = File_utils()

            data_utils = Data_utils()

            for trial in trials:
                ik_file_path = file_utils.get_ik_file_path(subject, trial)
                if not ik_file_path.exists():
                    print(
                        f"Warning: IK file not found for subject {subject}, trial {trial}. Skipping.",
                    )

                # IK
                ik_df = data_utils.create_df(ik_file_path)
                ik_norm_df = self.time_normalise_df(ik_df)

                # ID
                id_file_path = file_utils.get_id_file_path(subject, trial)
                if id_file_path is None or not id_file_path.exists():
                    print(
                        f"Warning: ID file not found for subject {subject}, trial {trial}. Skipping.",
                    )
                    continue

                id_df = data_utils.create_df(id_file_path)
                id_norm_df_time = self.time_normalise_df(id_df)
                # Normalise by body mass
                subject_mass = subject_utils.get_body_mass(subject)
                # subject_height = subject_utils.get_body_height(subject) # Optional if needed
                id_norm_df = id_norm_df_time.copy()
                for col in id_norm_df.columns:
                    if col == "time":
                        continue
                    id_norm_df[col] = id_norm_df_time[col] / subject_mass

                # Get muscle configuration - use all muscles for master results
                muscles_config = self._get_all_muscles_config(leg)

                # Normalised force
                force_df = self.get_norm_forces_df(subject, trial)
                norm_force_df = self.time_normalise_df(force_df)

                # Strain
                strain_df = self.get_muscle_strain_df(subject, trial)
                strain_norm_df = self.time_normalise_df(strain_df)
                # # Apply low-pass filter to all muscles
                # strain_norm_df = self._apply_lowpass_filter_to_muscles(
                #     strain_norm_df, muscles_config
                # )

                # EMG before CEINMS adjustment
                emg_df = self.get_emg_df(subject, trial)
                emg_norm_df = self.time_normalise_df(emg_df)
                # Remove whitespaces from column names
                emg_norm_df.columns = emg_norm_df.columns.str.strip()

                # EMG after CEINMS adjustment
                adjusted_emg_df = self.get_ceinms_adjusted_emg_df(subject, trial)
                emg_adjusted_norm_df = self.time_normalise_df(adjusted_emg_df)

                # ceinms normalized fiber lengths
                ceinms_fiber_length_file = file_utils.get_ceinms_normalized_fiber_lengths_file_path(
                    subject,
                    trial,
                )
                ceinms_fiber_length_df = data_utils.create_df(ceinms_fiber_length_file)
                ceinms_fiber_length_norm_df = self.time_normalise_df(
                    ceinms_fiber_length_df,
                )

                # Ceinms normalized tendon lengths
                ceinms_tendon_length_file = file_utils.get_ceinms_normalized_tendon_lengths_file_path(
                    subject,
                    trial,
                )
                ceinms_tendon_length_df = data_utils.create_df(ceinms_tendon_length_file)
                ceinms_tendon_length_norm_df = self.time_normalise_df(
                    ceinms_tendon_length_df,
                )

                # Apply low-pass filter to all muscles
                # ceinms_fiber_length_norm_df2 = self._apply_lowpass_filter_to_muscles(
                #     ceinms_fiber_length_norm_df, muscles_config
                # )

                # ceinms fiber velocity - normalized by optimal fiber length
                ceinms_fiber_velocity_df = self.get_norm_fiber_velocity_df(
                    subject,
                    trial,
                )
                ceinms_fiber_velocity_norm_df = self.time_normalise_df(
                    ceinms_fiber_velocity_df,
                )
                # Apply low-pass filter to all muscles
                # ceinms_fiber_velocity_norm_df = self._apply_lowpass_filter_to_muscles(
                #     ceinms_fiber_velocity_norm_df, muscles_config
                # )

                # ceinms fiber force
                ceinms_fiber_force_file = file_utils.get_ceinms_active_muscle_forces_file_path(subject, trial)
                ceinms_fiber_force_df = data_utils.create_df(ceinms_fiber_force_file)
                ceinms_fiber_force_norm_df = self.time_normalise_df(
                    ceinms_fiber_force_df,
                )

                # ceinms passive muscle force
                try:
                    ceinms_passive_force_file = file_utils.get_ceinms_passive_muscle_forces_file_path(subject, trial)
                    ceinms_passive_force_df = data_utils.create_df(ceinms_passive_force_file)
                    ceinms_passive_force_norm_df = self.time_normalise_df(
                        ceinms_passive_force_df,
                    )
                except FileNotFoundError:
                    ceinms_passive_force_norm_df = None
                # Apply low-pass filter to all muscles
                # ceinms_fiber_force_norm_df = self._apply_lowpass_filter_to_muscles(
                #     ceinms_fiber_force_norm_df, muscles_config
                # )

                # Ceinms activation
                ceinms_activation_path = file_utils.get_muscle_activation_file_path(
                    subject,
                    trial,
                )
                activation_df = data_utils.create_df(ceinms_activation_path)
                activation_norm_df = self.time_normalise_df(activation_df)

                # Prepare data dictionary for helper method
                data_dict = {
                    "ik_df": ik_norm_df,
                    "id_df": id_norm_df,
                    "norm_force_df": norm_force_df,
                    "strain_df": strain_norm_df,
                    "emg_df": emg_norm_df,
                    "emg_adjusted_df": emg_adjusted_norm_df,
                    "fiber_lengths_df": ceinms_fiber_length_norm_df,
                    "tendon_lengths_df": ceinms_tendon_length_norm_df,
                    "fiber_velocity_df": ceinms_fiber_velocity_norm_df,
                    "fiber_force_df": ceinms_fiber_force_norm_df,
                    "active_force_df": ceinms_fiber_force_norm_df,
                    "passive_force_df": ceinms_passive_force_norm_df,
                    "activation_df": activation_norm_df,
                }

                # Build trial DataFrame using helper method
                trial_df = self._build_trial_dataframe(
                    subject=subject,
                    trial=trial,
                    leg=leg,
                    data_dict=data_dict,
                    muscles_config=muscles_config,
                    num_rows=101,
                )

                all_results.append(trial_df)

        master_df = pd.concat(all_results, ignore_index=True)
        return master_df

    def get_master_results_df(self, time_normalized=True) -> pd.DataFrame:
        if time_normalized:
            master_df_path = Path().cwd() / "data" / "master_results.csv"
        else:
            master_df_path = Path().cwd() / "data" / "master_results_no_time_norm.csv"

        if not master_df_path.exists():
            raise FileNotFoundError(
                f"Master results file not found: {master_df_path}. Please run create_master_results_df() first.",
            )

        master_df = pd.read_csv(master_df_path)
        return master_df

    def create_master_results_df_no_time_norm(self):
        """
        Create master results dataframe without time normalization.
        Preserves original time series data with variable lengths per trial.
        """
        all_results = []
        subject_utils = Subject_utils()
        all_subjects = subject_utils.get_all_subjects()

        for subject in all_subjects:
            leg = subject_utils.get_measured_leg(subject)
            trials = subject_utils.get_all_relevant_trials(subject)
            file_utils = File_utils()
            data_utils = Data_utils()

            for trial in trials:
                ik_file_path = file_utils.get_ik_file_path(subject, trial)
                if not ik_file_path.exists():
                    print(
                        f"Warning: IK file not found for subject {subject}, trial {trial}. Skipping.",
                    )
                    continue

                # IK - no normalization
                ik_df = data_utils.create_df(ik_file_path)

                # ID
                id_file_path = file_utils.get_id_file_path(subject, trial)
                if id_file_path is None or not id_file_path.exists():
                    print(
                        f"Warning: ID file not found for subject {subject}, trial {trial}. Skipping.",
                    )
                    continue

                id_df = data_utils.create_df(id_file_path)
                # Normalise by body mass
                subject_mass = subject_utils.get_body_mass(subject)
                id_norm_df = id_df.copy()
                for col in id_norm_df.columns:
                    if col == "time":
                        continue
                    id_norm_df[col] = id_df[col] / subject_mass

                # Get muscle configuration - use all muscles for master results
                muscles_config = self._get_all_muscles_config(leg)

                # Normalised force - no time normalization
                force_df = self.get_norm_forces_df(subject, trial)

                # Strain - no time normalization
                strain_df = self.get_muscle_strain_df(subject, trial)
                # Apply low-pass filter to all muscles
                # strain_df = self._apply_lowpass_filter_to_muscles(
                #     strain_df, muscles_config
                # )

                # ceinms fiber velocity - normalized by optimal fiber length, no time normalization
                ceinms_fiber_velocity_df = self.get_norm_fiber_velocity_df(
                    subject,
                    trial,
                )
                # Apply low-pass filter to all muscles
                # ceinms_fiber_velocity_df = self._apply_lowpass_filter_to_muscles(
                #     ceinms_fiber_velocity_df, muscles_config
                # )

                # ceinms fiber force - no time normalization
                ceinms_fiber_force_file = file_utils.get_ceinms_active_muscle_forces_file_path(subject, trial)
                ceinms_fiber_force_df = data_utils.create_df(ceinms_fiber_force_file)

                # ceinms passive muscle force - no time normalization
                try:
                    ceinms_passive_force_file = file_utils.get_ceinms_passive_muscle_forces_file_path(subject, trial)
                    ceinms_passive_force_df = data_utils.create_df(ceinms_passive_force_file)
                except FileNotFoundError:
                    ceinms_passive_force_df = None
                # Apply low-pass filter to all muscles
                # ceinms_fiber_force_df = self._apply_lowpass_filter_to_muscles(
                #     ceinms_fiber_force_df, muscles_config
                # )

                # ceinms normalized fiber lengths - no time normalization
                ceinms_fiber_length_file = file_utils.get_ceinms_normalized_fiber_lengths_file_path(
                    subject,
                    trial,
                )
                ceinms_fiber_length_df = data_utils.create_df(ceinms_fiber_length_file)
                # ceinms_fiber_length_df = self._apply_lowpass_filter_to_muscles(
                #     ceinms_fiber_length_df, muscles_config
                # )

                activation_path = file_utils.get_muscle_activation_file_path(
                    subject,
                    trial,
                )
                activation_df = data_utils.create_df(activation_path)

                # Check that all dataframes have the same length
                lengths = {
                    "ik_df": len(ik_df),
                    "id_norm_df": len(id_norm_df),
                    "force_df": len(force_df),
                    "strain_df": len(strain_df),
                    "ceinms_fiber_velocity_df": len(ceinms_fiber_velocity_df),
                    "ceinms_fiber_force_df": len(ceinms_fiber_force_df),
                    "ceinms_fiber_length_df": len(ceinms_fiber_length_df),
                }
                if ceinms_passive_force_df is not None:
                    lengths["ceinms_passive_force_df"] = len(ceinms_passive_force_df)

                unique_lengths = set(lengths.values())
                if len(unique_lengths) > 1:
                    print(
                        f"Warning: Dataframes have different lengths for {subject}, {trial}:",
                    )
                    for name, length in lengths.items():
                        print(f"  {name}: {length}")

                # Determine the minimum length across all dataframes
                min_length = min(lengths.values())

                # Prepare data dictionary for helper method
                data_dict = {
                    "ik_df": ik_df,
                    "id_df": id_norm_df,
                    "norm_force_df": force_df,
                    "strain_df": strain_df,
                    "emg_df": None,  # No raw EMG in non-time-normalized version
                    "emg_adjusted_df": None,
                    "fiber_lengths_df": ceinms_fiber_length_df,
                    "fiber_velocity_df": ceinms_fiber_velocity_df,
                    "fiber_force_df": ceinms_fiber_force_df,
                    "active_force_df": ceinms_fiber_force_df,
                    "passive_force_df": ceinms_passive_force_df,
                    "activation_df": activation_df,
                }

                # Build trial DataFrame using helper method
                trial_df = self._build_trial_dataframe(
                    subject=subject,
                    trial=trial,
                    leg=leg,
                    data_dict=data_dict,
                    muscles_config=muscles_config,
                    num_rows=min_length,
                )

                all_results.append(trial_df)

        master_df = pd.concat(all_results, ignore_index=True)
        return master_df

    def get_master_results_df_no_time_norm(self) -> pd.DataFrame:
        file_utils = File_utils()
        master_df_path = file_utils.get_master_results_non_normalized_file_path()

        master_df = pd.read_csv(master_df_path)
        return master_df

    def create_split_results(self):
        master_df_path = Path().cwd() / "data" / "master_results.csv"

        if not master_df_path.exists():
            raise FileNotFoundError(
                f"Master results file not found: {master_df_path}. Please run create_master_results_df() first.",
            )

        master_df = pd.read_csv(master_df_path)
        unique_trials = sorted(master_df["trial"].unique())

        # Columns to exclude from parameter processing
        excluded_cols = ["subject_id", "trial", "index", "time", "leg"]

        # Get all parameter columns
        parameter_cols = [col for col in master_df.columns if col not in excluded_cols]

        output_dir_lf = Path().cwd() / "data" / "split_results" / "long_format"
        if not output_dir_lf.exists():
            output_dir_lf.mkdir(parents=True, exist_ok=True)

        output_dir_wf = Path().cwd() / "data" / "split_results" / "wide_format"
        if not output_dir_wf.exists():
            output_dir_wf.mkdir(parents=True, exist_ok=True)

        # Iterate through each parameter
        for param in parameter_cols:
            # Iterate through each trial
            for trial in unique_trials:
                # Filter master_df for this trial
                trial_data = master_df[master_df["trial"] == trial]

                if trial_data.empty:
                    print(f"Warning: No data found for trial '{trial}'. Skipping.")
                    continue

                # Create subject column names mapping
                trial_data_copy = trial_data.copy()
                trial_data_copy["subject_col"] = trial_data_copy["subject_id"].apply(
                    lambda x: f"subject_{str(x).zfill(3)}",
                )

                # Check if all subjects have 101 rows
                subject_counts = trial_data_copy.groupby("subject_id").size()
                invalid_subjects = subject_counts[subject_counts != 101]
                if not invalid_subjects.empty:
                    for subj, count in invalid_subjects.items():
                        print(
                            f"Warning: Subject {subj}, trial {trial} has {count} rows instead of 101",
                        )

                # Use pivot to reshape: index becomes row numbers, columns become subject IDs
                split_df = trial_data_copy.sort_values(["subject_id", "index"]).pivot(
                    index="index",
                    columns="subject_col",
                    values=param,
                )

                # Reset index to make 'index' a column
                split_df = split_df.reset_index()

                # Reorder columns to ensure consistency (index first, then sorted subject columns)
                cols = ["index"] + sorted(
                    [col for col in split_df.columns if col != "index"],
                )
                split_df = split_df[cols]

                # save the split_df to a CSV file
                split_df.to_csv(output_dir_lf / f"{trial}_{param}.csv", index=False)

        # Save split df to csv in wide format
        for file in output_dir_lf.glob("*.csv"):
            df = pd.read_csv(file)
            df_transpose = df.transpose()
            df_transpose.to_csv(output_dir_wf / f"{file.stem}.csv", header=False)

    def butter_lowpass_filter(self, data, cutoff, fs, order):
        from scipy.signal import butter, lfilter

        nyq = 0.5 * fs

        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)  # type: ignore
        y = lfilter(b, a, data)
        return y

    def prep_pre_df(self, df=None) -> pd.DataFrame:
        if df is None:
            master_df = pd.read_csv(Path().cwd() / "data" / "master_results.csv")
        else:
            master_df = df
        pre_trials = ["Run_baselineA1", "Run_baselineB1"]

        pre_df = master_df[master_df["trial"].isin(pre_trials)]

        cols_to_process = [col for col in pre_df.columns if col not in ["subject_id", "trial", "index", "time", "leg"]]
        # Build a compact copy to avoid grouping a highly fragmented frame.
        group_cols = ["subject_id", "index"] + cols_to_process
        pre_group_source = pre_df.loc[:, group_cols].copy()
        pre_grouped = pre_group_source.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()

        metadata = pre_df[["subject_id", "trial", "index", "time", "leg"]].drop_duplicates(
            subset=["subject_id", "index"]
        )

        # Merge and set trial name to reflect combined baseline
        pre_prep_df = metadata.merge(
            pre_grouped,
            on=["subject_id", "index"],
            how="left",
        )
        pre_prep_df["trial"] = "Run_baseline"

        return pre_prep_df

    def prep_post_df(self, df=None) -> pd.DataFrame:
        if df is None:
            master_df = pd.read_csv(Path().cwd() / "data" / "master_results.csv")
        else:
            master_df = df

        post_trial = ["RunL1"]

        post_prep_df = master_df[master_df["trial"].isin(post_trial)].copy()

        return post_prep_df

    def spm_analysis(self, p=0.0042) -> dict:
        results_path = Path("data/split_results/wide_format")
        if not results_path.exists():
            raise FileNotFoundError(
                f"Results path not found: {results_path}. Please run create_split_results() first.",
            )
        # Read master_results to get actual column names
        master_df = self.get_master_results_df()

        all_columns = master_df.columns.tolist()

        # Exclude metadata columns
        excluded_cols = ["subject_id", "trial", "index", "time", "leg", "pelvis_tx"]

        # Build parameters list from actual columns (exclude metadata and power columns)
        parameters = [col for col in all_columns if col not in excluded_cols and not col.endswith("_power")]

        # Initialize results dictionary with all parameters set to None
        parameters_results: dict = dict.fromkeys(parameters)

        p_values = []

        for param in parameters:
            to_merge = []

            # Collect all matching files first
            for file in results_path.glob("*.csv"):
                if "Run_baseline" in file.name and file.name.endswith(f"_{param}.csv"):
                    to_merge.append(file)

            # Now process the collected files
            if len(to_merge) == 0:
                print(
                    f"Warning: No baseline files found for parameter '{param}'. Skipping.",
                )
                continue
            if len(to_merge) > 2:
                raise Exception(
                    f"Too many baseline files found for parameter '{param}': {[f.name for f in to_merge]}. Please check the split results.",
                )
            if len(to_merge) == 2:
                # Read both baseline files
                df1 = pd.read_csv(to_merge[0], index_col=0)
                df2 = pd.read_csv(to_merge[1], index_col=0)

                # Average corresponding cells (same subject, same time point)
                pre_mean_df = pd.concat([df1, df2]).groupby(level=0).mean()
            else:
                # Only one baseline file exists
                pre_mean_df = pd.read_csv(to_merge[0], header=None, index_col=0)

            # Copy second column to first if strain file
            if "strain" in param:
                pre_mean_df.iloc[:, 0] = pre_mean_df.iloc[:, 1]
            save_path = results_path.parent / "spm" / f"spm_pre_{param}.csv"
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
            pre_mean_df.to_csv(save_path, index=False, header=False)

            # Now read the post trial file
            post_file = results_path / f"RunL1_{param}.csv"
            if not post_file.exists():
                raise FileNotFoundError(
                    f"Post trial file not found for parameter '{param}': {post_file}",
                )
            post_df = pd.read_csv(post_file, index_col=0)

            # Match subjects between PRE and POST
            common_subjects = pre_mean_df.index.intersection(post_df.index)
            if len(common_subjects) == 0:
                raise ValueError(
                    f"No common subjects found between PRE and POST for parameter '{param}'",
                )

            pre_matched = pre_mean_df.loc[common_subjects]
            post_matched = post_df.loc[common_subjects]

            if "strain" in param or "velocity" in param:
                pre_matched.iloc[:, 0] = pre_matched.iloc[:, 1]
                post_matched.iloc[:, 0] = post_matched.iloc[:, 1]

            save_path_post = results_path.parent / "spm" / f"spm_post_{param}.csv"

            if not save_path_post.parent.exists():
                save_path_post.parent.mkdir(parents=True, exist_ok=True)
            post_matched.to_csv(save_path_post, index=False, header=False)

            # Check for zero variance at each time point - skip SPM if any time point has zero variance
            pre_variance_per_timepoint = np.var(pre_matched.values, axis=0)
            post_variance_per_timepoint = np.var(post_matched.values, axis=0)

            if np.any(pre_variance_per_timepoint == 0) or np.any(
                post_variance_per_timepoint == 0,
            ):
                zero_nodes_pre = np.where(pre_variance_per_timepoint == 0)[0]
                zero_nodes_post = np.where(post_variance_per_timepoint == 0)[0]
                print(
                    f"Warning: Zero variance detected for parameter '{param}' at nodes {list(zero_nodes_pre) if len(zero_nodes_pre) > 0 else list(zero_nodes_post)}. Skipping SPM analysis.",
                )
                parameters_results[param] = None
                continue

            # Run SPM analysis on matched subjects
            t_paired = spm.stats.ttest_paired(pre_matched.values, post_matched.values)
            ti = t_paired.inference(p, two_tailed=True)
            parameters_results[param] = ti

            p_values.append(ti.p)

        return parameters_results

    def get_spm_sig_index(self, p, spm_results=None):
        if spm_results is None:
            spm_results = self.spm_analysis(p=p)
        sig_indices = {}
        for param, ti in spm_results.items():
            if ti is None:
                sig_indices[param] = []
                continue
            sig_index = abs(ti.z) > ti.zstar
            sig_indices[param] = sig_index
        return sig_indices

    def spm_analysis_moments_vs_torques(self, p=0.0042) -> dict:
        """
        Perform SPM analysis comparing ID moments vs simulated torques for each joint.
        Tests if the two measurement methods differ significantly at each point in the gait cycle.
        Returns results for both PRE and POST trials separately.
        """
        torques_df = self.create_torques_results_df()

        # Consolidate pre-trial data
        pre_trials = ["Run_baselineA1", "Run_baselineB1"]
        post_trials = ["RunL1"]

        pre_df = torques_df[torques_df["trial"].isin(pre_trials)].copy()
        post_df = torques_df[torques_df["trial"].isin(post_trials)].copy()

        # Group by subject and index to get mean for each subject at each timepoint
        cols_to_process = [
            "hip_flexion_id_moment",
            "hip_flexion_torque",
            "knee_angle_id_moment",
            "knee_angle_torque",
            "ankle_angle_id_moment",
            "ankle_angle_torque",
        ]

        # Build compact copies to avoid grouping highly fragmented frames.
        pre_group_source = pre_df.loc[:, ["subject_id", "index"] + cols_to_process].copy()
        post_group_source = post_df.loc[:, ["subject_id", "index"] + cols_to_process].copy()
        pre_grouped = pre_group_source.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()
        post_grouped = post_group_source.groupby(["subject_id", "index"])[cols_to_process].mean().reset_index()

        results = {}

        # Test each joint for PRE condition
        for id_col, torque_col, joint_name in [
            ("hip_flexion_id_moment", "hip_flexion_torque", "hip_flexion"),
            ("knee_angle_id_moment", "knee_angle_torque", "knee_angle"),
            ("ankle_angle_id_moment", "ankle_angle_torque", "ankle_angle"),
        ]:
            # PRE: Reshape data to (101 timepoints, n_subjects)
            pre_id_pivot = pre_grouped.pivot(
                index="index",
                columns="subject_id",
                values=id_col,
            )
            pre_torque_pivot = pre_grouped.pivot(
                index="index",
                columns="subject_id",
                values=torque_col,
            )

            # POST: Reshape data to (101 timepoints, n_subjects)
            post_id_pivot = post_grouped.pivot(
                index="index",
                columns="subject_id",
                values=id_col,
            )
            post_torque_pivot = post_grouped.pivot(
                index="index",
                columns="subject_id",
                values=torque_col,
            )

            # Match subjects between PRE and POST
            common_subjects = pre_id_pivot.columns.intersection(post_id_pivot.columns)

            if len(common_subjects) == 0:
                results[f"{joint_name}_pre"] = None
                results[f"{joint_name}_post"] = None
                continue

            # Get matched data for PRE
            pre_id_data = pre_id_pivot[common_subjects].values
            pre_torque_data = pre_torque_pivot[common_subjects].values

            # Remove any columns with NaN for PRE
            valid_cols_pre = ~(np.isnan(pre_id_data).any(axis=0) | np.isnan(pre_torque_data).any(axis=0))
            pre_id_data = pre_id_data[:, valid_cols_pre]
            pre_torque_data = pre_torque_data[:, valid_cols_pre]

            if pre_id_data.shape[1] > 0:
                # Paired t-test: ID moment vs simulated torque for PRE
                t_paired_pre = spm.stats.ttest_paired(pre_id_data.T, pre_torque_data.T)
                ti_pre = t_paired_pre.inference(0.05, two_tailed=True)
                results[f"{joint_name}_pre"] = ti_pre
            else:
                results[f"{joint_name}_pre"] = None

            # Get matched data for POST
            post_id_data = post_id_pivot[common_subjects].values
            post_torque_data = post_torque_pivot[common_subjects].values

            # Remove any columns with NaN for POST
            valid_cols_post = ~(np.isnan(post_id_data).any(axis=0) | np.isnan(post_torque_data).any(axis=0))
            post_id_data = post_id_data[:, valid_cols_post]
            post_torque_data = post_torque_data[:, valid_cols_post]

            if post_id_data.shape[1] > 0:
                # Paired t-test: ID moment vs simulated torque for POST
                t_paired_post = spm.stats.ttest_paired(
                    post_id_data.T,
                    post_torque_data.T,
                )
                ti_post = t_paired_post.inference(0.0166, two_tailed=True)
                results[f"{joint_name}_post"] = ti_post
            else:
                results[f"{joint_name}_post"] = None

        return results

    def get_spm_sig_index_moments_vs_torques(self, spm_results=None):
        """
        Get significance indices for moments vs torques SPM analysis.
        """
        if spm_results is None:
            spm_results = self.spm_analysis_moments_vs_torques()
        sig_indices = {}
        for param, ti in spm_results.items():
            if ti is None:
                sig_indices[param] = None
                continue
            sig_index = abs(ti.z) > ti.zstar
            sig_indices[param] = sig_index
        return sig_indices

    def get_peak_abs_strain_df(self, muscles: list | None = None) -> pd.DataFrame:
        """
        Calculate peak absolute strain for specified muscles.

        Args:
            muscles: List of muscle base names (can be full names or abbreviations).
                    If None, defaults to hamstring muscles: ['bflh', 'semiten', 'semimem']

        Returns:
            DataFrame with peak absolute strain pre, post, and change for each muscle

        """
        pre_df = self.prep_pre_df()
        post_df = self.prep_post_df()

        subject_ids = pre_df["subject_id"].unique()
        if subject_ids.tolist() != post_df["subject_id"].unique().tolist():
            raise ValueError("Subject IDs in pre and post data do not match.")

        # Mapping from full names to abbreviations used in column names
        name_mapping = {
            "biceps_femoris_long_head": "bflh",
            "semitendinosus": "semiten",
            "semimembranosus": "semimem",
        }

        # Default to hamstring muscles if not specified
        if muscles is None:
            muscles = ["bflh", "semiten", "semimem"]

        # Build column names dynamically
        columns = ["subject_id"]
        for muscle in muscles:
            columns.extend(
                [
                    f"{muscle}_peak_abs_strain_pre",
                    f"{muscle}_peak_abs_strain_post",
                    f"{muscle}_peak_abs_strain_change",
                ],
            )

        peak_abs_strain_df = pd.DataFrame({"subject_id": subject_ids}, columns=columns)

        # Calculate peak absolute strains for each muscle
        for muscle in muscles:
            # Map full name to abbreviation if needed
            muscle_abbrev = name_mapping.get(muscle, muscle)
            muscle_col = f"{muscle_abbrev}_abs_strain"

            # Vectorized: group by subject_id and get max of absolute values
            pre_peaks = pre_df.groupby("subject_id")[muscle_col].max()
            post_peaks = post_df.groupby("subject_id")[muscle_col].max()

            # Align with subject_ids order
            peak_abs_strain_df[f"{muscle}_peak_abs_strain_pre"] = pre_peaks.reindex(
                subject_ids,
            ).values
            peak_abs_strain_df[f"{muscle}_peak_abs_strain_post"] = post_peaks.reindex(
                subject_ids,
            ).values
            peak_abs_strain_df[f"{muscle}_peak_abs_strain_change"] = (
                peak_abs_strain_df[f"{muscle}_peak_abs_strain_post"]
                - peak_abs_strain_df[f"{muscle}_peak_abs_strain_pre"]
            )
        return peak_abs_strain_df

    def get_peak_norm_fiber_length_df(self, muscles: list | None = None) -> pd.DataFrame:
        """
        Calculate peak normalized fiber length for specified muscles.

        Args:
            muscles: List of muscle base names (can be full names or abbreviations).
                    If None, defaults to hamstring muscles: ['bflh', 'semiten', 'semimem']

        Returns:
            DataFrame with peak normalized fiber length pre, post, and change for each muscle

        """
        pre_df = self.prep_pre_df()
        post_df = self.prep_post_df()

        subject_ids = pre_df["subject_id"].unique()
        if subject_ids.tolist() != post_df["subject_id"].unique().tolist():
            raise ValueError("Subject IDs in pre and post data do not match.")

        # Mapping from full names to abbreviations used in column names
        name_mapping = {
            "biceps_femoris_long_head": "bflh",
            "semitendinosus": "semiten",
            "semimembranosus": "semimem",
        }

        # Default to hamstring muscles if not specified
        if muscles is None:
            muscles = ["bflh", "semiten", "semimem"]

        # Build column names dynamically
        columns = ["subject_id"]
        for muscle in muscles:
            columns.extend(
                [
                    f"{muscle}_peak_norm_fiber_length_pre",
                    f"{muscle}_peak_norm_fiber_length_post",
                    f"{muscle}_peak_norm_fiber_length_change",
                ],
            )

        peak_norm_fiber_length_df = pd.DataFrame({"subject_id": subject_ids}, columns=columns)

        # Calculate peak normalized fiber lengths for each muscle
        for muscle in muscles:
            # Map full name to abbreviation if needed
            muscle_abbrev = name_mapping.get(muscle, muscle)
            muscle_col = f"{muscle_abbrev}_fiber_lengths"

            # Vectorized: group by subject_id and get max values
            pre_peaks = pre_df.groupby("subject_id")[muscle_col].apply(
                lambda x: x.max(),
            )
            post_peaks = post_df.groupby("subject_id")[muscle_col].apply(
                lambda x: x.max(),
            )

            # Align with subject_ids order
            peak_norm_fiber_length_df[f"{muscle}_peak_norm_fiber_length_pre"] = pre_peaks.reindex(
                subject_ids,
            ).values
            peak_norm_fiber_length_df[f"{muscle}_peak_norm_fiber_length_post"] = post_peaks.reindex(
                subject_ids,
            ).values
            peak_norm_fiber_length_df[f"{muscle}_peak_norm_fiber_length_change"] = (
                peak_norm_fiber_length_df[f"{muscle}_peak_norm_fiber_length_post"]
                - peak_norm_fiber_length_df[f"{muscle}_peak_norm_fiber_length_pre"]
            )
        return peak_norm_fiber_length_df

    def get_peak_strain_df(self, muscles: list | None = None) -> pd.DataFrame:
        """Backward-compatible wrapper for peak normalized fiber length values.

        Returns the same values as get_peak_norm_fiber_length_df but with legacy
        column names containing "peak_strain".
        """
        peak_norm_fiber_length_df = self.get_peak_norm_fiber_length_df(muscles=muscles).copy()
        rename_map = {
            col: col.replace("peak_norm_fiber_length", "peak_strain")
            for col in peak_norm_fiber_length_df.columns
            if "peak_norm_fiber_length" in col
        }
        return peak_norm_fiber_length_df.rename(columns=rename_map)

    def get_peak_norm_fiber_length_summary_stats(
        self,
        muscles: list | None = None,
    ) -> pd.DataFrame:
        """
        Calculate percentage change from group means, along with standard deviation
        and 95% confidence interval of the absolute differences.

        Args:
            muscles: List of muscle base names. If None, defaults to hamstring muscles.

        Returns:
            DataFrame with statistics for each muscle including:
            - mean_pre: mean pre-fatigue peak strain across subjects
            - mean_post: mean post-fatigue peak strain across subjects
            - mean_diff_percent: percentage change from mean pre to mean post
            - sd_diff: standard deviation of absolute differences
            - ci_lower_percent: lower bound of 95% CI for percentage change
            - ci_upper_percent: upper bound of 95% CI for percentage change

        """
        from scipy import stats

        # Default to hamstring muscles if not specified
        if muscles is None:
            muscles = ["bflh", "semiten", "semimem"]

        peak_norm_fiber_length_df = self.get_peak_norm_fiber_length_df(muscles=muscles)

        summary_stats = []

        for muscle in muscles:
            pre_col = f"{muscle}_peak_norm_fiber_length_pre"
            post_col = f"{muscle}_peak_norm_fiber_length_post"
            change_col = f"{muscle}_peak_norm_fiber_length_change"

            # Calculate percentage change for each individual subject
            pct_changes = (
                (peak_norm_fiber_length_df[post_col] - peak_norm_fiber_length_df[pre_col])
                / peak_norm_fiber_length_df[pre_col]
            ) * 100

            # Calculate mean and SD of individual percentage changes
            mean_change_percent = pct_changes.mean()
            sd_percent = pct_changes.std(ddof=1)

            # Calculate mean pre and post values for reference
            mean_pre = peak_norm_fiber_length_df[pre_col].mean()
            mean_post = peak_norm_fiber_length_df[post_col].mean()

            # Calculate SD of absolute differences
            sd_change = peak_norm_fiber_length_df[change_col].std(ddof=1)

            # Calculate SE and CI based on percentage changes
            n = len(peak_norm_fiber_length_df)
            se_percent = sd_percent / np.sqrt(n)

            # Calculate 95% CI for percentage changes
            ci_percent = stats.t.interval(
                0.95,
                n - 1,
                loc=mean_change_percent,
                scale=se_percent,
            )

            ci_lower_percent = ci_percent[0]
            ci_upper_percent = ci_percent[1]

            summary_stats.append(
                {
                    "muscle": muscle,
                    "mean_pre": mean_pre,
                    "mean_post": mean_post,
                    "mean_change_percent": mean_change_percent,
                    "sd_change": sd_change,
                    "sd_percent": sd_percent,
                    "ci_lower_percent": ci_lower_percent,
                    "ci_upper_percent": ci_upper_percent,
                    "n_subjects": n,
                },
            )

        return pd.DataFrame(summary_stats)

    def get_peak_strain_summary_stats(
        self,
        muscles: list | None = None,
    ) -> pd.DataFrame:
        """Backward-compatible wrapper for normalized fiber length summary stats."""
        return self.get_peak_norm_fiber_length_summary_stats(muscles=muscles)

    def get_peak_norm_fiber_length_responders_summary_stats(
        self,
        muscles: list | None = None,
    ) -> pd.DataFrame:
        """
        Calculate summary statistics for peak strain responders only.

        Args:
            muscles: List of muscle base names. If None, defaults to hamstring muscles.

        Returns:
            DataFrame with statistics for each muscle including:
            - mean_pre: mean pre-fatigue peak strain across
            responders subjects
            - mean_post: mean post-fatigue peak strain across responders subjects
            - mean_change_percent: percentage change from mean pre to mean post
            - sd_change: standard deviation of absolute differences
            - ci_lower_percent: lower bound of 95% CI for percentage change
            - ci_upper_percent: upper bound of 95% CI for percentage change

        """
        from scipy import stats

        # Default to hamstring muscles if not specified
        if muscles is None:
            muscles = ["bflh", "semiten", "semimem"]

        peak_norm_fiber_length_df = self.get_peak_norm_fiber_length_df(muscles=muscles)
        responders = self.get_norm_fiber_length_responders_ids(muscles=muscles)
        summary_stats = []
        for muscle in muscles:
            pre_col = f"{muscle}_peak_norm_fiber_length_pre"
            post_col = f"{muscle}_peak_norm_fiber_length_post"
            change_col = f"{muscle}_peak_norm_fiber_length_change"

            responder_ids = responders[muscle]
            responder_df = peak_norm_fiber_length_df[peak_norm_fiber_length_df["subject_id"].isin(responder_ids)]

            # Calculate percentage change for each responder subject
            pct_changes = (responder_df[post_col] - responder_df[pre_col]) / responder_df[pre_col] * 100

            # Calculate mean and SD of individual percentage changes
            mean_change_percent = pct_changes.mean()
            sd_percent = pct_changes.std(ddof=1)

            # Calculate mean pre and post values for reference
            mean_pre = responder_df[pre_col].mean()
            mean_post = responder_df[post_col].mean()

            # Calculate SD of absolute differences
            sd_change = responder_df[change_col].std(ddof=1)

            # Calculate SE and CI based on percentage changes
            n = len(responder_df)
            se_percent = sd_percent / np.sqrt(n)

            # Calculate 95% CI for percentage changes
            ci_percent = stats.t.interval(
                0.95,
                n - 1,
                loc=mean_change_percent,
                scale=se_percent,
            )

            ci_lower_percent = ci_percent[0]
            ci_upper_percent = ci_percent[1]

            summary_stats.append(
                {
                    "muscle": muscle,
                    "mean_pre": mean_pre,
                    "mean_post": mean_post,
                    "mean_change_percent": mean_change_percent,
                    "sd_change": sd_change,
                    "sd_percent": sd_percent,
                    "ci_lower_percent": ci_lower_percent,
                    "ci_upper_percent": ci_upper_percent,
                    "n_subjects": n,
                },
            )

        return pd.DataFrame(summary_stats)

    def get_peak_strain_responders_summary_stats(
        self,
        muscles: list | None = None,
    ) -> pd.DataFrame:
        """Backward-compatible wrapper for normalized fiber length responder summary stats."""
        return self.get_peak_norm_fiber_length_responders_summary_stats(muscles=muscles)

    def get_peak_norm_force(self, muscles: list | None = None) -> pd.DataFrame:
        """
        Calculate peak normalized force for specified muscles.

        Args:
            muscles: List of muscle base names (without _norm_force suffix).
                    If None, defaults to hamstring muscles: ['bflh', 'semiten', 'semimem']

        Returns:
            DataFrame with peak normalized force pre and post for each muscle

        """
        pre_df = self.prep_pre_df()
        post_df = self.prep_post_df()

        subject_ids = pre_df["subject_id"].unique()
        if subject_ids.tolist() != post_df["subject_id"].unique().tolist():
            raise ValueError("Subject IDs in pre and post data do not match.")

        # Default to hamstring muscles if not specified
        if muscles is None:
            muscles = ["bflh", "semiten", "semimem"]

        # Build column names dynamically
        columns = ["subject_id"]
        for muscle in muscles:
            columns.extend(
                [
                    f"{muscle}_peak_norm_force_pre",
                    f"{muscle}_peak_norm_force_post",
                ],
            )

        peak_norm_force_df = pd.DataFrame({"subject_id": subject_ids}, columns=columns)

        # Calculate peak forces for each muscle
        for muscle in muscles:
            muscle_col = f"{muscle}_norm_force"

            # Vectorized: group by subject_id and get max of absolute values
            pre_peaks = pre_df.groupby("subject_id")[muscle_col].apply(
                lambda x: x.abs().max(),
            )
            post_peaks = post_df.groupby("subject_id")[muscle_col].apply(
                lambda x: x.abs().max(),
            )

            # Align with subject_ids order
            peak_norm_force_df[f"{muscle}_peak_norm_force_pre"] = pre_peaks.reindex(
                subject_ids,
            ).values
            peak_norm_force_df[f"{muscle}_peak_norm_force_post"] = post_peaks.reindex(
                subject_ids,
            ).values

        return peak_norm_force_df

    def get_activation_df(self, id: str, trial: str) -> pd.DataFrame:
        file_utils = File_utils()
        activation_file = file_utils.get_muscle_activation_file_path(id, trial)
        activation_df = Data_utils().create_df(activation_file)
        return activation_df

    def get_ceinms_adjusted_emg_df(self, id: str, trial: str) -> pd.DataFrame:
        file_utils = File_utils()
        adjusted_emg_file = file_utils.get_ceinms_adjusted_emgs_file_path(id, trial)
        adjusted_emg_df = Data_utils().create_df(adjusted_emg_file)

        return adjusted_emg_df

    def get_emg_df(self, id: str, trial: str) -> pd.DataFrame:
        file_utils = File_utils()
        emg_file = file_utils.get_emg_file_path(id, trial)
        emg_df = Data_utils().create_df(emg_file)

        return emg_df

    def run_correlation_analysis(self, y1, y2):
        return "bro"

    def get_rmse(self, y1, y2):
        return np.sqrt(np.mean((y1 - y2) ** 2))

    def get_r_squared(self, y1, y2):
        return np.corrcoef(y1, y2)[0, 1] ** 2

    def create_torques_results_df(self):
        torque_results = []

        subject_utils = Subject_utils()
        all_subjects = subject_utils.get_all_subjects()
        for subject in all_subjects:
            trials = []
            leg = subject_utils.get_measured_leg(subject)

            trials = subject_utils.get_all_relevant_trials(subject)

            for trial in trials:
                torque_path = File_utils().get_ceinms_torque_file_path(subject, trial)
                if not torque_path.exists():
                    raise FileNotFoundError(
                        f"Torque file not found for subject {subject}, trial {trial}",
                    )

                id_df = Data_utils().create_df(
                    File_utils().get_id_file_path(subject, trial),
                )
                id_norm_df_time = self.time_normalise_df(id_df)

                # Normalize by body mass
                subject_mass = subject_utils.get_body_mass(subject)
                id_norm_df = id_norm_df_time.copy()
                for col in id_norm_df.columns:
                    if col == "time":
                        continue
                    id_norm_df[col] = id_norm_df_time[col] / subject_mass

                torque_df = Data_utils().create_df(torque_path)
                torque_norm_df_time = self.time_normalise_df(torque_df)
                torque_norm_df = torque_norm_df_time.copy()
                for col in torque_norm_df.columns:
                    if col == "time":
                        continue
                    torque_norm_df[col] = torque_norm_df_time[col] / subject_mass

                hip_flexion_col = "hip_flexion_r_moment" if leg == "R" else "hip_flexion_l_moment"
                knee_angle_col = "knee_angle_r_moment" if leg == "R" else "knee_angle_l_moment"
                ankle_angle_col = "ankle_angle_r_moment" if leg == "R" else "ankle_angle_l_moment"

                hip_torque_col = "hip_flexion_r" if leg == "R" else "hip_flexion_l"
                knee_torque_col = "knee_angle_r" if leg == "R" else "knee_angle_l"
                ankle_torque_col = "ankle_angle_r" if leg == "R" else "ankle_angle_l"

                trial_df = pd.DataFrame(
                    {
                        "subject_id": [subject] * 101,
                        "trial": [trial] * 101,
                        "leg": [leg.lower()] * 101,
                        "index": torque_norm_df.index,
                        "time": torque_norm_df["time"].values,
                        "hip_flexion_torque": torque_norm_df[hip_torque_col].values,
                        "knee_angle_torque": torque_norm_df[knee_torque_col].values,
                        "ankle_angle_torque": torque_norm_df[ankle_torque_col].values,
                        "hip_flexion_id_moment": id_norm_df[hip_flexion_col].values,
                        "knee_angle_id_moment": id_norm_df[knee_angle_col].values,
                        "ankle_angle_id_moment": id_norm_df[ankle_angle_col].values,
                    },
                )
                torque_results.append(trial_df)

        torque_df = pd.concat(torque_results, ignore_index=True)
        return torque_df

    def create_torques_results_wide_df(self):
        torque_results_df = self.create_torques_results_df()

        unique_trials = sorted(torque_results_df["trial"].unique())

        excluded_cols = ["subject_id", "trial", "index", "time", "leg"]

        parameter_cols = [col for col in torque_results_df.columns if col not in excluded_cols]

        output_dir_lf = Path().cwd() / "data" / "split_results" / "long_format"
        if not output_dir_lf.exists():
            output_dir_lf.mkdir(parents=True, exist_ok=True)

        output_dir_wf = Path().cwd() / "data" / "split_results" / "wide_format"
        if not output_dir_wf.exists():
            output_dir_wf.mkdir(parents=True, exist_ok=True)

        # Iterate through each parameter
        for param in parameter_cols:
            # Iterate through each trial
            for trial in unique_trials:
                # Filter torque_results_df for this trial
                trial_data = torque_results_df[torque_results_df["trial"] == trial]

                if trial_data.empty:
                    print(f"Warning: No data found for trial '{trial}'. Skipping.")
                    continue

                # Create subject column names mapping
                trial_data_copy = trial_data.copy()
                trial_data_copy["subject_col"] = trial_data_copy["subject_id"].apply(
                    lambda x: f"subject_{str(x).zfill(3)}",
                )

                # Check if all subjects have 101 rows
                subject_counts = trial_data_copy.groupby("subject_id").size()
                invalid_subjects = subject_counts[subject_counts != 101]
                if not invalid_subjects.empty:
                    for subj, count in invalid_subjects.items():
                        print(
                            f"Warning: Subject {subj}, trial {trial} has {count} rows instead of 101",
                        )

                # Use pivot to reshape: index becomes row numbers, columns become subject IDs
                split_df = trial_data_copy.sort_values(["subject_id", "index"]).pivot(
                    index="index",
                    columns="subject_col",
                    values=param,
                )

                # Reset index to make 'index' a column
                split_df = split_df.reset_index()

                # Reorder columns to ensure consistency (index first, then sorted subject columns)
                cols = ["index"] + sorted(
                    [col for col in split_df.columns if col != "index"],
                )
                split_df = split_df[cols]

                # save the split_df to a CSV file
                split_df.to_csv(output_dir_lf / f"{trial}_{param}.csv", index=False)

    def get_muscle_parameters(self, muscles: list | None = None) -> pd.DataFrame:
        """
        Get muscle parameters from OpenSim and CEINMS models for specified muscles.

        Args:
            muscles: List of muscle base names (without _r/_l suffix).
                    If None, defaults to hamstring muscles: ['bflh', 'semiten', 'semimem']

        Returns:
            DataFrame with muscle parameters for each subject

        """
        file_utils = File_utils()
        all_subjects = Subject_utils().get_all_subjects()

        # Collect all rows in a list instead of repeatedly concatenating
        all_rows = []

        # Default to hamstring muscles if not specified
        if muscles is None:
            muscles = ["bflh", "semiten", "semimem"]

        for subject in all_subjects:
            leg = Subject_utils().get_measured_leg(subject)
            # opensim_model_file = file_utils.get_model_path(subject)
            opensim_model_file = (
                Path("data") / "models" / subject / f"{subject}_Rajagopal2015_FAI_originalMass_opt_N10_hans.osim"
            )
            if not opensim_model_file.exists():
                raise FileNotFoundError(
                    f"OpenSim model file not found for subject {subject}",
                )
            ceinms_cal_file = file_utils.get_ceinms_cal_model_file_path(subject)
            if not ceinms_cal_file.exists():
                raise FileNotFoundError(
                    f"CEINMS calibrated model file not found for subject {subject}",
                )

            ceinms_uncal_file = file_utils.get_ceinms_uncal_model_file_path(subject)
            if not ceinms_cal_file.exists() or not ceinms_uncal_file.exists():
                raise FileNotFoundError(
                    f"CEINMS model files not found for subject {subject}",
                )

            tree_opensim = ET.parse(opensim_model_file)
            root_opensim = tree_opensim.getroot()
            tree_uncal = ET.parse(ceinms_uncal_file)
            root_uncal = tree_uncal.getroot()
            tree_cal = ET.parse(ceinms_cal_file)
            root_cal = tree_cal.getroot()

            # Select muscles based on measured leg
            muscles_to_check = [f"{m}_{leg.lower()}" for m in muscles]

            # Initialize row data
            row_data = {"subject_id": subject}

            # Map muscle names to shorter names for column naming (remove _r/_l suffix)
            muscle_name_map = {f"{m}_{leg}": m for m in muscles for leg in ["r", "l"]}

            for muscle in muscles_to_check:
                # Get opensim values
                mtu = root_opensim.findall(".//Millard2012EquilibriumMuscle")
                if mtu is None:
                    raise ValueError(
                        f"ForceSet not found in OpenSim XML for subject {subject}",
                    )
                max_isometric_force_osim = None
                optimal_fibre_length_osim = None
                tendon_slack_length_osim = None

                for muscle_elem in mtu:
                    # In OpenSim XML, the muscle name is an attribute, not a child element
                    muscle_name = muscle_elem.get("name")
                    if muscle_name == muscle:
                        # OpenSim uses underscore naming convention
                        max_isometric_force_osim = muscle_elem.find(
                            "max_isometric_force",
                        )
                        optimal_fibre_length_osim = muscle_elem.find(
                            "optimal_fiber_length",
                        )
                        tendon_slack_length_osim = muscle_elem.find(
                            "tendon_slack_length",
                        )
                        break

                # Get uncalibrated values
                mtu_set_uncal = root_uncal.find("mtuSet")
                if mtu_set_uncal is None:
                    raise ValueError(
                        f"mtuSet not found in uncalibrated XML for subject {subject}",
                    )

                max_isometric_force_uncal = None
                optimal_fibre_length_uncal = None
                tendon_slack_length_uncal = None
                strength_coefficient_uncal = None

                for mtu in mtu_set_uncal.findall("mtu"):
                    muscle_name_elem = mtu.find("name")
                    if muscle_name_elem is not None and muscle_name_elem.text == muscle:
                        max_isometric_force_uncal = mtu.find("maxIsometricForce")
                        optimal_fibre_length_uncal = mtu.find("optimalFibreLength")
                        tendon_slack_length_uncal = mtu.find("tendonSlackLength")
                        strength_coefficient_uncal = mtu.find("strengthCoefficient")
                        break

                # Get calibrated values
                mtu_set_cal = root_cal.find("mtuSet")
                if mtu_set_cal is None:
                    raise ValueError(
                        f"mtuSet not found in calibrated XML for subject {subject}",
                    )
                max_isometric_force_cal = None
                optimal_fibre_length_cal = None
                tendon_slack_length_cal = None
                strength_coefficient_cal = None

                for mtu in mtu_set_cal.findall("mtu"):
                    muscle_name_elem = mtu.find("name")
                    if muscle_name_elem is not None and muscle_name_elem.text == muscle:
                        max_isometric_force_cal = mtu.find("maxIsometricForce")
                        optimal_fibre_length_cal = mtu.find("optimalFibreLength")
                        tendon_slack_length_cal = mtu.find("tendonSlackLength")
                        strength_coefficient_cal = mtu.find("strengthCoefficient")
                        break

                # Extract optimal fiber length values
                ofl_osim = (
                    float(optimal_fibre_length_osim.text)
                    if (optimal_fibre_length_osim is not None and optimal_fibre_length_osim.text is not None)
                    else None
                )
                ofl_cal = (
                    float(optimal_fibre_length_cal.text)
                    if (optimal_fibre_length_cal is not None and optimal_fibre_length_cal.text is not None)
                    else None
                )
                ofl_uncal = (
                    float(optimal_fibre_length_uncal.text)
                    if (optimal_fibre_length_uncal is not None and optimal_fibre_length_uncal.text is not None)
                    else None
                )

                # Extract tendon slack length values
                tsl_osim = (
                    float(tendon_slack_length_osim.text)
                    if (tendon_slack_length_osim is not None and tendon_slack_length_osim.text is not None)
                    else None
                )
                tsl_cal = (
                    float(tendon_slack_length_cal.text)
                    if (tendon_slack_length_cal is not None and tendon_slack_length_cal.text is not None)
                    else None
                )
                tsl_uncal = (
                    float(tendon_slack_length_uncal.text)
                    if (tendon_slack_length_uncal is not None and tendon_slack_length_uncal.text is not None)
                    else None
                )

                # Extract max isometric force values
                max_iso_force_osim = (
                    float(max_isometric_force_osim.text)
                    if (max_isometric_force_osim is not None and max_isometric_force_osim.text is not None)
                    else None
                )
                max_iso_force_cal = (
                    float(max_isometric_force_cal.text)
                    if (max_isometric_force_cal is not None and max_isometric_force_cal.text is not None)
                    else None
                )
                max_iso_force_uncal = (
                    float(max_isometric_force_uncal.text)
                    if (max_isometric_force_uncal is not None and max_isometric_force_uncal.text is not None)
                    else None
                )

                # Extract strength coefficient
                # Uncalibrated
                strength_coeff_uncal = (
                    float(strength_coefficient_uncal.text)
                    if (strength_coefficient_uncal is not None and strength_coefficient_uncal.text is not None)
                    else None
                )

                # Calibrated
                strength_coeff_cal = (
                    float(strength_coefficient_cal.text)
                    if (strength_coefficient_cal is not None and strength_coefficient_cal.text is not None)
                    else None
                )

                # Calculate max forces with strength coefficient applied

                max_iso_force_uncal_adjusted = (
                    max_iso_force_uncal * strength_coeff_uncal
                    if max_iso_force_uncal is not None and strength_coeff_uncal is not None
                    else None
                )
                max_iso_force_cal_adjusted = (
                    max_iso_force_cal * strength_coeff_cal
                    if max_iso_force_cal is not None and strength_coeff_cal is not None
                    else None
                )
                # Add to row data - grouped by source (OpenSim, Uncalibrated, Calibrated)
                # with consistent parameter order (optimal fiber length, tendon slack length, max force)
                muscle_short = muscle_name_map[muscle]

                # OpenSim parameters
                row_data[f"{muscle_short}_optimal_fibre_length_opensim"] = ofl_osim
                row_data[f"{muscle_short}_tendon_slack_length_opensim"] = tsl_osim
                row_data[f"{muscle_short}_max_isometric_force_opensim"] = max_iso_force_osim

                # Uncalibrated parameters
                row_data[f"{muscle_short}_optimal_fibre_length_uncalibrated"] = ofl_uncal
                row_data[f"{muscle_short}_tendon_slack_length_uncalibrated"] = tsl_uncal
                row_data[f"{muscle_short}_max_isometric_force_uncalibrated"] = max_iso_force_uncal_adjusted

                # Calibrated parameters
                row_data[f"{muscle_short}_optimal_fibre_length_calibrated"] = ofl_cal
                row_data[f"{muscle_short}_tendon_slack_length_calibrated"] = tsl_cal
                row_data[f"{muscle_short}_max_isometric_force_calibrated"] = max_iso_force_cal_adjusted

                # Strength coefficient
                row_data[f"{muscle_short}_strength_coefficient_uncalibrated"] = strength_coeff_uncal
                row_data[f"{muscle_short}_strength_coefficient_calibrated"] = strength_coeff_cal

            # Append row data to list
            all_rows.append(row_data)

        # Create DataFrame from all rows at once
        output_df = pd.DataFrame(all_rows)
        return output_df

    def get_norm_fiber_length_all_muscles_df(self):
        all_results = []
        subject_utils = Subject_utils()
        all_subjects = subject_utils.get_all_subjects()
        for subject in all_subjects:
            trials = []
            leg = subject_utils.get_measured_leg(subject)

            trials = subject_utils.get_all_relevant_trials(subject)

            file_utils = File_utils()

            data_utils = Data_utils()

            for trial in trials:
                norm_fiber_length_df = data_utils.create_df(
                    file_utils.get_ceinms_normalized_fiber_lengths_file_path(
                        subject,
                        trial,
                    ),
                )
                norm_fiber_length_norm_df = self.time_normalise_df(norm_fiber_length_df)

                muscles = [muscles for muscles in norm_fiber_length_norm_df.columns if muscles != "time"]

                # left or right leg muscles
                if leg == "R":
                    muscles = [muscle for muscle in muscles if "_r" in muscle.lower()]
                else:
                    muscles = [muscle for muscle in muscles if "_l" in muscle.lower()]

                # Create dictionary with renamed columns (remove _r or _l)
                muscle_data = {}
                for muscle in muscles:
                    # Remove _r or _l from the muscle name (handle both _r_ and _r at end)
                    clean_name = muscle.replace("_r_", "_").replace("_l_", "_")
                    clean_name = clean_name.replace("_r", "").replace("_l", "")
                    muscle_data[clean_name] = norm_fiber_length_norm_df[muscle].values

                trial_df = pd.DataFrame(
                    {
                        "subject_id": [subject] * 101,
                        "trial": [trial] * 101,
                        "leg": [leg.lower()] * 101,
                        "index": norm_fiber_length_norm_df.index,
                        "time": norm_fiber_length_norm_df["time"].values,
                        **muscle_data,
                    },
                )
                all_results.append(trial_df)

        output_df = pd.concat(all_results, ignore_index=True)
        return output_df

    def get_norm_tendon_length_all_muscles_df(self):
        all_results = []
        subject_utils = Subject_utils()
        all_subjects = subject_utils.get_all_subjects()
        for subject in all_subjects:
            trials = []
            leg = subject_utils.get_measured_leg(subject)

            trials = subject_utils.get_all_relevant_trials(subject)

            file_utils = File_utils()

            data_utils = Data_utils()

            for trial in trials:
                norm_tendon_length_df = data_utils.create_df(
                    file_utils.get_ceinms_normalized_tendon_lengths_file_path(
                        subject,
                        trial,
                    ),
                )
                norm_tendon_length_norm_df = self.time_normalise_df(norm_tendon_length_df)

                muscles = [muscles for muscles in norm_tendon_length_norm_df.columns if muscles != "time"]

                # left or right leg muscles
                if leg == "R":
                    muscles = [muscle for muscle in muscles if "_r" in muscle.lower()]
                else:
                    muscles = [muscle for muscle in muscles if "_l" in muscle.lower()]

                # Create dictionary with renamed columns (remove _r or _l)
                muscle_data = {}
                for muscle in muscles:
                    # Remove _r or _l from the muscle name (handle both _r_ and _r at end)
                    clean_name = muscle.replace("_r_", "_").replace("_l_", "_")
                    clean_name = clean_name.replace("_r", "").replace("_l", "")
                    muscle_data[clean_name] = norm_tendon_length_norm_df[muscle].values

                trial_df = pd.DataFrame(
                    {
                        "subject_id": [subject] * 101,
                        "trial": [trial] * 101,
                        "leg": [leg.lower()] * 101,
                        "index": norm_tendon_length_norm_df.index,
                        "time": norm_tendon_length_norm_df["time"].values,
                        **muscle_data,
                    },
                )
                all_results.append(trial_df)

        output_df = pd.concat(all_results, ignore_index=True)
        return output_df

    def get_speed_df(self):
        pre_df = self.prep_pre_df()
        post_df = self.prep_post_df()

        # Vectorized aggregations
        pre_agg = pre_df.groupby("subject_id").agg(
            {"time": ["first", "last"], "pelvis_tx": ["first", "last"]},
        )
        pre_agg.columns = ["time_start", "time_end", "pelvis_tx_start", "pelvis_tx_end"]

        post_agg = post_df.groupby("subject_id").agg(
            {"time": ["first", "last"], "pelvis_tx": ["first", "last"]},
        )
        post_agg.columns = [
            "time_start",
            "time_end",
            "pelvis_tx_start",
            "pelvis_tx_end",
        ]

        # Vectorized calculations
        speed_df = pd.DataFrame(
            {
                "subject_id": pre_agg.index,
                "pre_duration": pre_agg["time_end"] - pre_agg["time_start"],
                "post_duration": post_agg["time_end"] - post_agg["time_start"],
            },
        )

        speed_df["duration_change_abs"] = speed_df["post_duration"] - speed_df["pre_duration"]
        speed_df["duration_change_%"] = (speed_df["duration_change_abs"] / speed_df["pre_duration"]) * 100

        speed_df["pre_distance_m"] = pre_agg["pelvis_tx_end"] - pre_agg["pelvis_tx_start"]
        speed_df["post_distance_m"] = post_agg["pelvis_tx_end"] - post_agg["pelvis_tx_start"]
        speed_df["distance_change_m"] = speed_df["post_distance_m"] - speed_df["pre_distance_m"]

        speed_df["pre_speed_m_s"] = speed_df["pre_distance_m"] / speed_df["pre_duration"]
        speed_df["post_speed_m_s"] = speed_df["post_distance_m"] / speed_df["post_duration"]
        speed_df["speed_change_m_s"] = speed_df["post_speed_m_s"] - speed_df["pre_speed_m_s"]
        speed_df["speed_change_%"] = (speed_df["speed_change_m_s"] / speed_df["pre_speed_m_s"]) * 100

        speed_df = speed_df.reset_index(drop=True)
        return speed_df

    def get_norm_fiber_length_responders_ids(self, muscles: list | None = None) -> dict:
        """
        Identify subjects who are responders based on peak normalized fiber length changes.
        A responder is defined as having a percentage increase in peak normalized fiber length.

        Args:
            muscles: List of muscle base names. If None, defaults to hamstring muscles.

        Returns:
            Dictionary with muscle names as keys and lists of responder subject IDs as values.

        """
        # Default to hamstring muscles if not specified
        if muscles is None:
            muscles = ["bflh", "semiten", "semimem"]

        peak_norm_fiber_length_df = self.get_peak_norm_fiber_length_df(muscles=muscles)

        responders = {}

        for muscle in muscles:
            pre_col = f"{muscle}_peak_norm_fiber_length_pre"
            post_col = f"{muscle}_peak_norm_fiber_length_post"

            responder_ids = peak_norm_fiber_length_df[
                (peak_norm_fiber_length_df[post_col] - peak_norm_fiber_length_df[pre_col])
                / peak_norm_fiber_length_df[pre_col]
                * 100
                > 0
            ]["subject_id"].tolist()

            responders[muscle] = responder_ids

        return responders

    def get_strain_responders_ids(self, muscles: list | None = None) -> dict:
        """Backward-compatible wrapper for normalized fiber length responder IDs."""
        return self.get_norm_fiber_length_responders_ids(muscles=muscles)

    def get_norm_fiber_length_non_responders_ids(self, muscles: list | None = None) -> dict:
        """
        Identify non-responders based on peak normalized fiber length changes.
        A non-responder is defined as not meeting responder criteria.

        Args:
            muscles: List of muscle base names. If None, defaults to hamstring muscles.

        Returns:
            Dictionary with muscle names as keys and lists of non-responder subject IDs as values.

        """
        responders = self.get_norm_fiber_length_responders_ids(muscles=muscles)
        all_subjects = Subject_utils().get_all_subjects(use_leading_zeros=False)
        non_responders = {}

        for muscle in responders.keys():
            non_responder_ids = [int(subj) for subj in all_subjects if int(subj) not in responders[muscle]]
            non_responders[muscle] = non_responder_ids

        return non_responders

    def get_strain_non_responders_ids(self, muscles: list | None = None) -> dict:
        """Backward-compatible wrapper for normalized fiber length non-responder IDs."""
        return self.get_norm_fiber_length_non_responders_ids(muscles=muscles)

    def get_cam_positive_ids(self, filter: bool = True) -> list:
        file_utils = File_utils()

        all_subjects = Subject_utils().get_all_subjects(use_leading_zeros=False)

        demographics_df = pd.read_csv(file_utils.get_subject_demographics_file_path())

        result = demographics_df.loc[
            demographics_df["Group"].isin(["FAIM", "FAIS"]),
            "Subject",
        ]

        # Filter to include only subjects in all_subjects
        if filter:
            result = result[result.astype(int).isin([int(subj) for subj in all_subjects])]

        # Always convert to int before returning
        result = result.astype(int)

        return result.tolist()

    def get_non_cam_ids(self) -> list:
        all_subjects = Subject_utils().get_all_subjects(use_leading_zeros=False)
        cam_positive_ids = self.get_cam_positive_ids()

        non_cam_ids = [int(subj) for subj in all_subjects if int(subj) not in cam_positive_ids]

        return non_cam_ids

    def get_cam_norm_fiber_length_responders_ids(self, muscles: list | None = None) -> dict:
        """
        Identify CAM positive subjects who are responders based on peak normalized fiber length changes.

        Args:
            muscles: List of muscle base names. If None, defaults to hamstring muscles.

        Returns:
            Dictionary with muscle names as keys and lists of CAM positive responder subject IDs as values.

        """
        cam_positive_ids = self.get_cam_positive_ids()

        responders = self.get_norm_fiber_length_responders_ids(muscles=muscles)

        cam_norm_fiber_length_responders = {}

        for muscle, responder_ids in responders.items():
            cam_responder_ids = [subj for subj in responder_ids if subj in cam_positive_ids]
            cam_norm_fiber_length_responders[muscle] = cam_responder_ids

        return cam_norm_fiber_length_responders

    def get_cam_strain_responders_ids(self, muscles: list | None = None) -> dict:
        """Backward-compatible wrapper for CAM normalized fiber length responder IDs."""
        return self.get_cam_norm_fiber_length_responders_ids(muscles=muscles)

    def get_peak_joint_angles(self) -> pd.DataFrame:
        """
        Calculate peak joint angles (knee, hip, ankle) for all subjects.

        Returns:
            DataFrame with peak angles for pre (baseline average) and post (RunL1) trials,
            plus the change between them.

        """
        file_utils = File_utils()
        data_utils = Data_utils()
        subject_utils = Subject_utils()

        all_results = []
        all_subjects = subject_utils.get_all_subjects()
        pre_trials = ["Run_baselineA1", "Run_baselineB1"]

        for subject in all_subjects:
            leg = subject_utils.get_measured_leg(subject)
            knee_angle_col = f"knee_angle_{leg.lower()}"
            hip_angle_col = f"hip_flexion_{leg.lower()}"
            ankle_angle_col = f"ankle_angle_{leg.lower()}"

            trials = subject_utils.get_all_relevant_trials(subject)

            # Collect peaks for each trial
            trial_peaks = {}
            for trial in trials:
                try:
                    ik_df = data_utils.create_df(
                        file_utils.get_ik_file_path(subject, trial),
                    )
                    ik_norm_df = self.time_normalise_df(ik_df)

                    trial_peaks[trial] = {
                        "knee": ik_norm_df[knee_angle_col].max(),
                        "hip": ik_norm_df[hip_angle_col].max(),
                        "ankle": ik_norm_df[ankle_angle_col].max(),
                    }
                except Exception:
                    continue

            # Calculate pre (average of baselines) and post values
            if not any(t in trial_peaks for t in pre_trials) or "RunL1" not in trial_peaks:
                continue

            pre_knee = np.mean(
                [trial_peaks[t]["knee"] for t in pre_trials if t in trial_peaks],
            )
            pre_hip = np.mean(
                [trial_peaks[t]["hip"] for t in pre_trials if t in trial_peaks],
            )
            pre_ankle = np.mean(
                [trial_peaks[t]["ankle"] for t in pre_trials if t in trial_peaks],
            )

            post_knee = trial_peaks["RunL1"]["knee"]
            post_hip = trial_peaks["RunL1"]["hip"]
            post_ankle = trial_peaks["RunL1"]["ankle"]

            all_results.append(
                {
                    "subject_id": subject,
                    "leg": leg.lower(),
                    "peak_knee_angle_pre": pre_knee,
                    "peak_knee_angle_post": post_knee,
                    "peak_knee_angle_change": post_knee - pre_knee,
                    "peak_hip_angle_pre": pre_hip,
                    "peak_hip_angle_post": post_hip,
                    "peak_hip_angle_change": post_hip - pre_hip,
                    "peak_ankle_angle_pre": pre_ankle,
                    "peak_ankle_angle_post": post_ankle,
                    "peak_ankle_angle_change": post_ankle - pre_ankle,
                },
            )

        return pd.DataFrame(all_results)

    def get_peak_joint_angle_velocities(self) -> pd.DataFrame:
        """
        Calculate peak joint angular velocities (knee, hip) for all subjects.

        Returns:
            DataFrame with peak angular velocities for pre (baseline average) and
            post (RunL1) trials, plus the change between them.

        """
        file_utils = File_utils()
        data_utils = Data_utils()
        subject_utils = Subject_utils()

        all_results = []
        all_subjects = subject_utils.get_all_subjects()
        pre_trials = ["Run_baselineA1", "Run_baselineB1"]

        for subject in all_subjects:
            leg = subject_utils.get_measured_leg(subject)
            knee_angle_col = f"knee_angle_{leg.lower()}"
            hip_angle_col = f"hip_flexion_{leg.lower()}"

            trials = subject_utils.get_all_relevant_trials(subject)

            # Collect peak velocities for each trial
            trial_knee_peaks = {}
            trial_hip_peaks = {}
            for trial in trials:
                try:
                    ik_df = data_utils.create_df(
                        file_utils.get_ik_file_path(subject, trial),
                    )
                    ik_norm_df = self.time_normalise_df(ik_df)

                    # Calculate angular velocity
                    time_diff = ik_norm_df["time"].diff()
                    knee_velocity = (ik_norm_df[knee_angle_col].diff() / time_diff).abs()
                    hip_velocity = (ik_norm_df[hip_angle_col].diff() / time_diff).abs()

                    trial_knee_peaks[trial] = knee_velocity.max()
                    trial_hip_peaks[trial] = hip_velocity.max()
                except Exception:
                    continue

            # Calculate pre (average of baselines) and post values
            if not any(t in trial_knee_peaks for t in pre_trials) or "RunL1" not in trial_knee_peaks:
                continue

            pre_knee_velocity = np.mean(
                [trial_knee_peaks[t] for t in pre_trials if t in trial_knee_peaks],
            )
            pre_hip_velocity = np.mean(
                [trial_hip_peaks[t] for t in pre_trials if t in trial_hip_peaks],
            )

            post_knee_velocity = trial_knee_peaks["RunL1"]
            post_hip_velocity = trial_hip_peaks["RunL1"]

            all_results.append(
                {
                    "subject_id": subject,
                    "leg": leg.lower(),
                    "peak_knee_velocity_pre": pre_knee_velocity,
                    "peak_knee_velocity_post": post_knee_velocity,
                    "peak_knee_velocity_change": post_knee_velocity - pre_knee_velocity,
                    "peak_hip_velocity_pre": pre_hip_velocity,
                    "peak_hip_velocity_post": post_hip_velocity,
                    "peak_hip_velocity_change": post_hip_velocity - pre_hip_velocity,
                },
            )

        return pd.DataFrame(all_results)

    def get_joint_range_of_motion(self) -> pd.DataFrame:
        """
        Calculate joint range of motion (ROM) for knee and hip for all subjects.

        Returns:
            DataFrame with ROM for pre (baseline average) and post (RunL1) trials,
            plus the change between them.

        """
        file_utils = File_utils()
        data_utils = Data_utils()
        subject_utils = Subject_utils()

        all_results = []
        all_subjects = subject_utils.get_all_subjects()
        pre_trials = ["Run_baselineA1", "Run_baselineB1"]

        for subject in all_subjects:
            leg = subject_utils.get_measured_leg(subject)
            knee_angle_col = f"knee_angle_{leg.lower()}"
            hip_angle_col = f"hip_flexion_{leg.lower()}"

            trials = subject_utils.get_all_relevant_trials(subject)

            # Collect ROM for each trial
            trial_rom = {}
            for trial in trials:
                try:
                    ik_df = data_utils.create_df(
                        file_utils.get_ik_file_path(subject, trial),
                    )
                    ik_norm_df = self.time_normalise_df(ik_df)

                    knee_rom = ik_norm_df[knee_angle_col].max() - ik_norm_df[knee_angle_col].min()
                    hip_rom = ik_norm_df[hip_angle_col].max() - ik_norm_df[hip_angle_col].min()

                    trial_rom[trial] = {"knee": knee_rom, "hip": hip_rom}
                except Exception:
                    continue

            # Calculate pre (average of baselines) and post values
            if not any(t in trial_rom for t in pre_trials) or "RunL1" not in trial_rom:
                continue

            pre_knee_rom = np.mean(
                [trial_rom[t]["knee"] for t in pre_trials if t in trial_rom],
            )
            pre_hip_rom = np.mean(
                [trial_rom[t]["hip"] for t in pre_trials if t in trial_rom],
            )

            post_knee_rom = trial_rom["RunL1"]["knee"]
            post_hip_rom = trial_rom["RunL1"]["hip"]

            all_results.append(
                {
                    "subject_id": subject,
                    "leg": leg.lower(),
                    "knee_rom_pre": pre_knee_rom,
                    "knee_rom_post": post_knee_rom,
                    "knee_rom_change": post_knee_rom - pre_knee_rom,
                    "hip_rom_pre": pre_hip_rom,
                    "hip_rom_post": post_hip_rom,
                    "hip_rom_change": post_hip_rom - pre_hip_rom,
                },
            )

        return pd.DataFrame(all_results)

    def _get_force_identifier(self, subject_id, trial) -> str:
        """
        Helper function to determine the correct GRF column identifier based on subject and trial.
        This accounts for any inconsistencies in column naming across different subjects or trials.

        Args:
            subject: Subject ID
            trial: Trial name

        Returns:
            The correct column name for the ground reaction force data.

        """
        file_utils = File_utils()

        external_loads_xml_path = file_utils.get_external_loads_xml_file_path(
            subject_id,
            trial,
        )

        tree = ET.parse(external_loads_xml_path)
        root = tree.getroot()
        force_identifier_elem = root.find(".//ExternalForce/force_identifier")

        if force_identifier_elem is None:
            raise ValueError(
                f"force_identifier element not found in XML for subject {subject_id}, trial {trial}",
            )

        force_identifier = str(force_identifier_elem.text)

        return force_identifier

    def get_contact_time_df(self, time_normalize: bool = False) -> pd.DataFrame:
        """
        Calculate ground contact time for all subjects.

        Args:
            time_normalize: If True, returns contact time as number of frames (0-100) in normalized gait cycle.
            If False, returns absolute contact time in seconds.

        Returns:
            DataFrame with contact times for pre (baseline average) and post (RunL1) trials,
            plus the change between them.

        """
        file_utils = File_utils()
        data_utils = Data_utils()
        subject_utils = Subject_utils()

        all_results = []
        all_subjects = subject_utils.get_all_subjects()
        pre_trials = ["Run_baselineA1", "Run_baselineB1"]

        for subject in all_subjects:
            trials = subject_utils.get_all_relevant_trials(subject)

            trial_contact_times = {}

            for trial in trials:
                try:
                    # Get force identifier to determine which GRF columns to use
                    force_identifier = self._get_force_identifier(subject, trial)

                    grf_df = data_utils.create_df(
                        file_utils.get_external_loads_mot_file_path(subject, trial),
                    )

                    # Filter columns that belong to this force identifier
                    force_cols = [col for col in grf_df.columns if col.strip().startswith(force_identifier)]

                    if time_normalize:
                        # Normalize to 101 frames (0-100)
                        grf_df = self.time_normalise_df(grf_df)
                        # Reset index to 0-100 for frame-based indexing
                        grf_df = grf_df.reset_index(drop=True)
                        # Filter to rows with non-zero GRF data using force identifier columns
                        filtered_df = grf_df[grf_df[force_cols].any(axis=1)]
                        if len(filtered_df) > 0:
                            # Contact time is the span of frames where GRF is active
                            # Use index (frame number) instead of time column
                            contact_time = filtered_df.index[-1] - filtered_df.index[0]
                            trial_contact_times[trial] = contact_time
                    else:
                        # Filter to rows with non-zero GRF data using force identifier columns
                        filtered_df = grf_df[grf_df[force_cols].any(axis=1)]
                        if len(filtered_df) > 0:
                            contact_time = filtered_df["time"].iloc[-1] - filtered_df["time"].iloc[0]
                            trial_contact_times[trial] = contact_time
                except Exception as e:
                    warnings.warn(f"Skipping trial {trial} for subject {subject}: {e}")
                    continue

            # Calculate pre (average of baselines) and post values
            if not any(t in trial_contact_times for t in pre_trials) or "RunL1" not in trial_contact_times:
                continue

            pre_contact_time = np.mean(
                [trial_contact_times[t] for t in pre_trials if t in trial_contact_times],
            )
            post_contact_time = trial_contact_times["RunL1"]

            all_results.append(
                {
                    "subject_id": subject,
                    "contact_time_pre": pre_contact_time,
                    "contact_time_post": post_contact_time,
                    "contact_time_change": post_contact_time - pre_contact_time,
                },
            )

        return pd.DataFrame(all_results)

    def get_initial_foot_contact_time_df(
        self,
        time_normalized: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate initial foot contact time for all subjects.

        Args:
            time_normalized: If True, returns frame index (0-100) where initial contact occurs in normalized gait cycle.
                           If False, returns absolute time in seconds from trial start.

        Returns:
            DataFrame with initial foot contact times for pre (baseline average) and post (RunL1) trials,
            plus the change between them.

        """
        file_utils = File_utils()
        data_utils = Data_utils()
        subject_utils = Subject_utils()

        all_results = []
        all_subjects = subject_utils.get_all_subjects()
        pre_trials = ["Run_baselineA1", "Run_baselineB1"]

        for subject in all_subjects:
            trials = subject_utils.get_all_relevant_trials(subject)

            trial_initial_contact_times = {}
            for trial in trials:
                try:
                    # Get force identifier to determine which GRF columns to use
                    force_identifier = self._get_force_identifier(subject, trial)

                    ik_mot = data_utils.create_df(
                        file_utils.get_ik_file_path(subject, trial),
                    )

                    trial_frame = []

                    trial_frame.append(ik_mot["time"].iloc[0])
                    trial_frame.append(ik_mot["time"].iloc[-1])

                    grf_df = data_utils.create_df(
                        file_utils.get_external_loads_mot_file_path(subject, trial),
                    )

                    # Cut off grf df at the start and end of the trial based on ik mot time
                    grf_df = grf_df[(grf_df["time"] >= trial_frame[0]) & (grf_df["time"] <= trial_frame[1])]

                    # Filter columns that belong to this force identifier
                    force_cols = [col for col in grf_df.columns if col.strip().startswith(force_identifier)]

                    if time_normalized:
                        # Normalize to 101 frames (0-100)
                        grf_df = self.time_normalise_df(grf_df)
                        # Reset index to 0-100 for frame-based indexing
                        grf_df = grf_df.reset_index(drop=True)
                        # Filter to rows with non-zero GRF data using force identifier columns
                        filtered_df = grf_df[grf_df[force_cols].any(axis=1)]
                        if len(filtered_df) > 0:
                            # Return the frame index where initial contact occurs
                            # Use index (frame number) instead of time column
                            initial_contact_time = filtered_df.index[0]
                            trial_initial_contact_times[trial] = initial_contact_time
                    else:
                        # Filter to rows with non-zero GRF data using force identifier columns
                        filtered_df = grf_df[grf_df[force_cols].any(axis=1)]
                        if len(filtered_df) > 0:
                            initial_contact_time = filtered_df["time"].iloc[0]
                            trial_initial_contact_times[trial] = initial_contact_time
                except Exception:
                    continue

            # Calculate pre (average of baselines) and post values
            if (
                not any(t in trial_initial_contact_times for t in pre_trials)
                or "RunL1" not in trial_initial_contact_times
            ):
                continue

            pre_initial_contact_time = np.mean(
                [trial_initial_contact_times[t] for t in pre_trials if t in trial_initial_contact_times],
            )
            post_initial_contact_time = trial_initial_contact_times["RunL1"]

            all_results.append(
                {
                    "subject_id": subject,
                    "initial_contact_time_pre": pre_initial_contact_time,
                    "initial_contact_time_post": post_initial_contact_time,
                    "initial_contact_time_change": post_initial_contact_time - pre_initial_contact_time,
                },
            )

        return pd.DataFrame(all_results)

    def get_step_length_df(self) -> pd.DataFrame:
        """
        Calculate step length for all subjects.

        Returns:
            DataFrame with step lengths for pre (baseline average) and post (RunL1) trials,
            plus the change between them.

        """
        file_utils = File_utils()
        data_utils = Data_utils()
        subject_utils = Subject_utils()

        all_results = []
        all_subjects = subject_utils.get_all_subjects()
        pre_trials = ["Run_baselineA1", "Run_baselineB1"]

        for subject in all_subjects:
            trials = subject_utils.get_all_relevant_trials(subject)

            trial_step_lengths = {}
            for trial in trials:
                try:
                    ik_df = data_utils.create_df(
                        file_utils.get_ik_file_path(subject, trial),
                    )

                    pelvis_tx = ik_df["pelvis_tx"].values
                    non_zero_values = pelvis_tx[pelvis_tx != 0]

                    if len(non_zero_values) > 0:
                        step_length = non_zero_values[-1] - non_zero_values[0]
                        trial_step_lengths[trial] = step_length
                except Exception:
                    continue

            # Calculate pre (average of baselines) and post values
            if not any(t in trial_step_lengths for t in pre_trials) or "RunL1" not in trial_step_lengths:
                continue

            pre_step_length = np.mean(
                [trial_step_lengths[t] for t in pre_trials if t in trial_step_lengths],
            )
            post_step_length = trial_step_lengths["RunL1"]

            all_results.append(
                {
                    "subject_id": subject,
                    "step_length_pre": pre_step_length,
                    "step_length_post": post_step_length,
                    "step_length_change": post_step_length - pre_step_length,
                },
            )

        return pd.DataFrame(all_results)

    def get_regression_df(self) -> pd.DataFrame:
        """
        Compile regression data from multiple sources into a single DataFrame.

        Returns:
            DataFrame with pre/post/change values for all biomechanical variables.

        """
        # Load all needed dataframes once (not in loop)
        avg_speed_df = self.get_speed_df()
        peak_angles_df = self.get_peak_joint_angles()
        peak_angle_velocities_df = self.get_peak_joint_angle_velocities()
        rom_df = self.get_joint_range_of_motion()
        contact_time_df = self.get_contact_time_df()
        step_length_df = self.get_step_length_df()
        peak_norm_fiber_length_df = self.get_peak_norm_fiber_length_df()

        # Ensure consistent subject_id types
        for df in [
            avg_speed_df,
            peak_angles_df,
            peak_angle_velocities_df,
            rom_df,
            contact_time_df,
            step_length_df,
        ]:
            df["subject_id"] = df["subject_id"].astype(int)

        # Get subjects and body mass
        subject_utils = Subject_utils()
        all_subjects = subject_utils.get_all_subjects()
        subject_ids = [int(s) for s in all_subjects]

        body_mass_data = pd.DataFrame(
            {
                "subject_id": subject_ids,
                "body_mass": [subject_utils.get_body_mass(s) for s in all_subjects],
            },
        )

        # Rename columns to match final naming convention
        speed_cols = {
            "pre_speed_m_s": "running_speed_pre",
            "post_speed_m_s": "running_speed_post",
            "speed_change_m_s": "running_speed_change",
        }
        avg_speed_df = avg_speed_df.rename(columns=speed_cols)

        angle_cols = {
            "peak_knee_angle_pre": "knee_peak_angle_pre",
            "peak_knee_angle_post": "knee_peak_angle_post",
            "peak_knee_angle_change": "knee_peak_angle_change",
            "peak_hip_angle_pre": "hip_peak_angle_pre",
            "peak_hip_angle_post": "hip_peak_angle_post",
            "peak_hip_angle_change": "hip_peak_angle_change",
        }
        peak_angles_df = peak_angles_df.rename(columns=angle_cols)

        velocity_cols = {
            "peak_knee_velocity_pre": "knee_peak_velocity_pre",
            "peak_knee_velocity_post": "knee_peak_velocity_post",
            "peak_knee_velocity_change": "knee_peak_velocity_change",
            "peak_hip_velocity_pre": "hip_peak_velocity_pre",
            "peak_hip_velocity_post": "hip_peak_velocity_post",
            "peak_hip_velocity_change": "hip_peak_velocity_change",
        }
        peak_angle_velocities_df = peak_angle_velocities_df.rename(
            columns=velocity_cols,
        )

        rom_cols = {
            "knee_rom_pre": "knee_range_of_motion_pre",
            "knee_rom_post": "knee_range_of_motion_post",
            "knee_rom_change": "knee_range_of_motion_change",
            "hip_rom_pre": "hip_range_of_motion_pre",
            "hip_rom_post": "hip_range_of_motion_post",
            "hip_rom_change": "hip_range_of_motion_change",
        }
        rom_df = rom_df.rename(columns=rom_cols)

        # Merge all dataframes on subject_id
        result_df = body_mass_data

        # Select only needed columns for each merge
        dfs_to_merge = [
            (
                avg_speed_df,
                [
                    "subject_id",
                    "running_speed_pre",
                    "running_speed_post",
                    "running_speed_change",
                ],
            ),
            (
                peak_angles_df,
                [
                    "subject_id",
                    "knee_peak_angle_pre",
                    "knee_peak_angle_post",
                    "knee_peak_angle_change",
                    "hip_peak_angle_pre",
                    "hip_peak_angle_post",
                    "hip_peak_angle_change",
                ],
            ),
            (
                peak_angle_velocities_df,
                [
                    "subject_id",
                    "knee_peak_velocity_pre",
                    "knee_peak_velocity_post",
                    "knee_peak_velocity_change",
                    "hip_peak_velocity_pre",
                    "hip_peak_velocity_post",
                    "hip_peak_velocity_change",
                ],
            ),
            (
                rom_df,
                [
                    "subject_id",
                    "knee_range_of_motion_pre",
                    "knee_range_of_motion_post",
                    "knee_range_of_motion_change",
                    "hip_range_of_motion_pre",
                    "hip_range_of_motion_post",
                    "hip_range_of_motion_change",
                ],
            ),
            (
                contact_time_df,
                [
                    "subject_id",
                    "contact_time_pre",
                    "contact_time_post",
                    "contact_time_change",
                ],
            ),
            (
                step_length_df,
                [
                    "subject_id",
                    "step_length_pre",
                    "step_length_post",
                    "step_length_change",
                ],
            ),
            (
                peak_norm_fiber_length_df,
                [
                    "subject_id",
                    "bflh_peak_norm_fiber_length_pre",
                    "bflh_peak_norm_fiber_length_post",
                    "bflh_peak_norm_fiber_length_change",
                    "semiten_peak_norm_fiber_length_pre",
                    "semiten_peak_norm_fiber_length_post",
                    "semiten_peak_norm_fiber_length_change",
                    "semimem_peak_norm_fiber_length_pre",
                    "semimem_peak_norm_fiber_length_post",
                    "semimem_peak_norm_fiber_length_change",
                ],
            ),
        ]

        for df, cols in dfs_to_merge:
            result_df = result_df.merge(df[cols], on="subject_id", how="inner")

        return result_df

    def multiple_linear_regression(
        self,
        type: str = "strain",
        muscle: str = "bflh",
        check_assumptions: bool = True,
        create_plots: bool = True,
        print_results: bool = True,
        output_file: Path | None = None,
        interactive: bool = False,
    ) -> dict:
        """
        Perform multiple linear regression analysis for muscle strain.

        Args:
            type: Type of regression - "strain" (pre values) or "delta_strain" (changes).
            muscle: Muscle to analyze - "bflh", "semiten", or "semimem".
            check_assumptions: Whether to check regression assumptions (default: True).
            create_plots: Whether to create diagnostic plots (default: True).
            print_results: Whether to print regression results summary (default: True).
            output_file: Optional Path to append detailed results in markdown format.
            interactive: If True and plotly is available, create interactive plots (default: False).

        Returns:
            Dictionary containing:
            - regression_results: dict with muscle, type, intercept, r_squared, and coefficients
            - assumptions: dict with assumption check results (if check_assumptions=True)

        Note:
            When check_assumptions=True, validates linearity, normality,
            homoscedasticity, and multicollinearity before interpreting results.

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

        if type == "strain":
            input_vars = ["body_mass"] + [f"{var}_pre" for var in base_vars if var != "body_mass"]
        elif type == "delta_strain":
            input_vars = ["body_mass"] + [f"{var}_change" for var in base_vars if var != "body_mass"]
        else:
            raise ValueError("type must be 'strain' or 'delta_strain'")

        output_vars = (
            [f"{muscle}_peak_norm_fiber_length_pre"]
            if type == "strain"
            else [f"{muscle}_peak_norm_fiber_length_change"]
        )

        regression_df = self.get_regression_df()

        # Keep subject_ids for plotting
        subject_ids = np.asarray(regression_df["subject_id"].values) if "subject_id" in regression_df.columns else None

        # Convert to numpy arrays to avoid type issues with sklearn
        X = regression_df[input_vars].values

        # Ensure y is a 1D numpy array
        y_raw = regression_df[output_vars].squeeze()
        if isinstance(y_raw, pd.DataFrame):
            y = y_raw.values.ravel()
        elif isinstance(y_raw, pd.Series):
            y = y_raw.values
        else:
            y = np.asarray(y_raw).ravel()

        # Explicitly ensure y is np.ndarray for type checker
        y = np.asarray(y)

        # Fit model once
        reg = linear_model.LinearRegression()
        reg.fit(X, y)

        # Extract coefficients and intercept, handling both scalar and array formats
        # coef_ can be 1D or 2D depending on single vs multiple targets
        coefficients = reg.coef_.ravel() if hasattr(reg.coef_, "ravel") else reg.coef_

        # intercept_ can be scalar (float) or 0-d array for single target
        intercept = float(np.atleast_1d(reg.intercept_)[0])

        # Calculate standardized coefficients (beta coefficients) for variable importance
        # Standardize X and y to get comparable coefficients
        from sklearn.preprocessing import StandardScaler

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Fit on standardized data
        reg_scaled = linear_model.LinearRegression()
        reg_scaled.fit(X_scaled, y_scaled)
        standardized_coefs = reg_scaled.coef_.ravel() if hasattr(reg_scaled.coef_, "ravel") else reg_scaled.coef_

        # Create importance ranking by absolute standardized coefficient
        importance_dict = {
            var: {
                "coef": float(coef),
                "std_coef": float(std_coef),
                "abs_std_coef": float(abs(std_coef)),
            }
            for var, coef, std_coef in zip(input_vars, coefficients, standardized_coefs)
        }

        # Sort by absolute standardized coefficient (descending)
        sorted_importance = sorted(
            importance_dict.items(),
            key=lambda x: x[1]["abs_std_coef"],
            reverse=True,
        )

        # Build regression results
        results = {
            "regression_results": {
                "muscle": muscle,
                "type": type,
                "intercept": intercept,
                "r_squared": float(reg.score(X, y)),
                "coefficients": {var: float(coef) for var, coef in zip(input_vars, coefficients)},
                "standardized_coefficients": {
                    var: float(std_coef) for var, std_coef in zip(input_vars, standardized_coefs)
                },
                "variable_importance": importance_dict,
                "ranked_variables": [
                    {
                        "variable": var,
                        "standardized_coef": info["std_coef"],
                        "abs_standardized_coef": info["abs_std_coef"],
                    }
                    for var, info in sorted_importance
                ],
            },
        }

        # Check assumptions if requested
        if check_assumptions:
            results["assumptions"] = self.regression_assumption_check(
                X,
                y,
                model=reg,
                create_plots=create_plots,
                interactive=interactive,
                subject_ids=subject_ids,
                column_names=input_vars,
            )

        # Print results if requested
        if print_results:
            # Map muscle codes to readable names
            muscle_names = {
                "bflh": "Biceps Femoris Long Head (BFLH)",
                "semiten": "Semitendinosus",
                "semimem": "Semimembranosus",
            }
            muscle_name = muscle_names.get(muscle, muscle)

            print(f"\n--- {muscle_name} ---")
            print(f"R²: {results['regression_results']['r_squared']:.4f}")

            if check_assumptions and "assumptions" in results:
                print(
                    f"All assumptions met: {results['assumptions']['summary']['all_assumptions_met']}",
                )

                # Print which assumptions failed
                assumptions_summary = results["assumptions"]["summary"]
                failed_assumptions = [
                    key
                    for key, value in assumptions_summary.items()
                    if key != "all_assumptions_met" and key != "linearity" and value == "FAIL"
                ]
                if failed_assumptions:
                    print(f"Failed assumptions: {', '.join(failed_assumptions)}")

            # Print top 3 influencers
            print("\nTop 3 Influencers (by absolute standardized coefficient):")
            for i, var_info in enumerate(
                results["regression_results"]["ranked_variables"][:3],
                1,
            ):
                print(
                    f"  {i}. {var_info['variable']}: {var_info['standardized_coef']:+.3f}",
                )

        # Save to markdown file if requested
        if output_file is not None:
            self._save_regression_to_markdown(results, output_file)

        return results

    def _save_regression_to_markdown(self, results: dict, output_file: Path) -> None:
        """
        Save regression results to a markdown file.

        Args:
            results: Dictionary containing regression results and assumptions
            output_file: Path to the markdown file to save (appends to existing file)

        """
        # Map muscle codes to readable names
        muscle_names = {
            "bflh": "Biceps Femoris Long Head (BFLH)",
            "semiten": "Semitendinosus",
            "semimem": "Semimembranosus",
        }

        reg_results = results["regression_results"]
        muscle_name = muscle_names.get(reg_results["muscle"], reg_results["muscle"])
        analysis_type = "Baseline Strain" if reg_results["type"] == "strain" else "Strain Changes (Delta)"

        # Create markdown content
        md_content = []
        md_content.append(f"## {muscle_name}\n\n")
        md_content.append(f"**Analysis Type:** {analysis_type}\n\n")

        # R² with explanation
        r_squared = reg_results["r_squared"]
        md_content.append(f"**R² (Coefficient of Determination):** {r_squared:.4f}\n\n")
        md_content.append("*Interpretation:* ")
        r_squared_pct = r_squared * 100
        if r_squared < 0.3:
            strength = "weak"
        elif r_squared < 0.5:
            strength = "moderate"
        elif r_squared < 0.7:
            strength = "substantial"
        else:
            strength = "strong"
        md_content.append(
            f"R² = {r_squared:.4f} indicates that {r_squared_pct:.1f}% of the variance in ",
        )
        md_content.append(
            f"{analysis_type.lower()} is explained by the predictor variables in the model. ",
        )
        md_content.append(f"This represents a **{strength}** model fit. ")
        if r_squared < 0.3:
            md_content.append(
                "The low R² suggests that important predictors may be missing or that the ",
            )
            md_content.append(
                "relationship is complex and not well-captured by linear regression. ",
            )
        elif r_squared < 0.5:
            md_content.append(
                "While the model captures some variance, substantial unexplained variance remains, ",
            )
            md_content.append("suggesting other factors also influence the outcome. ")
        elif r_squared < 0.7:
            md_content.append(
                "The model captures most of the systematic variance, though some unexplained variance remains. ",
            )
        else:
            md_content.append(
                "The model explains most of the variance in the outcome variable. ",
            )
        md_content.append(
            f"The remaining {100 - r_squared_pct:.1f}% is due to factors not included in the model ",
        )
        md_content.append("or inherent variability.\n\n")

        # Intercept with explanation
        intercept = reg_results["intercept"]
        md_content.append(f"**Intercept:** {intercept:.4f}\n\n")
        md_content.append("*Interpretation:* ")
        md_content.append(f"The intercept ({intercept:.4f}) represents the predicted ")
        md_content.append(
            f"{analysis_type.lower()} when all predictor variables are zero. ",
        )
        if reg_results["type"] == "strain":
            if intercept > 0:
                md_content.append(
                    "The positive intercept suggests a baseline strain value even under hypothetical ",
                )
                md_content.append("zero conditions for all predictors. ")
            else:
                md_content.append(
                    "The negative intercept is theoretically possible but may indicate the model ",
                )
                md_content.append(
                    "is extrapolating beyond the range of observed data. ",
                )
        elif abs(intercept) < 0.01:
            md_content.append(
                "The near-zero intercept suggests minimal baseline change when predictors are at zero, ",
            )
            md_content.append("which is expected for a change (delta) model. ")
        else:
            md_content.append(
                "The non-zero intercept indicates a baseline shift in strain change ",
            )
            md_content.append("independent of the measured predictors. ")
        md_content.append(
            "**Note:** Intercept interpretation should be done cautiously as predictor values ",
        )
        md_content.append("of zero may be outside the observed data range.\n\n")

        # Regression equation
        md_content.append("### Regression Equation\n\n")
        md_content.append("The full linear regression model:\n\n")

        # Build equation
        outcome_var = "Strain" if reg_results["type"] == "strain" else "ΔStrain"
        equation_parts = [f"{outcome_var} = {intercept:.6f}"]

        # Add each predictor term
        for var, coef in reg_results["coefficients"].items():
            sign = "+" if coef >= 0 else "-"
            equation_parts.append(f" {sign} {abs(coef):.6f} × {var}")

        # Join with line breaks for readability (split into multiple lines)
        md_content.append("```\n")
        md_content.append(equation_parts[0] + "\n")
        for term in equation_parts[1:]:
            md_content.append("    " + term + "\n")
        md_content.append("```\n\n")

        md_content.append("*Interpretation:* ")
        md_content.append(
            "This equation shows the predicted outcome as a linear combination of all predictor variables. ",
        )
        md_content.append(
            "Each coefficient represents the change in the outcome for a one-unit increase in that ",
        )
        md_content.append("predictor, holding all other variables constant. ")
        if reg_results["type"] == "strain":
            md_content.append(
                "For example, a one-unit increase in a predictor with coefficient +0.003 would ",
            )
            md_content.append("increase the predicted strain by 0.003 units.\n\n")
        else:
            md_content.append(
                "For example, a one-unit increase in a predictor with coefficient -0.002 would ",
            )
            md_content.append(
                "decrease the predicted strain change by 0.002 units.\n\n",
            )

        # Assumptions section
        if "assumptions" in results:
            md_content.append("### Assumption Checks\n\n")
            assumptions_summary = results["assumptions"]["summary"]
            all_met = assumptions_summary.get("all_assumptions_met", False)
            md_content.append(
                f"**All assumptions met:** {'✅ Yes' if all_met else '⚠️ No'}\n\n",
            )

            # List individual assumptions
            md_content.append("| Assumption | Status |\n")
            md_content.append("|------------|--------|\n")
            for key, value in assumptions_summary.items():
                if key != "all_assumptions_met":
                    status_icon = "✅ PASS" if value == "PASS" else "❌ FAIL"
                    md_content.append(
                        f"| {key.replace('_', ' ').title()} | {status_icon} |\n",
                    )
            md_content.append("\n")

            # Add detailed explanations for each assumption
            md_content.append("#### Detailed Assumption Results and Interpretation\n\n")

            # 1. Linearity
            md_content.append("**1. Linearity**\n\n")
            md_content.append(
                f"*Status:* {assumptions_summary.get('linearity', 'N/A')}\n\n",
            )
            md_content.append(
                "*Test Method:* Visual inspection of residuals vs fitted values plot\n\n",
            )
            md_content.append("*Interpretation:* ")
            md_content.append(
                "Linearity is assessed through visual inspection of the residuals vs fitted values plot. ",
            )
            md_content.append(
                "A random scatter of residuals around zero (no patterns) indicates linearity is satisfied. ",
            )
            md_content.append(
                "Systematic patterns (curves, funnels, or clusters) suggest non-linear relationships that ",
            )
            md_content.append(
                "the model fails to capture. If patterns are present, consider adding polynomial terms, ",
            )
            md_content.append(
                "interaction effects, or transforming variables. **Note:** Please examine the diagnostic plot ",
            )
            md_content.append("for this regression to assess linearity visually.\n\n")

            # 2. Normality
            if "normality" in results["assumptions"]:
                norm_data = results["assumptions"]["normality"]
                md_content.append("**2. Normality of Residuals**\n\n")
                md_content.append(
                    f"*Status:* {assumptions_summary.get('normality', 'N/A')}\n\n",
                )
                md_content.append("*Test Method:* Shapiro-Wilk test\n\n")
                md_content.append("*Test Results:*\n")
                md_content.append(
                    f"- Shapiro-Wilk statistic: {norm_data['shapiro_stat']:.4f}\n",
                )
                md_content.append(f"- p-value: {norm_data['shapiro_p']:.4f}\n")
                md_content.append("- Significance level: α = 0.05\n\n")
                md_content.append("*Interpretation:* ")
                if norm_data["passes_test"]:
                    md_content.append(
                        f"With p = {norm_data['shapiro_p']:.4f} > 0.05, we fail to reject the null hypothesis. ",
                    )
                    md_content.append(
                        "The residuals are approximately normally distributed, which supports valid ",
                    )
                    md_content.append(
                        "hypothesis testing and confidence intervals for regression coefficients.\n\n",
                    )
                else:
                    md_content.append(
                        f"With p = {norm_data['shapiro_p']:.4f} < 0.05, we reject the null hypothesis. ",
                    )
                    md_content.append(
                        "The residuals show significant departure from normality. This may affect ",
                    )
                    md_content.append(
                        "the validity of hypothesis tests and confidence intervals, though parameter ",
                    )
                    md_content.append(
                        "estimates remain unbiased. Consider robust regression methods or data transformations.\n\n",
                    )

            # 3. Homoscedasticity
            if "homoscedasticity" in results["assumptions"]:
                homo_data = results["assumptions"]["homoscedasticity"]
                md_content.append("**3. Homoscedasticity (Constant Variance)**\n\n")
                md_content.append(
                    f"*Status:* {assumptions_summary.get('homoscedasticity', 'N/A')}\n\n",
                )
                md_content.append("*Test Method:* Breusch-Pagan test\n\n")
                md_content.append("*Test Results:*\n")
                md_content.append(
                    f"- Breusch-Pagan statistic: {homo_data['bp_stat']:.4f}\n",
                )
                md_content.append(f"- p-value: {homo_data['bp_p']:.4f}\n")
                md_content.append("- Significance level: α = 0.05\n\n")
                md_content.append("*Interpretation:* ")
                if homo_data["passes_test"]:
                    md_content.append(
                        f"With p = {homo_data['bp_p']:.4f} > 0.05, we fail to reject the null hypothesis. ",
                    )
                    md_content.append(
                        "The variance of residuals is approximately constant across fitted values ",
                    )
                    md_content.append(
                        "(homoscedastic). This supports the reliability of standard errors and hypothesis tests.\n\n",
                    )
                else:
                    md_content.append(
                        f"With p = {homo_data['bp_p']:.4f} < 0.05, we reject the null hypothesis. ",
                    )
                    md_content.append(
                        "The variance of residuals changes across fitted values (heteroscedastic). ",
                    )
                    md_content.append(
                        "This violates the constant variance assumption and may lead to inefficient estimates ",
                    )
                    md_content.append(
                        "and incorrect standard errors. Consider weighted least squares or robust standard errors.\n\n",
                    )

            # 4. Multicollinearity
            if "multicollinearity" in results["assumptions"]:
                multi_data = results["assumptions"]["multicollinearity"]
                md_content.append("**4. Multicollinearity**\n\n")
                md_content.append(
                    f"*Status:* {assumptions_summary.get('multicollinearity', 'N/A')}\n\n",
                )
                md_content.append("*Test Method:* Variance Inflation Factor (VIF)\n\n")
                md_content.append("*Test Results (VIF values):*\n")
                vif_values = multi_data.get("vif_values", {})
                for var, vif in vif_values.items():
                    if isinstance(vif, (int, float)):
                        warning = " ⚠️ HIGH" if vif >= 10 else ""
                        md_content.append(f"- {var}: {vif:.2f}{warning}\n")
                md_content.append(
                    "- Threshold: VIF < 10 (acceptable), VIF ≥ 10 (concerning)\n\n",
                )
                md_content.append("*Interpretation:* ")
                if multi_data["passes_test"]:
                    md_content.append(
                        "All VIF values are below 10, indicating acceptable levels of multicollinearity. ",
                    )
                    md_content.append(
                        "The predictor variables are sufficiently independent, allowing for stable and ",
                    )
                    md_content.append("interpretable regression coefficients.\n\n")
                else:
                    high_vif_vars = [
                        var for var, vif in vif_values.items() if isinstance(vif, (int, float)) and vif >= 10
                    ]
                    md_content.append(
                        f"High multicollinearity detected (VIF ≥ 10 for: {', '.join(high_vif_vars)}). ",
                    )
                    md_content.append(
                        "Strong correlations between predictors inflate standard errors and make ",
                    )
                    md_content.append(
                        "coefficient estimates unstable. Consider removing redundant variables, ",
                    )
                    md_content.append(
                        "combining correlated predictors, or using regularization methods (Ridge/Lasso).\n\n",
                    )

            # Overall interpretation
            md_content.append("#### Overall Model Validity\n\n")
            if all_met:
                md_content.append(
                    "✅ All testable regression assumptions are satisfied (normality, homoscedasticity, multicollinearity). ",
                )
                md_content.append(
                    "The model results are valid and coefficient estimates, standard errors, and hypothesis tests ",
                )
                md_content.append(
                    "are reliable. **However**, always review the linearity diagnostic plot to confirm no ",
                )
                md_content.append("systematic patterns exist in the residuals.\n\n")
            else:
                md_content.append(
                    "⚠️ One or more assumptions are violated. Interpret results with caution:\n",
                )
                if assumptions_summary.get("normality") == "FAIL":
                    md_content.append(
                        "- **Normality violation**: Hypothesis tests and confidence intervals may be unreliable\n",
                    )
                if assumptions_summary.get("homoscedasticity") == "FAIL":
                    md_content.append(
                        "- **Homoscedasticity violation**: Standard errors may be incorrect; hypothesis tests unreliable\n",
                    )
                if assumptions_summary.get("multicollinearity") == "FAIL":
                    md_content.append(
                        "- **Multicollinearity**: Coefficient estimates are unstable; difficult to interpret individual effects\n",
                    )
                md_content.append(
                    "\n**Important:** Also check the linearity diagnostic plot for systematic patterns in residuals.\n\n",
                )

        # Variable importance section
        md_content.append("### Variable Importance (Standardized Coefficients)\n\n")
        md_content.append(
            "Standardized coefficients allow direct comparison of variable importance.\n\n",
        )
        md_content.append("| Rank | Variable | Standardized Coef | Direction |\n")
        md_content.append("|------|----------|------------------|-----------|\n")

        for i, var_info in enumerate(reg_results["ranked_variables"], 1):
            std_coef = var_info["standardized_coef"]
            direction = "↑ Positive" if std_coef > 0 else "↓ Negative"
            md_content.append(
                f"| {i} | {var_info['variable']} | {std_coef:+.4f} | {direction} |\n",
            )

        # Raw coefficients section
        md_content.append("\n### Raw Regression Coefficients\n\n")
        md_content.append("| Variable | Coefficient |\n")
        md_content.append("|----------|-------------|\n")
        for var, coef in reg_results["coefficients"].items():
            md_content.append(f"| {var} | {coef:+.6f} |\n")

        md_content.append("\n---\n\n")

        # Write to file (append mode)
        with open(output_file, "a", encoding="utf-8") as f:
            f.writelines(md_content)

    def regression_assumption_check(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        model: linear_model.LinearRegression | None = None,
        create_plots: bool = True,
        interactive: bool = False,
        subject_ids: np.ndarray | None = None,
        column_names: list[str] | None = None,
    ) -> dict:
        """
        Check regression assumptions for linear regression model.

        Args:
            X: Predictor variables (DataFrame or numpy array)
            y: Response variable (Series or numpy array)
            model: Fitted LinearRegression model. If None, will fit a new model.
            create_plots: Whether to create diagnostic plots (default: True)
            interactive: If True and plotly is available, create interactive plots (default: False)
            subject_ids: Optional array of subject IDs for hover labels in interactive plots
            column_names: Optional list of column names for predictors (used when X is numpy array)

        Returns:
            Dictionary containing assumption check results:
            - linearity: dict with residuals and fitted values
            - normality: dict with shapiro_stat, shapiro_p, passes_test
            - homoscedasticity: dict with bp_stat, bp_p, passes_test
            - multicollinearity: dict with VIF values for each predictor
            - summary: dict with overall pass/fail for each assumption
            - figures: list of matplotlib figure objects or plotly figures (if create_plots=True)

        """
        import matplotlib.pyplot as plt
        from scipy.stats import shapiro

        # Check if plotly is available for interactive plots
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False
            interactive = False  # Fallback to matplotlib

        # Convert to numpy arrays for consistent processing
        X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_array = y.values if isinstance(y, pd.Series) else np.asarray(y)

        # Explicitly ensure they are numpy arrays for type checker
        X_array = np.asarray(X_array)
        y_array = np.asarray(y_array)

        # Keep column names for VIF labels
        if column_names is None:
            if isinstance(X, pd.DataFrame):
                column_names = X.columns.tolist()
            else:
                column_names = [f"X{i}" for i in range(X_array.shape[1])]

        # Fit model if not provided
        if model is None:
            model = linear_model.LinearRegression()
            model.fit(X_array, y_array)

        # Get predictions and residuals
        y_pred = model.predict(X_array)
        residuals = y_array - y_pred

        results = {}
        figures = []

        # 1. Linearity: Residuals vs Fitted Values
        results["linearity"] = {
            "fitted_values": y_pred,
            "residuals": residuals,
            "description": "Check residuals vs fitted plot - should show random scatter with no pattern",
        }

        if create_plots:
            if interactive and PLOTLY_AVAILABLE:
                # Get subject IDs for hover labels
                if subject_ids is not None:
                    hover_ids = subject_ids
                elif isinstance(X, pd.DataFrame) and "subject_id" in X.columns:
                    hover_ids = X["subject_id"].values
                elif isinstance(X, pd.DataFrame) and hasattr(X, "index"):
                    hover_ids = X.index.values
                else:
                    hover_ids = np.arange(len(y_pred))

                fig = go.Figure()  # type: ignore
                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=y_pred,
                        y=residuals,
                        mode="markers",
                        marker=dict(size=8, color="blue", opacity=0.5),
                        text=[f"Subject {sid}" for sid in hover_ids],
                        hovertemplate="<b>%{text}</b><br>Fitted: %{x:.4f}<br>Residual: %{y:.4f}<extra></extra>",
                    ),
                )
                fig.add_hline(y=0, line_color="red", line_dash="dash", line_width=2)
                fig.update_layout(
                    title="Residuals vs Fitted Values (Linearity Check)",
                    xaxis_title="Fitted Values",
                    yaxis_title="Residuals",
                    hovermode="closest",
                    height=500,
                )
                figures.append((fig, "linearity"))
            else:
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.axhline(y=0, color="r", linestyle="--")
                plt.xlabel("Fitted Values")
                plt.ylabel("Residuals")
                plt.title("Residuals vs Fitted Values (Linearity Check)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                figures.append((fig, "linearity"))

        # 2. Normality of Residuals: Shapiro-Wilk test and Q-Q plot
        shapiro_stat, shapiro_p = shapiro(residuals)
        passes_normality = shapiro_p > 0.05

        results["normality"] = {
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "passes_test": passes_normality,
            "description": f"Shapiro-Wilk p-value: {shapiro_p:.4f} ({'PASS' if passes_normality else 'FAIL'} at α=0.05)",
        }

        if create_plots:
            if interactive and PLOTLY_AVAILABLE:
                from scipy import stats as sp_stats

                # Create subplot for Q-Q plot and histogram
                fig = make_subplots(  # type: ignore
                    rows=1,
                    cols=2,
                    subplot_titles=(
                        "Q-Q Plot (Normality Check)",
                        f"Residual Distribution (Shapiro-Wilk p={shapiro_p:.4f})",
                    ),
                )

                # Q-Q plot
                (osm, osr), (slope, intercept, r) = sp_stats.probplot(
                    residuals,
                    dist="norm",
                )
                osm = np.asarray(osm)
                osr = np.asarray(osr)
                slope = float(slope)  # type: ignore
                intercept = float(intercept)  # type: ignore
                # Get subject IDs for hover labels
                if subject_ids is not None:
                    hover_ids = subject_ids
                elif isinstance(X, pd.DataFrame) and "subject_id" in X.columns:
                    hover_ids = X["subject_id"].values
                elif isinstance(X, pd.DataFrame) and hasattr(X, "index"):
                    hover_ids = X.index.values
                else:
                    hover_ids = np.arange(len(residuals))

                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=osm,
                        y=osr,
                        mode="markers",
                        marker=dict(size=6, color="blue"),
                        name="Data",
                        text=[f"Subject {sid}" for sid in hover_ids],
                        hovertemplate="<b>%{text}</b><br>Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                # Add reference line
                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=osm,
                        y=slope * osm + intercept,
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="Reference",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )

                # Histogram
                fig.add_trace(
                    go.Histogram(  # type: ignore
                        x=residuals,
                        nbinsx=30,
                        marker=dict(
                            color="blue",
                            opacity=0.7,
                            line=dict(color="black", width=1),
                        ),
                        name="Residuals",
                        showlegend=False,
                        hovertemplate="Bin: %{x}<br>Count: %{y}<extra></extra>",
                    ),
                    row=1,
                    col=2,
                )

                fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=1)
                fig.update_yaxes(title_text="Sample Quantiles", row=1, col=1)
                fig.update_xaxes(title_text="Residuals", row=1, col=2)
                fig.update_yaxes(title_text="Frequency", row=1, col=2)
                fig.update_layout(height=500, hovermode="closest")
                figures.append((fig, "normality"))
            else:
                from scipy import stats as sp_stats

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Q-Q plot
                sp_stats.probplot(residuals, dist="norm", plot=ax1)
                ax1.set_title("Q-Q Plot (Normality Check)")
                ax1.grid(True, alpha=0.3)

                # Histogram of residuals
                ax2.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
                ax2.set_xlabel("Residuals")
                ax2.set_ylabel("Frequency")
                ax2.set_title(f"Residual Distribution\nShapiro-Wilk p={shapiro_p:.4f}")
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                figures.append((fig, "normality"))

        # 3. Homoscedasticity: Breusch-Pagan test
        # Add constant to X for Breusch-Pagan test (required by statsmodels)
        import statsmodels.api as sm

        X_with_const = sm.add_constant(X_array)
        bp_test = het_breuschpagan(residuals, X_with_const)
        bp_stat, bp_p, _, _ = bp_test
        passes_homoscedasticity = bp_p > 0.05

        results["homoscedasticity"] = {
            "bp_stat": float(bp_stat),
            "bp_p": float(bp_p),
            "passes_test": passes_homoscedasticity,
            "description": f"Breusch-Pagan p-value: {bp_p:.4f} ({'PASS' if passes_homoscedasticity else 'FAIL'} at α=0.05)",
        }

        if create_plots:
            if interactive and PLOTLY_AVAILABLE:
                # Get subject IDs for hover labels
                if subject_ids is not None:
                    hover_ids = subject_ids
                elif isinstance(X, pd.DataFrame) and "subject_id" in X.columns:
                    hover_ids = X["subject_id"].values
                elif isinstance(X, pd.DataFrame) and hasattr(X, "index"):
                    hover_ids = X.index.values
                else:
                    hover_ids = np.arange(len(y_pred))

                fig = go.Figure()  # type: ignore
                fig.add_trace(
                    go.Scatter(  # type: ignore
                        x=y_pred,
                        y=np.abs(residuals),
                        mode="markers",
                        marker=dict(size=8, color="blue", opacity=0.5),
                        text=[f"Subject {sid}" for sid in hover_ids],
                        hovertemplate="<b>%{text}</b><br>Fitted: %{x:.4f}<br>Abs Residual: %{y:.4f}<extra></extra>",
                    ),
                )
                fig.update_layout(
                    title=f"Absolute Residuals vs Fitted Values (Homoscedasticity Check)<br>Breusch-Pagan p={bp_p:.4f}",
                    xaxis_title="Fitted Values",
                    yaxis_title="Absolute Residuals",
                    hovermode="closest",
                    height=500,
                )
                figures.append((fig, "homoscedasticity"))
            else:
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(y_pred, np.abs(residuals), alpha=0.5)
                plt.xlabel("Fitted Values")
                plt.ylabel("Absolute Residuals")
                plt.title(
                    f"Absolute Residuals vs Fitted Values (Homoscedasticity Check)\nBreusch-Pagan p={bp_p:.4f}",
                )
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                figures.append((fig, "homoscedasticity"))

        # 4. Multicollinearity: VIF
        vif_data = {}
        for i, col_name in enumerate(column_names):
            try:
                vif = variance_inflation_factor(X_array, i)
                vif_data[col_name] = float(vif)
            except Exception as e:
                vif_data[col_name] = f"Error: {e!s}"

        # Check if any VIF > 10 (common threshold)
        vif_values = [v for v in vif_data.values() if isinstance(v, (int, float))]
        passes_vif = all(v < 10 for v in vif_values) if vif_values else False

        results["multicollinearity"] = {
            "vif_values": vif_data,
            "passes_test": passes_vif,
            "description": f"VIF values {'all < 10 (PASS)' if passes_vif else 'some >= 10 (FAIL)'} - threshold is 10",
        }

        if create_plots:
            fig = plt.figure(figsize=(10, 6))
            vif_df = pd.DataFrame(
                {"Variable": list(vif_data.keys()), "VIF": list(vif_data.values())},
            )
            # Filter only numeric VIF values for plotting
            vif_df_numeric = vif_df[vif_df["VIF"].apply(lambda x: isinstance(x, (int, float)))]

            if not vif_df_numeric.empty:
                plt.barh(vif_df_numeric["Variable"], vif_df_numeric["VIF"])
                plt.axvline(x=10, color="r", linestyle="--", label="Threshold (10)")
                plt.xlabel("VIF")
                plt.ylabel("Predictor Variables")
                plt.title("Variance Inflation Factor (Multicollinearity Check)")
                plt.legend()
                plt.grid(True, alpha=0.3, axis="x")
                plt.tight_layout()
                figures.append((fig, "multicollinearity"))

        # Summary
        results["summary"] = {
            "linearity": "Visual inspection required - check residuals vs fitted plot",
            "normality": "PASS" if passes_normality else "FAIL",
            "homoscedasticity": "PASS" if passes_homoscedasticity else "FAIL",
            "multicollinearity": "PASS" if passes_vif else "FAIL",
            "all_assumptions_met": passes_normality and passes_homoscedasticity and passes_vif,
        }

        # Add figures to results if created
        if create_plots:
            results["figures"] = figures

        return results

    def pca(
        self,
        X: pd.DataFrame | np.ndarray,
        n_components: int | None = None,
        scale_data: bool = True,
    ) -> dict:
        """
        Perform Principal Component Analysis (PCA) on the given dataset.

        Args:
            X: Input data (DataFrame or numpy array)
            n_components: Number of principal components to compute. If None, computes all components.
            scale_data: Whether to standardize the data before PCA (default: True)

        Returns:
            Dictionary containing PCA results:
            - pca_model: Fitted PCA model object
            - explained_variance_ratio: Array of variance explained by each principal component
            - components: Array of principal component loadings
            - scores: Array of transformed data in PCA space

        """
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.preprocessing import StandardScaler

        # Convert to numpy array if DataFrame is provided
        X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        # Scale data if requested
        if scale_data:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)
        else:
            X_scaled = X_array

        # Fit PCA model
        pca_model = SklearnPCA(n_components=n_components)
        scores = pca_model.fit_transform(X_scaled)

        return {
            "pca_model": pca_model,
            "explained_variance_ratio": pca_model.explained_variance_ratio_,
            "components": pca_model.components_,
            "scores": scores,
        }
