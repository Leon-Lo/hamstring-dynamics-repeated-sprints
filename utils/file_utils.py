import os
import shutil
from pathlib import Path

import pandas as pd


class File_utils:
    def __init__(
        self,
        model_base_dir: Path | None = None,
        sim_base_dir: Path | None = None,
        generic_files_dir: Path | None = None,
    ):
        self.model_base_dir = model_base_dir if model_base_dir is not None else Path("data/models")

        self.sim_base_dir = sim_base_dir if sim_base_dir is not None else Path("data/simulations")
        self.generic_files_dir = (
            generic_files_dir if generic_files_dir is not None else Path("data/generic_setup_files")
        )

    def get_sim_run_dir(self, run_id: str) -> Path:
        sim_dir = self.sim_base_dir / run_id
        if not sim_dir.is_dir():
            raise FileNotFoundError(f"Simulation directory {sim_dir} does not exist.")
        return sim_dir

    def get_session_dir(self, run_id: str) -> Path:
        sim_dir = self.get_sim_run_dir(run_id)
        session_dir = sim_dir / "pre"
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session directory {session_dir} does not exist.")
        return session_dir

    def get_trial_dirs(self, run_id: str) -> list:
        sim_dir = self.get_sim_run_dir(run_id)
        trial_dirs = list(sim_dir.glob("pre/*"))
        if not trial_dirs:
            raise FileNotFoundError(f"No trial directories found in {sim_dir}.")
        return trial_dirs

    def get_trial_dir(self, run_id, trial_name: str) -> Path:
        trial_dirs = self.get_trial_dirs(run_id)
        for trial_dir in trial_dirs:
            if trial_name in str(trial_dir):
                return trial_dir
        raise FileNotFoundError(
            f"Trial directory {trial_name} not found in {self.get_sim_run_dir(run_id)}.",
        )

    def get_output_dir(self, run_id: str, trial_name: str) -> Path:
        sim_dir = self.get_trial_dir(run_id, trial_name)
        outputs_dir_path = sim_dir

        if not outputs_dir_path.is_dir():
            try:
                outputs_dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise OSError(f"Error creating outputs directory: {e}")
        return outputs_dir_path

    def get_model_path(self, model_id: str) -> Path:
        # Find the model file for a specific model ID and ignore files with "2015" in the name
        model_pattern = f"*{model_id}_*.osim"
        try:
            all_matches = list(self.model_base_dir.glob(model_pattern))
            # Filter out files where model_id is part of a year (e.g., "2015" contains "015")
            # Keep only files where model_id appears as a standalone number after path separator
            model_search_results = []
            for path in all_matches:
                # Check if the model_id appears right after a separator or at the start
                filename = path.name
                # Look for pattern: start of string or '/' or '_' followed by model_id followed by '_'
                if filename.startswith(f"{model_id}_") or f"_{model_id}_" in filename:
                    model_search_results.append(path)
        except Exception as e:
            raise OSError(f"Error searching for model {model_id}: {e}")

        if not model_search_results:
            raise FileNotFoundError(
                f"No model file found for ID {model_id} in {self.model_base_dir}",
            )

        if len(model_search_results) > 1:
            print(
                f"Multiple model files found for ID {model_id} in {self.model_base_dir}. {model_search_results[0]} will be used.",
            )

        found_path = model_search_results[0]

        if not found_path.is_file():
            raise FileNotFoundError(f"Model file {found_path} is not a file.")

        return found_path

    def get_adjusted_ceinms_model_path(self, subject_id: str) -> Path:
        model_pattern = f"{subject_id}_Rajagopal2015_FAI_originalMass_opt_N10_hans.xml"
        try:
            model_search_results = list(self.model_base_dir.rglob(model_pattern))
        except Exception as e:
            raise OSError(
                f"Error searching for adjusted CEINMS model for subject {subject_id}: {e}",
            )
        if not model_search_results:
            raise FileNotFoundError(
                f"No adjusted CEINMS model file found for subject ID {subject_id} in {self.model_base_dir}",
            )
        if len(model_search_results) > 1:
            print(
                f"Multiple adjusted CEINMS model files found for subject ID {subject_id} in {self.model_base_dir}. {model_search_results[0]} will be used.",
            )

        return model_search_results[0]

    def get_marker_file_path(self, run_id: str, trial_name: str) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            marker_file = output_dir.glob("markers.trc")

            # if multiple marker files are found, raise an error
            marker_file_list = list(marker_file)
            if not marker_file_list:
                raise FileNotFoundError(
                    f"No marker file found for trial {trial_name} in run {run_id}.",
                )
            if len(marker_file_list) > 1:
                raise ValueError(
                    f"Multiple marker files found for trial {trial_name} in run {run_id}. Please ensure only one marker file exists.",
                )
            return marker_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Marker file for trial {trial_name} in run {run_id} could not be found: {e}",
            )

    def get_ik_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            ik_file = output_dir.glob("ik.mot")
            # if mzultiple ik files are found, raise an error
            ik_file_list = list(ik_file)
            if not ik_file_list:
                raise FileNotFoundError(
                    f"No IK file found for trial {trial_name} in subject {run_id}.",
                )
            if len(ik_file_list) > 1:
                raise ValueError(
                    f"Multiple IK files found for trial {trial_name} in subject {run_id}. Please ensure only one IK file exists.",
                )
            return ik_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"IK file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_id_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            id_file = output_dir.glob("id.sto")
            # if mzultiple id files are found, raise an error
            id_file_list = list(id_file)
            if not id_file_list:
                raise FileNotFoundError(
                    f"No ID file found for trial {trial_name} in subject {run_id}.",
                )
            if len(id_file_list) > 1:
                raise ValueError(
                    f"Multiple ID files found for trial {trial_name} in subject {run_id}. Please ensure only one ID file exists.",
                )
            return id_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"ID file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_external_loads_xml_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            external_loads_file = output_dir.glob("externalloads.xml")
            # if multiple external loads files are found, raise an error
            external_loads_file_list = list(external_loads_file)
            if not external_loads_file_list:
                raise FileNotFoundError(
                    f"No external loads file found for trial {trial_name} in subject {run_id}.",
                )
            if len(external_loads_file_list) > 1:
                raise ValueError(
                    f"Multiple external loads files found for trial {trial_name} in subject {run_id}. Please ensure only one external loads file exists.",
                )
            return external_loads_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"External loads file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_external_loads_mot_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            external_loads_file = output_dir.glob("externalloads.mot")
            # if multiple external loads files are found, raise an error
            external_loads_file_list = list(external_loads_file)
            if not external_loads_file_list:
                raise FileNotFoundError(
                    f"No external loads file found for trial {trial_name} in subject {run_id}.",
                )
            if len(external_loads_file_list) > 1:
                raise ValueError(
                    f"Multiple external loads files found for trial {trial_name} in subject {run_id}. Please ensure only one external loads file exists.",
                )
            return external_loads_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"External loads file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_actuators_so_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            actuators_so_file = output_dir.glob("actuators_so.xml")
            # if multiple actuators so files are found, raise an error
            actuators_so_file_list = list(actuators_so_file)
            if not actuators_so_file_list:
                raise FileNotFoundError(
                    f"No actuators so file found for trial {trial_name} in subject {run_id}.",
                )
            if len(actuators_so_file_list) > 1:
                raise ValueError(
                    f"Multiple actuators so files found for trial {trial_name} in subject {run_id}. Please ensure only one actuators so file exists.",
                )
            return actuators_so_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Actuators so file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_so_forces_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            so_forces_file = output_dir.glob("*_force.sto")
            # if multiple so forces files are found, raise an error
            so_forces_file_list = list(so_forces_file)
            if not so_forces_file_list:
                raise FileNotFoundError(
                    f"No SO forces file found for trial {trial_name} in subject {run_id}.",
                )
            if len(so_forces_file_list) > 1:
                raise ValueError(
                    f"Multiple SO forces files found for trial {trial_name} in subject {run_id}. Please ensure only one SO forces file exists.",
                )
            return so_forces_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"SO forces file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_mtu_length_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            mtu_length_file = output_dir.glob("*_Length.sto")
            # if multiple mtu length files are found, raise an error
            mtu_length_file_list = list(mtu_length_file)
            if not mtu_length_file_list:
                raise FileNotFoundError(
                    f"No MTU length file found for trial {trial_name} in subject {run_id}.",
                )
            if len(mtu_length_file_list) > 1:
                raise ValueError(
                    f"Multiple MTU length files found for trial {trial_name} in subject {run_id}. Please ensure only one MTU length file exists.",
                )
            return mtu_length_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"MTU length file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_fiber_length_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            fiber_length_file = output_dir.glob("*_FiberLength.sto")
            # if multiple fiber length files are found, raise an error
            fiber_length_file_list = list(fiber_length_file)
            if not fiber_length_file_list:
                raise FileNotFoundError(
                    f"No fiber length file found for trial {trial_name} in subject {run_id}.",
                )
            if len(fiber_length_file_list) > 1:
                raise ValueError(
                    f"Multiple fiber length files found for trial {trial_name} in subject {run_id}. Please ensure only one fiber length file exists.",
                )
            return fiber_length_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Fiber length file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_norm_fiber_length_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            norm_fiber_length_file = output_dir.glob("*_NormalizedFiberLength.sto")
            # if multiple norm fiber length files are found, raise an error
            norm_fiber_length_file_list = list(norm_fiber_length_file)
            if not norm_fiber_length_file_list:
                raise FileNotFoundError(
                    f"No normalized fiber length file found for trial {trial_name} in subject {run_id}.",
                )
            if len(norm_fiber_length_file_list) > 1:
                raise ValueError(
                    f"Multiple normalized fiber length files found for trial {trial_name} in subject {run_id}. Please ensure only one normalized fiber length file exists.",
                )
            return norm_fiber_length_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Normalized fiber length file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_fiber_velocity_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            fiber_velocity_file = output_dir.glob("*_FiberVelocity.sto")
            # if multiple fiber velocity files are found, raise an error
            fiber_velocity_file_list = list(fiber_velocity_file)
            if not fiber_velocity_file_list:
                raise FileNotFoundError(
                    f"No fiber velocity file found for trial {trial_name} in subject {run_id}.",
                )
            if len(fiber_velocity_file_list) > 1:
                raise ValueError(
                    f"Multiple fiber velocity files found for trial {trial_name} in subject {run_id}. Please ensure only one fiber velocity file exists.",
                )
            return fiber_velocity_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Fiber velocity file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_fiber_force_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            fiber_force_file = output_dir.glob("*_FiberForce.sto")
            # if multiple fiber force files are found, raise an error
            fiber_force_file_list = list(fiber_force_file)
            if not fiber_force_file_list:
                raise FileNotFoundError(
                    f"No fiber force file found for trial {trial_name} in subject {run_id}.",
                )
            if len(fiber_force_file_list) > 1:
                raise ValueError(
                    f"Multiple fiber force files found for trial {trial_name} in subject {run_id}. Please ensure only one fiber force file exists.",
                )
            return fiber_force_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Fiber force file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_muscle_activation_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name) / "Optimised"
            muscle_activation_file = output_dir.glob("Activations.sto")
            # if multiple muscle activation files are found, raise an error
            muscle_activation_file_list = list(muscle_activation_file)
            if not muscle_activation_file_list:
                raise FileNotFoundError(
                    f"No muscle activation file found for trial {trial_name} in subject {run_id}.",
                )
            if len(muscle_activation_file_list) > 1:
                raise ValueError(
                    f"Multiple muscle activation files found for trial {trial_name} in subject {run_id}. Please ensure only one muscle activation file exists.",
                )
            return muscle_activation_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Muscle activation file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_emg_file_path(self, run_id: str, trial_name: str) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            emg_file = output_dir.glob("emg.mot")
            # if multiple emg files are found, raise an error
            emg_file_list = list(emg_file)
            if not emg_file_list:
                raise FileNotFoundError(
                    f"No EMG file found for trial {trial_name} in subject {run_id}.",
                )
            if len(emg_file_list) > 1:
                raise ValueError(
                    f"Multiple EMG files found for trial {trial_name} in subject {run_id}. Please ensure only one EMG file exists.",
                )
            return emg_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"EMG file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_ceinms_muscle_forces_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            ceinms_output_dir = output_dir / "Optimised"
            ceinms_muscles_forces_file = ceinms_output_dir.glob("*MuscleForces.sto")
            # if multiple ceinms muscles forces files are found, raise an error
            ceinms_muscles_forces_file_list = list(ceinms_muscles_forces_file)
            if not ceinms_muscles_forces_file_list:
                raise FileNotFoundError(
                    f"No CEINMS muscles forces file found for trial {trial_name} in subject {run_id}.",
                )
            if len(ceinms_muscles_forces_file_list) > 1:
                raise ValueError(
                    f"Multiple CEINMS muscles forces files found for trial {trial_name} in subject {run_id}. Please ensure only one CEINMS muscles forces file exists.",
                )
            return ceinms_muscles_forces_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CEINMS muscles forces file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_ceinms_active_muscle_forces_file_path(
        self,
        run_id: str,
        trial_name,
    ) -> Path:
        output_dir = self.get_output_dir(run_id, trial_name)
        ceinms_output_dir = output_dir / "Optimised"
        ceinms_active_muscle_forces_file = ceinms_output_dir.glob(
            "MuscleForcesActive.sto",
        )
        # if multiple ceinms active muscle forces files are found, raise an error
        ceinms_active_muscle_forces_file_list = list(ceinms_active_muscle_forces_file)
        if not ceinms_active_muscle_forces_file_list:
            raise FileNotFoundError(
                f"No CEINMS active muscle forces file found for trial {trial_name} in subject {run_id}.",
            )
        if len(ceinms_active_muscle_forces_file_list) > 1:
            raise ValueError(
                f"Multiple CEINMS active muscle forces files found for trial {trial_name} in subject {run_id}. Please ensure only one CEINMS active muscle forces file exists.",
            )
        return ceinms_active_muscle_forces_file_list[0]

    def get_ceinms_passive_muscle_forces_file_path(
        self,
        run_id: str,
        trial_name,
    ) -> Path:
        output_dir = self.get_output_dir(run_id, trial_name)
        ceinms_output_dir = output_dir / "Optimised"
        ceinms_passive_muscle_forces_file = ceinms_output_dir.glob(
            "MuscleForcesPassive.sto",
        )
        # if multiple ceinms passive muscle forces files are found, raise an error
        ceinms_passive_muscle_forces_file_list = list(ceinms_passive_muscle_forces_file)
        if not ceinms_passive_muscle_forces_file_list:
            raise FileNotFoundError(
                f"No CEINMS passive muscle forces file found for trial {trial_name} in subject {run_id}.",
            )
        if len(ceinms_passive_muscle_forces_file_list) > 1:
            raise ValueError(
                f"Multiple CEINMS passive muscle forces files found for trial {trial_name} in subject {run_id}. Please ensure only one CEINMS passive muscle forces file exists.",
            )
        return ceinms_passive_muscle_forces_file_list[0]

    def get_ceinms_fiber_lengths_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            ceinms_output_dir = output_dir / "Optimised"
            ceinms_fiber_lengths_file = ceinms_output_dir.glob("FibreLengths.sto")
            # if multiple ceinms fiber lengths files are found, raise an error
            ceinms_fiber_lengths_file_list = list(ceinms_fiber_lengths_file)
            if not ceinms_fiber_lengths_file_list:
                raise FileNotFoundError(
                    f"No CEINMS fiber lengths file found for trial {trial_name} in subject {run_id}.",
                )
            if len(ceinms_fiber_lengths_file_list) > 1:
                raise ValueError(
                    f"Multiple CEINMS fiber lengths files found for trial {trial_name} in subject {run_id}. Please ensure only one CEINMS fiber lengths file exists.",
                )
            return ceinms_fiber_lengths_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CEINMS fiber lengths file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_ceinms_normalized_fiber_lengths_file_path(
        self,
        run_id: str,
        trial_name,
    ) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            ceinms_output_dir = output_dir / "Optimised"
            ceinms_normalized_fiber_lengths_file = ceinms_output_dir.glob(
                "NormFibreLengths.sto",
            )
            # if multiple ceinms normalized fiber lengths files are found, raise an error
            ceinms_normalized_fiber_lengths_file_list = list(
                ceinms_normalized_fiber_lengths_file,
            )
            if not ceinms_normalized_fiber_lengths_file_list:
                raise FileNotFoundError(
                    f"No CEINMS normalized fiber lengths file found for trial {trial_name} in subject {run_id}.",
                )
            if len(ceinms_normalized_fiber_lengths_file_list) > 1:
                raise ValueError(
                    f"Multiple CEINMS normalized fiber lengths files found for trial {trial_name} in subject {run_id}. Please ensure only one CEINMS normalized fiber lengths file exists.",
                )
            return ceinms_normalized_fiber_lengths_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CEINMS normalized fiber lengths file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_ceinms_tendon_lengths_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            ceinms_output_dir = output_dir / "Optimised"
            ceinms_tendon_lengths_file = ceinms_output_dir.glob("TendonLengths.sto")
            # if multiple ceinms tendon lengths files are found, raise an error
            ceinms_tendon_lengths_file_list = list(ceinms_tendon_lengths_file)
            if not ceinms_tendon_lengths_file_list:
                raise FileNotFoundError(
                    f"No CEINMS tendon lengths file found for trial {trial_name} in subject {run_id}.",
                )
            if len(ceinms_tendon_lengths_file_list) > 1:
                raise ValueError(
                    f"Multiple CEINMS tendon lengths files found for trial {trial_name} in subject {run_id}. Please ensure only one CEINMS tendon lengths file exists.",
                )
            return ceinms_tendon_lengths_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CEINMS tendon lengths file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_ceinms_normalized_tendon_lengths_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            ceinms_output_dir = output_dir / "Optimised"
            ceinms_normalized_tendon_lengths_file = ceinms_output_dir.glob("NormTendonLengths.sto")
            # if multiple ceinms normalized tendon lengths files are found, raise an error
            ceinms_normalized_tendon_lengths_file_list = list(ceinms_normalized_tendon_lengths_file)
            if not ceinms_normalized_tendon_lengths_file_list:
                raise FileNotFoundError(
                    f"No CEINMS normalized tendon lengths file found for trial {trial_name} in subject {run_id}.",
                )
            if len(ceinms_normalized_tendon_lengths_file_list) > 1:
                raise ValueError(
                    f"Multiple CEINMS normalized tendon lengths files found for trial {trial_name} in subject {run_id}. Please ensure only one CEINMS normalized tendon lengths file exists.",
                )
            return ceinms_normalized_tendon_lengths_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CEINMS normalized tendon lengths file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_ceinms_fiber_velocities_file_path(self, run_id: str, trial_name) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name)
            ceinms_output_dir = output_dir / "Optimised"
            ceinms_fiber_velocities_file = ceinms_output_dir.glob("FibreVelocities.sto")
            # if multiple ceinms fiber velocities files are found, raise an error
            ceinms_fiber_velocities_file_list = list(ceinms_fiber_velocities_file)
            if not ceinms_fiber_velocities_file_list:
                raise FileNotFoundError(
                    f"No CEINMS fiber velocities file found for trial {trial_name} in subject {run_id}.",
                )
            if len(ceinms_fiber_velocities_file_list) > 1:
                raise ValueError(
                    f"Multiple CEINMS fiber velocities files found for trial {trial_name} in subject {run_id}. Please ensure only one CEINMS fiber velocities file exists.",
                )
            return ceinms_fiber_velocities_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CEINMS fiber velocities file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_ceinms_setup_file_path(self, run_id: str, trial_name) -> Path:
        try:
            sim_dir = self.get_trial_dir(run_id, trial_name)
            ceinms_setup_file_list = list(sim_dir.glob("setup_ceinms_optimise.xml"))
            if not ceinms_setup_file_list:
                raise FileNotFoundError(
                    f"No CEINMS setup file found for trial {trial_name} in subject {run_id}.",
                )
            if len(ceinms_setup_file_list) > 1:
                raise ValueError(
                    f"Multiple CEINMS setup files found for trial {trial_name} in subject {run_id}. Please ensure only one CEINMS setup file exists.",
                )
            return ceinms_setup_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CEINMS setup file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_ceinms_adjusted_emgs_file_path(self, run_id: str, trial_name: str) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name) / "Optimised"

            adjusted_emgs_file = output_dir.glob("AdjustedEmgs.sto")
            # if multiple muscle activation files are found, raise an error
            adjusted_emgs_file_list = list(adjusted_emgs_file)
            if not adjusted_emgs_file_list:
                raise FileNotFoundError(
                    f"No CEINMS adjusted EMGs file found for trial {trial_name} in subject {run_id}.",
                )
            if len(adjusted_emgs_file_list) > 1:
                raise ValueError(
                    f"Multiple CEINMS adjusted EMGs files found for trial {trial_name} in subject {run_id}. Please ensure only one CEINMS adjusted EMGs file exists.",
                )
            return adjusted_emgs_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CEINMS adjusted EMGs file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_ceinms_torque_file_path(self, run_id: str, trial_name: str) -> Path:
        try:
            output_dir = self.get_output_dir(run_id, trial_name) / "Optimised"

            torque_file = output_dir.glob("Torques.sto")

            # if multiple torque files are found, raise an error
            torque_file_list = list(torque_file)
            if not torque_file_list:
                raise FileNotFoundError(
                    f"No CEINMS torque file found for trial {trial_name} in subject {run_id}.",
                )
            if len(torque_file_list) > 1:
                raise ValueError(
                    f"Multiple CEINMS torque files found for trial {trial_name} in subject {run_id}. Please ensure only one CEINMS torque file exists.",
                )
            return torque_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CEINMS torque file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_ceinms_cal_model_file_path(self, run_id: str) -> Path:
        sim_dir = self.get_sim_run_dir(run_id) / "pre"
        ceinms_cal_model = list(sim_dir.glob("subjectCalibrated_hybrid.xml"))
        if not ceinms_cal_model:
            raise FileNotFoundError(
                f"No CEINMS calibrated model file found in run {run_id}.",
            )
        if len(ceinms_cal_model) > 1:
            raise ValueError(
                f"Multiple CEINMS calibrated model files found in run {run_id}. Please ensure only one CEINMS calibrated model file exists.",
            )
        return ceinms_cal_model[0]

    def get_ceinms_uncal_model_file_path(self, run_id: str) -> Path:
        sim_dir = self.get_sim_run_dir(run_id) / "pre"
        ceinms_uncal_model = list(sim_dir.glob("subjectUncalibrated.xml"))
        if not ceinms_uncal_model:
            raise FileNotFoundError(
                f"No CEINMS uncalibrated model file found in run {run_id}.",
            )
        if len(ceinms_uncal_model) > 1:
            raise ValueError(
                f"Multiple CEINMS uncalibrated model files found in run {run_id}. Please ensure only one CEINMS uncalibrated model file exists.",
            )
        return ceinms_uncal_model[0]

    def get_marker_errors_all_file_path(self, subject_id: str, trial_name: str) -> Path:
        try:
            trial_dir = self.get_trial_dir(subject_id, trial_name)
            marker_errors_all_file_list = list(trial_dir.glob("*marker_errors_all.sto"))
            if not marker_errors_all_file_list:
                raise FileNotFoundError(
                    f"No marker errors all file found for trial {trial_name} in subject {subject_id}.",
                )
            if len(marker_errors_all_file_list) > 1:
                raise ValueError(
                    f"Multiple marker errors all files found for trial {trial_name} in subject {subject_id}. Please ensure only one marker errors all file exists.",
                )
            return marker_errors_all_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Marker errors all file for trial {trial_name} in subject {subject_id} could not be found: {e}",
            )

    def get_c3d_file_path(self, run_id: str, trial_name: str) -> Path:
        try:
            trial_dir = self.get_trial_dir(run_id, trial_name)
            c3d_file_list = list(trial_dir.glob("*.c3d"))
            if not c3d_file_list:
                raise FileNotFoundError(
                    f"No C3D file found for trial {trial_name} in subject {run_id}.",
                )
            if len(c3d_file_list) > 1:
                raise ValueError(
                    f"Multiple C3D files found for trial {trial_name} in subject {run_id}. Please ensure only one C3D file exists.",
                )
            return c3d_file_list[0]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"C3D file for trial {trial_name} in subject {run_id} could not be found: {e}",
            )

    def get_master_results_file_path(self) -> Path:
        master_results_path = Path("data") / "master_results.csv"
        if not master_results_path.is_file():
            raise FileNotFoundError(
                f"Master results file {master_results_path} not found.",
            )
        return master_results_path

    def get_master_results_non_normalized_file_path(self) -> Path:
        master_results_non_normalized_path = Path("data") / "master_results_no_time_norm.csv"
        if not master_results_non_normalized_path.is_file():
            raise FileNotFoundError(
                f"Master non-normalized results file {master_results_non_normalized_path} not found.",
            )
        return master_results_non_normalized_path

    def load_trc_file(self, path: Path, output: int = 0) -> pd.DataFrame:
        """
        Load a TRC (Track Row Column) file into a pandas DataFrame.

        Args:
            path: Path to the TRC file
            output: If 1, prints column names for debugging

        Returns:
            pandas DataFrame with the TRC data

        """
        if not path.exists():
            raise FileNotFoundError(f"TRC file not found at {path}")

        # Find line with '#Frame' to skip the header
        header_line = None
        try:
            with open(path) as file:
                for i, line in enumerate(file):
                    if "Frame#" in line:
                        header_line = i
                        break
        except Exception as e:
            raise OSError(f"Error reading file at {path}: {e}")

        if header_line is None:
            raise ValueError("Could not find 'Frame#' header line in TRC file")

        # Read headers and sub-headers
        try:
            with open(path) as file:
                lines = file.readlines()
                headers = lines[header_line].strip().split("\t")
                sub_headers = lines[header_line + 1].strip().split("\t")
        except Exception as e:
            raise OSError(f"Error reading headers from file at {path}: {e}")

        # Keep first two empty for Frame# and Time
        sub_headers = [""] * 2 + sub_headers[2:]

        # Combine headers
        for idx in range(len(headers)):
            if headers[idx] == "" and idx > 0:
                headers[idx] = headers[idx - 1]

        for idx in range(len(headers)):
            if idx < len(sub_headers) and sub_headers[idx] != "":
                headers[idx] = f"{headers[idx]}_{sub_headers[idx]}"

        # Read the data
        try:
            data = pd.read_csv(path, sep=r"\s+", header=header_line + 1)
            data.columns = headers
        except Exception as e:
            raise OSError(f"Error reading data from file at {path}: {e}")

        if output == 1:
            print(data.columns)

        return data

    def load_sto_file(self, filepath: Path) -> pd.DataFrame:
        """
        Load a .sto file into a pandas DataFrame.
        Assumes the file format from OpenSim.

        Args:
            filepath: Path to the .sto file

        Returns:
            pandas DataFrame with the .sto file data

        """
        if not filepath.exists():
            raise FileNotFoundError(f"STO file not found at {filepath}")

        header_line = None
        try:
            with open(filepath) as f:
                for i, line in enumerate(f):
                    if "endheader" in line:
                        header_line = i + 1
                        break
        except Exception as e:
            raise OSError(f"Error reading file at {filepath}: {e}")

        if header_line is None:
            raise ValueError("STO file 'endheader' not found")

        # Read the data, skipping metadata
        try:
            df = pd.read_csv(filepath, sep="\t", header=header_line)
            # Rename time column to be consistent
            if "time" in df.columns:
                df = df.rename(columns={"time": "Time"})
        except Exception as e:
            raise OSError(f"Error reading data from STO file at {filepath}: {e}")

        return df

    def write_sto_header(self, writer, dataFrame):
        """
        Writes the header for a .sto file.

        Args:
            writer (TextIOWrapper): The file writer object.
            dataFrame (pd.DataFrame): The DataFrame containing the data.

        """
        writer.write(f"{os.path.basename(writer.name)}\n")
        writer.write("version=1\n")
        writer.write(f"nRows={dataFrame.shape[0]}\n")
        writer.write(f"nColumns={dataFrame.shape[1]}\n")
        writer.write("in_degrees=yes\n")
        writer.write("endheader\n")

    def write_mot_header(self, writer, dataFrame):
        """
        Writes the header for a .mot file.

        Args:
            writer (TextIOWrapper): The file writer object.
            dataFrame (pd.DataFrame): The DataFrame containing the data.

        """
        writer.write(f"{os.path.basename(writer.name)}\n")
        writer.write("version=1\n")
        writer.write(f"nRows={dataFrame.shape[0]}\n")
        writer.write(f"nColumns={dataFrame.shape[1] + 1}\n")
        writer.write("in_degrees=yes\n")
        writer.write("endheader\n")

    def write_sto_file(self, dataFrame, filepath: Path) -> None:
        """
        Writes a pandas DataFrame to a .sto file with a specified header.

        Args:
            dataFrame (pd.DataFrame): The DataFrame to write.
            file_path (str): The path where the .sto file will be saved.
            header (list): A list of strings representing the header lines to write.

        """
        if not filepath.parent.exists():
            os.makedirs(filepath.parent)
            print(f"Created directory: {filepath.parent}")

        # make time lowercase
        if "Time" in dataFrame.columns:
            dataFrame = dataFrame.rename(columns={"Time": "time"})

        with open(filepath, "w", newline="") as f:
            if filepath.suffix == ".mot":
                self.write_mot_header(f, dataFrame)
            elif filepath.suffix == ".sto":
                self.write_sto_header(f, dataFrame)

            # bring time column to front
            dataFrame = dataFrame[["time"] + [col for col in dataFrame.columns if col != "time"]]

            # Write the data without extra line spaces
            dataFrame.to_csv(f, sep="\t", index=False, float_format="%.6f")

    def update_sto_file(self, filepath: Path) -> None:
        """
        Update an existing .sto file with new data from a DataFrame.

        Args:
            filepath: Path to the existing .sto file
            new_data_df: DataFrame with new data to write

        """
        if not filepath.exists():
            raise FileNotFoundError(f"STO file not found at {filepath}")

        try:
            df = self.load_sto_file(filepath)
        except Exception as e:
            raise OSError(f"Error updating STO file at {filepath}: {e}")

        # Write the updated DataFrame back to the .sto file
        self.write_sto_file(df, filepath)

    @staticmethod
    def ceinms_setup_organizer(subject_id: str, trial_name=None, leg=None) -> None:
        import xml.etree.ElementTree as ET

        import opensim as osim

        from utils.subject_utils import Subject_utils

        if leg is None:
            leg = Subject_utils().get_measured_leg(subject_id).lower()  # 'l' or 'r'
        # Get ceinms_cfg_optimise
        ceinms_cfg_optimise = Path(f"ceinms/setupFiles/ceinms_cfg_optimise_{leg}.xml")

        # Get excitation_generator
        excitation_generator = Path(f"ceinms/setupFiles/excitationGenerator_{leg}.xml")
        # Get the trial directory
        if trial_name is None:
            raise ValueError("Trial name must be provided.")

        # copy files to correct folder
        trial_dir = File_utils().get_trial_dir(subject_id, trial_name)
        session_dir = trial_dir.parent
        shutil.copy(ceinms_cfg_optimise, trial_dir / "ceinms_cfg_optimise.xml")
        shutil.copy(excitation_generator, session_dir / "excitationGenerator.xml")

        print(
            f"Copied {ceinms_cfg_optimise} to {trial_dir / 'ceinms_cfg_optimise.xml'} and {excitation_generator} to {session_dir / 'excitationGenerator.xml'}",
        )

        # Get input Data and write correct time
        input_Data = Path("ceinms/setupFiles/inputData.xml")
        ik_path = File_utils().get_ik_file_path(subject_id, trial_name)
        coordinates = osim.Storage(str(ik_path))
        time = [coordinates.getFirstTime(), coordinates.getLastTime()]
        time_input = f"{time[0]} {time[1]}"
        tree = ET.parse(input_Data)
        root = tree.getroot()
        elem = root.find("startStopTime")
        if elem is not None:
            if elem is not None:
                elem.text = time_input
                dst_inputData_file = trial_dir / "inputData.xml"
                tree.write(dst_inputData_file)

    def get_subject_demographics_file_path(self) -> Path:
        subject_demographics_path = Path("data") / "subject_demographics.csv"
        if not subject_demographics_path.is_file():
            raise FileNotFoundError(
                f"Subject demographics file {subject_demographics_path} not found.",
            )
        return subject_demographics_path


# if __name__ == "__main__":
#     file_utils = File_utils()
#     # ask user with of the utils to use
#     functionList = dir(file_utils)

#     for _, func in enumerate(functionList):
#         if not func.startswith("_"):
#             print(f"{func}")

#     while True:
#         command = input("Select an option from the selection:")

#         if command == "load_sto_file":
#             path = input("Enter the path to the .sto file:")
#             df = file_utils.load_sto_file(Path(path))
#             print(df.head())
#             continue

#         if command == "update_sto_file":
#             path = input("Enter the output path for the .sto file:")
#             file_utils.update_sto_file(Path(path))
#             print(f"STO file updated at {path}")
#             continue

#         try:
#             file_utils.__getattribute__(command)()
#         except AttributeError:
#             print(f"Function {command} not found in File_utils class.")
#         except TypeError as e:
#             print(f"Error calling function {command}: {e}")
#             print(f"check lines {e.__traceback__.tb_lineno}")
#             print("No all functions are fully developed ...")
