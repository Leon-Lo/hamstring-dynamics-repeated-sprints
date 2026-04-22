import sys
from pathlib import Path

from tests.test_file_utils import Project_root

sys.path.append(str(Path(__file__).parent.parent))
import opensim as osim

from src.model_manager import OpensimModelManager


class Test_model_manager:
    def test_get_model_path(self, monkeypatch):
        monkeypatch.chdir(Project_root)
        id = "010"
        # Test getting the model path
        model_manager = OpensimModelManager(id)
        model_path = model_manager.get_model_path()
        assert model_path is not None
        assert model_path.suffix == ".osim"

    def test_load_model(self, monkeypatch):
        monkeypatch.chdir(Project_root)
        id = "010"
        # Test loading the model
        model_manager = OpensimModelManager(id)
        model = model_manager.load_model()
        assert isinstance(model, osim.Model)

    def get_model(self, monkeypatch):
        monkeypatch.chdir(Project_root)
        id = "010"
        # Test getting the model
        model_manager = OpensimModelManager(id)
        model = model_manager.get_model()
        assert isinstance(model, osim.Model)

    def test_get_state(self, monkeypatch):
        monkeypatch.chdir(Project_root)
        id = "010"
        # Test getting the state
        model_manager = OpensimModelManager(id)
        state = model_manager.get_state()
        assert isinstance(state, osim.State)

    def test_get_muscle_groups(self, monkeypatch, capsys):
        monkeypatch.chdir(Project_root)
        id = "010"
        model_manager = OpensimModelManager(id)
        muscle_groups = model_manager.get_muscle_groups()

        assert isinstance(muscle_groups, dict)
        assert "hip_flex_r" in muscle_groups

    def test_get_ids(self, monkeypatch):

        ids = OpensimModelManager.get_ids()
        print(ids)
        assert isinstance(ids, list)
        assert len(ids) == len(set(ids))  # Ensure all IDs are unique
        assert all(isinstance(i, str) for i in ids)  # Ensure all IDs are strings
