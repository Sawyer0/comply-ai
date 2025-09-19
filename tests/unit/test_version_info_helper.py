from src.llama_mapper.versioning.version_manager import VersionManager


def test_version_info_dict_has_keys():
    vm = VersionManager()
    d = vm.get_version_info_dict()
    assert set(["taxonomy", "frameworks", "model"]).issubset(d.keys())
    assert isinstance(d["taxonomy"], str)
    assert isinstance(d["frameworks"], str)
    assert isinstance(d["model"], str)
