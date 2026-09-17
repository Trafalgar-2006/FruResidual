from fruresidual.config import load_config


def test_config_loads_paths():
    cfg = load_config()
    assert cfg.data_raw.name == "raw"
    assert cfg.neuprint_dataset == "male-cns:v1.0"
