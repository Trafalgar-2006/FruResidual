from fruresidual.config import load_config
from fruresidual.features.fru_dsx_join import _POSITIVE_VALUES


def test_config_loads_paths():
    cfg = load_config()
    assert cfg.data_raw.name == "raw"
    assert cfg.neuprint_dataset == "male-cns:v1.0"


def test_configured_aggregation_rule_is_actually_implemented():
    # config.yaml previously advertised a "majority" rule that was never
    # implemented (has_fru_dsx would KeyError on it), and was set to a
    # different rule than the one every script hardcoded -- so the knob
    # looked live while changing it did nothing. Now that the scripts
    # genuinely read it, an unimplemented value would break Phases 1/3/4
    # at runtime instead of being silently ignored.
    cfg = load_config()
    assert cfg.fru_dsx_aggregation in _POSITIVE_VALUES
