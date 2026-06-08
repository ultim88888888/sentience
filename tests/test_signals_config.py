from signals import config

def test_seed_sectors_are_lowercase_kebab_and_nonempty():
    assert config.SEED_SECTORS
    for s in config.SEED_SECTORS:
        assert s == s.lower()
        assert " " not in s  # kebab-case ids

def test_paths_point_under_data():
    assert config.SIGNAL_OUT_DIR.parts[-2:] == ("data", "signal")
    assert config.DISTILLATE_CACHE.name == "transcript_distillates.jsonl"

def test_default_window_is_holding_period_scale():
    assert config.DEFAULT_WINDOW_MONTHS == 18
