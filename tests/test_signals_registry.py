from signals.registry import Registry, load_registry, save_registry
from signals import config

def test_seed_registry_contains_seed_sectors():
    reg = Registry.seed()
    assert "zk" in reg.sectors
    assert reg.tokens == []

def test_mint_appends_and_is_idempotent():
    reg = Registry.seed()
    n = len(reg.sectors)
    reg.mint_sector("intent-solvers")
    reg.mint_sector("intent-solvers")  # idempotent
    assert reg.sectors.count("intent-solvers") == 1
    assert len(reg.sectors) == n + 1

def test_mint_token_records_parent():
    reg = Registry.seed()
    reg.mint_token("HYPE", parent_sector="perp-dex")
    assert "HYPE" in reg.tokens
    assert reg.token_parent["HYPE"] == "perp-dex"

def test_save_load_roundtrip(tmp_path):
    reg = Registry.seed()
    reg.mint_sector("intent-solvers")
    reg.mint_token("HYPE", parent_sector="perp-dex")
    p = tmp_path / "registry.json"
    save_registry(reg, p)
    reg2 = load_registry(p)
    assert reg2.sectors == reg.sectors
    assert reg2.token_parent == reg.token_parent

def test_load_missing_returns_seed(tmp_path):
    reg = load_registry(tmp_path / "nope.json")
    assert "zk" in reg.sectors
