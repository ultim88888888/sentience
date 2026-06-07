import pandas as pd
from attribution.roster import Roster

def _roster():
    df = pd.DataFrame([
        {"slug": "tim-roughgarden", "name": "Tim Roughgarden", "title": "Head of Research"},
        {"slug": "chris-dixon", "name": "Chris Dixon", "title": "Managing Partner"},
        {"slug": "ali-yahya", "name": "Ali Yahya", "title": "General Partner"},
    ])
    return Roster(df)

def test_exact_match_is_a16z():
    m = _roster().resolve("Tim Roughgarden")
    assert m.slug == "tim-roughgarden" and m.is_a16z is True

def test_last_name_match():
    assert _roster().resolve("Roughgarden").slug == "tim-roughgarden"

def test_case_insensitive():
    assert _roster().resolve("chris dixon").slug == "chris-dixon"

def test_unknown_is_external_guest():
    m = _roster().resolve("Ron Rothblum")
    assert m.slug is None and m.is_a16z is False

def test_ambiguous_last_name_returns_none_match():
    df = pd.DataFrame([
        {"slug": "j-lee", "name": "Jay Lee", "title": "x"},
        {"slug": "k-lee", "name": "Kim Lee", "title": "y"},
    ])
    m = Roster(df).resolve("Lee")
    assert m.slug is None and m.ambiguous is True

def test_empty_name_is_not_a16z():
    m = _roster().resolve(None)
    assert m.slug is None and m.is_a16z is False

def test_first_name_promotes_only_when_corroborated():
    # A bare first name ("Thanks, Tim") must NOT be tagged a16z on its own — an audience
    # member or guest with the same first name would be misattributed (corroboration fix).
    assert _roster().resolve("Tim").is_a16z is False        # uncorroborated -> external
    # ...but resolves once corroborated present in the episode (allow) or via host-prior.
    assert _roster().resolve("Tim", allow=["tim-roughgarden"]).slug == "tim-roughgarden"
    assert _roster().resolve("Tim", prefer=["tim-roughgarden"]).slug == "tim-roughgarden"

def test_ambiguous_first_name_disambiguated_by_prior():
    df = pd.DataFrame([
        {"slug": "justin-thaler", "name": "Justin Thaler", "title": "Research"},
        {"slug": "justin-other", "name": "Justin Other", "title": "x"},
    ])
    r = Roster(df)
    assert r.resolve("Justin").ambiguous is True                       # tie, no prior
    assert r.resolve("Justin", prefer=["justin-thaler"]).slug == "justin-thaler"  # prior breaks tie

def test_last_name_still_wins_for_single_token():
    assert _roster().resolve("Roughgarden").slug == "tim-roughgarden"
