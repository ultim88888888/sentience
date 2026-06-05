"""Auth header/cookie construction. 1Password (`op`) subprocess is mocked."""
from unittest.mock import patch

from scrapers.linkedin.auth import Auth, load_auth


def test_voyager_headers_carry_cookie_and_csrf():
    auth = Auth(scrapedo_token="TOK", li_at="LIAT", jsessionid="ajax:123")
    h = auth.voyager_headers()
    assert h["Csrf-Token"] == "ajax:123"
    assert h["Cookie"] == 'li_at=LIAT; JSESSIONID="ajax:123"'
    assert h["X-RestLi-Protocol-Version"] == "2.0.0"
    assert "User-Agent" in h


def test_load_auth_reads_three_secrets_and_strips_jsessionid_quotes():
    # `op` returns each item's credential in order; JSESSIONID may come quoted.
    with patch("scrapers.linkedin.auth.subprocess.check_output",
               side_effect=["TOK\n", "LIAT\n", '"ajax:123"\n']) as m:
        auth = load_auth()
    assert auth.scrapedo_token == "TOK"
    assert auth.li_at == "LIAT"
    assert auth.jsessionid == "ajax:123"   # surrounding quotes stripped
    assert m.call_count == 3
