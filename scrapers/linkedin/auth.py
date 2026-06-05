"""Read LinkedIn auth secrets from 1Password and build Voyager request headers."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .config import (JSESSIONID_OP_ITEM, LI_AT_OP_ITEM, OP_VAULT,
                     SCRAPEDO_OP_ITEM, UA)


def _op_read(item: str) -> str:
    return subprocess.check_output(
        ["op", "item", "get", item, "--vault", OP_VAULT,
         "--fields", "credential", "--reveal"],
        text=True,
    ).strip()


@dataclass
class Auth:
    scrapedo_token: str
    li_at: str
    jsessionid: str   # raw value, no surrounding quotes

    def voyager_headers(self) -> dict[str, str]:
        # scrape.do forwards these upstream because we pass customHeaders=true.
        # LinkedIn requires Csrf-Token == the JSESSIONID cookie value.
        return {
            "User-Agent": UA,
            "Accept": "application/json",
            "Csrf-Token": self.jsessionid,
            "X-RestLi-Protocol-Version": "2.0.0",
            "Cookie": f'li_at={self.li_at}; JSESSIONID="{self.jsessionid}"',
        }


def load_auth() -> Auth:
    return Auth(
        scrapedo_token=_op_read(SCRAPEDO_OP_ITEM),
        li_at=_op_read(LI_AT_OP_ITEM),
        jsessionid=_op_read(JSESSIONID_OP_ITEM).strip('"'),
    )
