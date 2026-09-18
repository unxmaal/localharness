"""Which server produced the tokens.

The gateway exists so a new inference method costs a config line plus an eval
run rather than a redesign. That claim has never been tested: mlx_lm.server is
the only engine this project has served, so nothing ever needed to say which
one a receipt came from.

It matters the moment a second one appears. `evals.core.comparable()` decides
whether two runs may share a table, and with no record of the engine two runs
across DIFFERENT servers look like the same exam -- the same failure as the
same card under two operating systems producing one accelerator string while
the instruments differed. Issue #190.

The HOST is deliberately not part of this: comparable() already argues that the
same aliases served from another machine answer the same questions. What
changes the exam is the implementation, not the address.
"""
from __future__ import annotations

import os

#: Set this when serving the text lane through something else. A config line,
#: which is the whole claim under test.
ENV_VAR = "TEXT_ENGINE"

#: What scripts/serve-mlx.sh starts. Kept here rather than only in the shell so
#: the receipt and the launcher cannot disagree; a test asserts they match.
DEFAULT = "mlx_lm.server"


def text_engine(environ=None) -> str:
    """The name of the server behind the text lane."""
    environ = os.environ if environ is None else environ
    return (environ.get(ENV_VAR) or "").strip() or DEFAULT
