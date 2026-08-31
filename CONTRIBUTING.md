# Contributing

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[insight,dev]"
.venv/bin/pytest -m "not network"   # fast, no network
.venv/bin/ruff check .
```

Before opening a pull request:

- `ruff check .` is clean and `pytest -m "not network"` passes.
- If you touched the search providers, also run `pytest -m network` — those are
  the tests that catch a third-party API changing shape.
- If you touched `sigil/evidence.py`, be aware that **any** change to the
  canonical serialisation invalidates every previously anchored record. That is
  a breaking change to the on-chain format, not a refactor; bump the `SCHEMA`
  string in `sigil/__init__.py` and say so in the PR.
- If you touched `contracts/SigilRegistry.sol`, delete `artifacts/` so the
  contract is recompiled, and re-run the chain tests.

Adding a search provider means implementing the `SearchProvider` protocol in
`sigil/search/base.py`: yield `Candidate` objects and record every network call
into `self.trace`. The trace is what lets a run be audited afterwards, so a
provider that does not populate it is not finished.
