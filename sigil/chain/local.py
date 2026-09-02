"""An in-process EVM (py-evm) whose state survives between CLI invocations.

eth-tester chains are normally ephemeral, which would make ``sigil anchor`` and
``sigil verify`` meaningless as separate commands - verify would run against a
fresh, empty chain. So the chain's underlying key/value store is snapshotted to
disk after every write and rehydrated on load. The result behaves like a real
node for our purposes: state is durable, history is append-only, and a record
written by one process is genuinely read back from chain state by another.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eth_tester import EthereumTester, PyEVMBackend


class PersistentLocalChain:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.backend = PyEVMBackend()
        self.meta: dict[str, Any] = {}
        self._load()
        self.tester = EthereumTester(backend=self.backend)

    # -- persistence -----------------------------------------------------

    def _kv(self) -> dict:
        return self.backend.chain.chaindb.db.wrapped_db.kv_store

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        # An unreadable snapshot starts the chain from genesis rather than
        # refusing to run. Decoding is inside the try for the same reason the
        # read is: a truncated or hand-edited file fails on malformed hex or a
        # non-mapping just as easily as on bad JSON, and neither is worth a
        # traceback. `sigil chain info` reports the record count, so an empty
        # chain is visible rather than silent.
        try:
            blob = json.loads(self.state_path.read_text())
            kv = {bytes.fromhex(k): bytes.fromhex(v)
                  for k, v in blob.get("kv", {}).items()}
        except (json.JSONDecodeError, OSError, ValueError, AttributeError):
            return
        if not kv:
            return
        store = self._kv()
        store.clear()
        store.update(kv)
        # Rebuild the chain object against the restored db so it picks up the
        # persisted canonical head instead of the genesis it was born with.
        chain_class = type(self.backend.chain)
        self.backend.chain = chain_class(self.backend.chain.chaindb.db)
        self.meta = blob.get("meta", {})

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "meta": self.meta,
            "kv": {k.hex(): v.hex() for k, v in self._kv().items()},
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob))
        tmp.replace(self.state_path)  # atomic: a crash mid-write cannot corrupt state

    def reset(self) -> None:
        if self.state_path.exists():
            self.state_path.unlink()
        self.backend = PyEVMBackend()
        self.tester = EthereumTester(backend=self.backend)
        self.meta = {}
