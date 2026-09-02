"""One client over two chain backends: a persistent in-process EVM, or any RPC node.

The same code path anchors and verifies in both cases, so a demo run on the
local chain and a real run on Polygon Amoy exercise identical logic - only the
provider and the signing strategy differ.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from web3 import Web3
from web3.exceptions import ContractLogicError

from ..config import STATE_PATH, Config
from ..evidence import Evidence, subject_ref
from .compile import compile_registry
from .local import PersistentLocalChain


@dataclass
class Verification:
    """The result of checking a local evidence bundle against chain state."""

    evidence_hash: str
    anchored: bool
    on_chain: dict[str, Any] | None = None
    similarity_matches: bool = False
    subject_matches: bool = False
    probe_matches: bool | None = None
    source_image_intact: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Every check that was actually run has to pass.

        The optional checks are tri-state: None means "not requested", and a
        check that was not run must not be allowed to read as a pass.
        """
        core = self.anchored and self.similarity_matches and self.subject_matches
        optional = [c for c in (self.probe_matches, self.source_image_intact) if c is not None]
        return bool(core) and all(optional)


class ChainClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.backend = cfg.chain_backend
        self.local: PersistentLocalChain | None = None
        self.artifact = compile_registry()
        self.w3, self.account = self._connect()
        self.contract = None

    # -- connection ------------------------------------------------------

    def _connect(self):
        if self.backend == "local":
            from web3.providers.eth_tester import EthereumTesterProvider

            self.local = PersistentLocalChain(STATE_PATH)
            w3 = Web3(EthereumTesterProvider(self.local.tester))
            return w3, w3.eth.accounts[0]

        if not self.cfg.rpc_url:
            raise RuntimeError("SIGIL_RPC_URL is required when SIGIL_CHAIN=rpc")
        if not self.cfg.private_key:
            raise RuntimeError("SIGIL_PRIVATE_KEY is required when SIGIL_CHAIN=rpc")

        w3 = Web3(Web3.HTTPProvider(self.cfg.rpc_url, request_kwargs={"timeout": 60}))
        try:
            # Polygon, BSC and most testnets are proof-of-authority and put
            # oversized data in the extraData field, which the default block
            # formatter rejects.
            from web3.middleware import ExtraDataToPOAMiddleware

            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except ImportError:  # pragma: no cover - older web3
            pass

        if not w3.is_connected():
            raise RuntimeError(f"cannot reach RPC endpoint {self.cfg.rpc_url}")
        acct = w3.eth.account.from_key(self.cfg.private_key)
        return w3, acct

    @property
    def address(self) -> str:
        return self.account if isinstance(self.account, str) else self.account.address

    @property
    def chain_id(self) -> int:
        return self.w3.eth.chain_id

    def _remember(self, key: str, value: Any) -> None:
        if self.local:
            self.local.meta[key] = value
            self.local.save()

    def _recall(self, key: str) -> Any:
        if self.local:
            return self.local.meta.get(key)
        return None

    # -- deployment ------------------------------------------------------

    def deployed_address(self) -> str | None:
        if self.backend == "rpc":
            return self.cfg.contract_address
        return self._recall("contract_address")

    def ensure_deployed(self) -> str:
        addr = self.deployed_address()
        if addr and self._code_at(addr):
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(addr), abi=self.artifact["abi"]
            )
            return addr
        return self.deploy()

    def _code_at(self, addr: str) -> bool:
        try:
            return len(self.w3.eth.get_code(Web3.to_checksum_address(addr))) > 0
        except Exception:  # noqa: BLE001
            return False

    def deploy(self) -> str:
        factory = self.w3.eth.contract(
            abi=self.artifact["abi"], bytecode=self.artifact["bytecode"]
        )
        receipt = self._send(factory.constructor())
        addr = receipt["contractAddress"]
        self.contract = self.w3.eth.contract(address=addr, abi=self.artifact["abi"])
        self._remember("contract_address", addr)
        return addr

    # -- transactions ----------------------------------------------------

    def _send(self, fn) -> dict:
        if self.backend == "local":
            tx_hash = fn.transact({"from": self.address, "gas": 3_000_000})
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            if self.local:
                self.local.save()
            return dict(receipt)

        nonce = self.w3.eth.get_transaction_count(self.address)
        tx = fn.build_transaction(
            {
                "from": self.address,
                "nonce": nonce,
                "chainId": self.chain_id,
            }
        )
        # Let the node price the transaction, but never let a bad estimate
        # strand the run: pad gas by 25%. Estimation is also where an unfunded
        # key usually fails first - most nodes refuse to price a transaction
        # the sender cannot pay for - so both steps get the same handling.
        with self._funding_errors():
            tx["gas"] = int(self.w3.eth.estimate_gas(tx) * 1.25)
            signed = self.w3.eth.account.sign_transaction(tx, self.cfg.private_key)
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            tx_hash = self.w3.eth.send_raw_transaction(raw)
        return dict(self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300))

    @contextlib.contextmanager
    def _funding_errors(self):
        """Turn a node's funding complaint into the one instruction that fixes it."""
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is about funds
            text = str(exc).lower()
            if "insufficient funds" in text or "gas required exceeds" in text:
                raise RuntimeError(
                    f"{self.address} has no funds on chain {self.chain_id}. "
                    "Fund it from a testnet faucet (`sigil chain address` shows "
                    "the address and its balance), or use SIGIL_CHAIN=local."
                ) from exc
            raise

    def anchor(self, evidence: Evidence) -> dict[str, Any]:
        self.ensure_deployed()
        ehash = evidence.evidence_hash()
        sref = subject_ref(evidence.probe.embedding_sha256, self.cfg.subject_salt)

        if self.contract.functions.isAnchored(ehash).call():
            existing = self.lookup(ehash)
            return {
                "already_anchored": True,
                "evidence_hash": "0x" + ehash.hex(),
                "subject_ref": "0x" + sref.hex(),
                **(existing or {}),
            }

        receipt = self._send(
            self.contract.functions.anchor(ehash, evidence.similarity_bps(), sref)
        )
        tx_hash = receipt["transactionHash"]
        tx_hex = tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash)
        if not tx_hex.startswith("0x"):
            tx_hex = "0x" + tx_hex
        return {
            "already_anchored": False,
            "evidence_hash": "0x" + ehash.hex(),
            "subject_ref": "0x" + sref.hex(),
            "tx_hash": tx_hex,
            "block_number": receipt["blockNumber"],
            "gas_used": receipt["gasUsed"],
            "contract": self.contract.address,
            "chain_id": self.chain_id,
            "explorer": self._explorer(tx_hex),
        }

    def _explorer(self, tx_hex: str) -> str | None:
        if self.backend != "rpc":
            return None
        return f"{self.cfg.explorer_base.rstrip('/')}/tx/{tx_hex}"

    # -- reads -----------------------------------------------------------

    def lookup(self, evidence_hash: bytes) -> dict[str, Any] | None:
        self.ensure_deployed()
        try:
            rec = self.contract.functions.get(evidence_hash).call()
        except (ContractLogicError, Exception):  # noqa: BLE001 - NotAnchored revert
            return None
        submitter, anchored_at, similarity_bps, sref = rec
        if not anchored_at:
            return None
        sref_hex = sref.hex() if isinstance(sref, (bytes, bytearray)) else str(sref)
        return {
            "submitter": submitter,
            "anchored_at": int(anchored_at),
            "similarity_bps": int(similarity_bps),
            "subject_ref": "0x" + sref_hex.removeprefix("0x"),
        }

    def verify(
        self,
        evidence: Evidence,
        probe_embedding_sha256: str | None = None,
        recheck_source: bool = False,
    ) -> Verification:
        """Recompute the hash from local bytes and check it against chain state."""
        ehash = evidence.evidence_hash()
        v = Verification(evidence_hash="0x" + ehash.hex(), anchored=False)

        record = self.lookup(ehash)
        if record is None:
            v.notes.append(
                "No record for this evidence hash. Either it was never anchored, "
                "or the bundle has been modified since it was."
            )
            return v

        v.anchored = True
        v.on_chain = record
        v.similarity_matches = record["similarity_bps"] == evidence.similarity_bps()
        if not v.similarity_matches:
            v.notes.append(
                f"similarity mismatch: chain says {record['similarity_bps']} bps, "
                f"bundle says {evidence.similarity_bps()} bps"
            )

        expected = "0x" + subject_ref(evidence.probe.embedding_sha256, self.cfg.subject_salt).hex()
        v.subject_matches = expected == record["subject_ref"]
        if not v.subject_matches:
            v.notes.append(
                "subject commitment on chain does not match this bundle's probe "
                "(different face, or a different SIGIL_SUBJECT_SALT)"
            )

        if probe_embedding_sha256 is not None:
            v.probe_matches = probe_embedding_sha256 == evidence.probe.embedding_sha256
            if not v.probe_matches:
                v.notes.append(
                    "the supplied probe image does not re-encode to the embedding "
                    "recorded in this bundle"
                )

        if recheck_source:
            v.source_image_intact = self._recheck_source(evidence, v)

        return v

    def _recheck_source(self, evidence: Evidence, v: Verification) -> bool | None:
        """Re-download the matched post image and confirm it is byte-identical.

        The chain proves the bundle has not changed. This proves the *world* has
        not changed underneath it - if the post was edited, its image swapped or
        the account deleted, that shows up here rather than passing silently.
        """
        from ..evidence import sha256_hex
        from ..search.http import fetch_image, make_session

        blob = fetch_image(make_session(), evidence.match.image_url, 30.0)
        if blob is None:
            v.notes.append(
                "source image is no longer retrievable - the post may have been "
                "deleted or made private since it was anchored"
            )
            return False
        intact = sha256_hex(blob) == evidence.match.image_sha256
        if not intact:
            v.notes.append(
                "source image still exists but its bytes changed since anchoring"
            )
        return intact

    def total_anchored(self) -> int:
        self.ensure_deployed()
        return int(self.contract.functions.total().call())
