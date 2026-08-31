// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title SigilRegistry
/// @notice Append-only registry of face-match evidence hashes.
/// @dev The registry deliberately stores no image, no text and no biometric.
///      It stores a keccak256 over a canonical evidence bundle, plus a salted
///      commitment to the subject. That keeps an irreversible public ledger
///      free of personal data while still making the claim verifiable: anyone
///      holding the bundle can recompute the hash and check it against chain
///      state, and nobody - including the submitter - can alter a record after
///      the fact.
contract SigilRegistry {
    struct Record {
        address submitter;      // who anchored it
        uint64  anchoredAt;     // block timestamp; 0 means "no record"
        uint32  similarityBps;  // cosine similarity in basis points
        bytes32 subjectRef;     // salted commitment to the probe subject
    }

    mapping(bytes32 => Record) private _records;
    bytes32[] private _anchored;

    event Anchored(
        bytes32 indexed evidenceHash,
        address indexed submitter,
        bytes32 indexed subjectRef,
        uint64  anchoredAt,
        uint32  similarityBps
    );

    error EmptyHash();
    error AlreadyAnchored(bytes32 evidenceHash);
    error NotAnchored(bytes32 evidenceHash);

    /// @notice Record an evidence hash. Reverts if that exact hash already exists.
    /// @dev Rejecting duplicates is what makes the record tamper-evident rather
    ///      than merely tamper-resistant: a second anchor cannot quietly
    ///      overwrite the first, so the earliest timestamp for a bundle stands.
    function anchor(
        bytes32 evidenceHash,
        uint32 similarityBps,
        bytes32 subjectRef
    ) external returns (uint64 anchoredAt) {
        if (evidenceHash == bytes32(0)) revert EmptyHash();
        if (_records[evidenceHash].anchoredAt != 0) revert AlreadyAnchored(evidenceHash);

        anchoredAt = uint64(block.timestamp);
        _records[evidenceHash] = Record({
            submitter: msg.sender,
            anchoredAt: anchoredAt,
            similarityBps: similarityBps,
            subjectRef: subjectRef
        });
        _anchored.push(evidenceHash);

        emit Anchored(evidenceHash, msg.sender, subjectRef, anchoredAt, similarityBps);
    }

    /// @notice Fetch a record. Reverts when the hash was never anchored.
    function get(bytes32 evidenceHash) external view returns (Record memory) {
        Record memory r = _records[evidenceHash];
        if (r.anchoredAt == 0) revert NotAnchored(evidenceHash);
        return r;
    }

    /// @notice Non-reverting existence check, for callers that expect misses.
    function isAnchored(bytes32 evidenceHash) external view returns (bool) {
        return _records[evidenceHash].anchoredAt != 0;
    }

    /// @notice Total number of distinct evidence hashes anchored.
    function total() external view returns (uint256) {
        return _anchored.length;
    }

    /// @notice Evidence hash at an index, for enumerating the registry.
    function hashAt(uint256 index) external view returns (bytes32) {
        return _anchored[index];
    }
}
