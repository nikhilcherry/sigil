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

    /// @notice Emitted when someone puts a check of a record on the record.
    /// @dev Deliberately says nothing about whether the *bundle* verified.
    ///      This contract cannot see a bundle - it holds a hash - so the only
    ///      honest claim it can log is "at this time, this address asked about
    ///      this hash, and the registry did (or did not) hold it". The six
    ///      checks in `sigil verify` are computed off-chain against the file.
    event VerificationLogged(
        bytes32 indexed evidenceHash,
        address indexed checkedBy,
        bool    anchored,
        uint64  checkedAt
    );

    /// @notice Record that this hash was checked, and return what was found.
    /// @dev `isAnchored` answers the same question for free, and that is the
    ///      one a normal verify calls - which is why this is separate rather
    ///      than folded into it. This one costs gas and exists for the case
    ///      where the *checking* is the thing worth being able to prove later:
    ///      an auditor who wants a timestamped, unforgeable trace that they
    ///      looked and what they were told. A view function leaves no such
    ///      trace, and a log kept off-chain would be one its own author could
    ///      edit afterwards.
    ///
    ///      Never reverts on a miss. "That hash is not here" is a real and
    ///      useful thing to have anchored - it is how a reader later
    ///      establishes that a bundle did *not* exist at a given time - so it
    ///      is returned and logged rather than thrown away as an error.
    function logVerification(bytes32 evidenceHash)
        external
        returns (bool anchored, uint64 checkedAt)
    {
        anchored = _records[evidenceHash].anchoredAt != 0;
        checkedAt = uint64(block.timestamp);
        emit VerificationLogged(evidenceHash, msg.sender, anchored, checkedAt);
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
