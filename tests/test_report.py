"""Every panel the demo recording puts on screen, rendered at least once.

There are no resubmissions for this build, so a formatting crash partway
through a recorded run is the expensive kind of bug: cheap to prevent, and
invisible to a test suite that only ever exercises the happy path in the dark.
"""

import pytest
from rich.console import Console

from sigil import report
from sigil.chain import Verification
from sigil.search.base import Candidate
from sigil.search.matcher import MatchResult, ScoredCandidate


@pytest.fixture(autouse=True)
def captured(monkeypatch):
    """Render to a real console with a fixed width, into a buffer."""
    console = Console(width=100, record=True, force_terminal=False)
    monkeypatch.setattr(report, "console", console)
    return console


def _scored(similarity=0.76):
    return ScoredCandidate(
        candidate=Candidate(
            platform="bluesky", image_url="https://cdn/img.jpg",
            post_url="https://bsky.app/profile/who.bsky.social/post/1",
            post_uri="at://did:plc:who/app.bsky.feed.post/1",
            author_handle="who.bsky.social", author_did="did:plc:who",
            author_display_name="Who", text="a post",
            created_at="2026-01-01", discovered_via="app.bsky.feed.getAuthorFeed",
        ),
        similarity=similarity, image_sha256="ab" * 32,
        faces_in_image=1, matched_bbox=[1, 2, 3, 4],
    )


def test_probe_panel_renders(captured, probe_ref):
    report.probe_panel(probe_ref, "examples/probe.jpg")
    out = captured.export_text()
    assert "insightface" in out
    assert probe_ref.embedding_sha256[:16] in out.replace("…", "")


def test_search_panel_renders_with_candidates(captured):
    result = MatchResult(best=_scored(), ranked=[_scored(), _scored(0.2)],
                         images_examined=57, images_with_faces=9, faces_examined=11)
    report.search_panel(result, 0.38, ["bluesky"])
    out = captured.export_text()
    assert "57" in out
    assert "0.380" in out


def test_search_panel_reports_reused_scores_only_when_there_were_some(captured):
    plain = MatchResult(best=None, images_examined=3)
    report.search_panel(plain, 0.38, ["bluesky"])
    assert "duplicate images" not in captured.export_text()

    captured.export_text(clear=True)
    deduped = MatchResult(best=None, images_examined=3, inference_reused=2)
    report.search_panel(deduped, 0.38, ["bluesky"])
    assert "duplicate images" in captured.export_text()


def test_search_panel_renders_with_no_candidates_at_all(captured):
    """The empty case is the one a live demo is most likely to hit."""
    report.search_panel(MatchResult(best=None), 0.38, ["bluesky"])
    assert "Search completed" in captured.export_text()


def test_no_match_panel_renders_with_and_without_ranked(captured):
    report.no_match_panel(MatchResult(best=None, images_examined=12), 0.38)
    assert "No candidate cleared" in captured.export_text()

    captured.export_text(clear=True)
    report.no_match_panel(
        MatchResult(best=None, ranked=[_scored(0.31)], images_examined=12), 0.38
    )
    assert "0.3100" in captured.export_text()


def test_match_panel_renders(captured, evidence):
    report.match_panel(evidence)
    out = captured.export_text()
    assert "bluesky" in out
    assert "0.7412" in out


def test_anchor_panel_renders_a_fresh_anchor(captured):
    report.anchor_panel({
        "already_anchored": False, "tx_hash": "0x" + "a" * 64, "block_number": 9,
        "gas_used": 97122, "chain_id": 80002, "contract": "0x" + "b" * 40,
        "evidence_hash": "0x" + "c" * 64, "subject_ref": "0x" + "d" * 64,
        "explorer": "https://amoy.polygonscan.com/tx/0x" + "a" * 64,
    })
    out = captured.export_text()
    assert "97,122" in out
    assert "80002" in out


def test_anchor_panel_renders_a_duplicate(captured):
    """The append-only path prints a different shape and has its own way to break."""
    report.anchor_panel({
        "already_anchored": True, "anchored_at": 1788354667,
        "contract": "0x" + "b" * 40, "evidence_hash": "0x" + "c" * 64,
        "subject_ref": "0x" + "d" * 64,
    })
    assert "already anchored" in captured.export_text()


def test_verification_panel_renders_pass_and_fail(captured):
    ok = Verification(
        evidence_hash="0x" + "c" * 64, anchored=True, similarity_matches=True,
        subject_matches=True, probe_matches=True, source_image_intact=True,
        on_chain={"submitter": "0x" + "e" * 40, "anchored_at": 1788354667}, notes=[],
    )
    report.verification_panel(ok)
    assert "VERIFIED" in captured.export_text()

    captured.export_text(clear=True)
    bad = Verification(
        evidence_hash="0x" + "c" * 64, anchored=True, similarity_matches=False,
        subject_matches=True, probe_matches=None, source_image_intact=None,
        on_chain=None, notes=["similarity on chain does not match the bundle"],
    )
    report.verification_panel(bad)
    out = captured.export_text()
    assert "NOT VERIFIED" in out
    assert "does not match" in out


def test_identity_table_marks_what_cleared_the_bar(captured):
    report.identity_table({
        "index_size": 3820, "threshold": 0.45,
        "hits": [
            {"name": "A Known Person", "similarity": 0.61,
             "source": "en.wikipedia", "accepted": True},
            {"name": "A Lookalike", "similarity": 0.31,
             "source": "fr.wikipedia", "accepted": False},
        ],
    }, echo=True)
    out = captured.export_text()
    assert "3820" in out
    assert "A Known Person" in out
    assert "A Lookalike" in out


def test_identity_table_returns_the_table_when_it_is_not_echoing(captured):
    """The CLI prints the returned table itself, through the progress console,
    so the non-echo branch has to hand back something printable."""
    table = report.identity_table({"index_size": 1, "hits": [
        {"name": "Someone", "similarity": 0.5, "source": "en.wikipedia",
         "accepted": True},
    ]})

    assert table is not None
    assert captured.export_text() == "", "it printed despite echo=False"
    Console(width=80).print(table)


def test_identity_table_renders_with_no_hits(captured):
    """"I don't know" is a legitimate outcome and still has to print."""
    report.identity_table({"index_size": 0, "hits": []}, echo=True)
    assert "Identity index" in captured.export_text()


def test_stage_headers_render(captured):
    report.stage(1, 5, "Face scan")
    assert "Face scan" in captured.export_text()


def test_the_anchored_candidate_is_shown_even_when_it_is_outside_the_top_few():
    """It usually is, since a social identity claim outranks a higher cosine."""
    from sigil.report import console, search_panel
    from sigil.search.base import Candidate
    from sigil.search.matcher import MatchResult, ScoredCandidate

    def scored(sim, handle, kind="web", photo=0.99):
        return ScoredCandidate(
            candidate=Candidate(
                platform="p", image_url=f"https://{handle}",
                post_url="https://p", post_uri="at://p", author_handle=handle,
                author_did="", author_display_name=handle, text="",
                created_at="", discovered_via="v", source_kind=kind),
            similarity=sim, image_sha256=handle, faces_in_image=1,
            matched_bbox=[0, 0, 1, 1], photo_similarity=photo)

    ranked = [scored(0.99 - i / 1000, f"copy{i}") for i in range(12)]
    winner = scored(0.7596, "aoc", kind="social", photo=0.02)
    ranked.append(winner)
    result = MatchResult(best=winner, ranked=ranked)

    with console.capture() as cap:
        search_panel(result, 0.38, ["bluesky", "google-vision-web"])
    out = cap.get()
    assert "aoc" in out, "the anchored candidate was not shown at all"
    assert "◀" in out


def _scored_for_report(sim, faces, bbox, photo=0.02, kind="social"):
    from sigil.search.base import Candidate
    from sigil.search.matcher import ScoredCandidate

    return ScoredCandidate(
        candidate=Candidate(
            platform="bluesky", image_url="https://i", post_url="https://p",
            post_uri="at://p", author_handle="who", author_did="",
            author_display_name="Who", text="", created_at="",
            discovered_via="v", source_kind=kind),
        similarity=sim, image_sha256="ab" * 32, faces_in_image=faces,
        matched_bbox=bbox, photo_similarity=photo, rank=1)


def test_the_match_panel_says_when_the_matched_image_was_a_group_photo(evidence):
    """A match in a twelve-person photo means less than the same score alone.

    The calibration section names exactly that as a source of error, so the
    match panel should not let a reader assume a headshot.
    """
    from sigil.report import console, match_panel

    with console.capture() as cap:
        match_panel(evidence, _scored_for_report(0.76, 5, [10, 20, 60, 80]))
    out = cap.get()
    assert "faces in that image" in out
    assert "group photo" in out
    assert "[10, 20, 60, 80]" in out


def test_the_match_panel_does_not_cry_group_photo_over_a_portrait(evidence):
    from sigil.report import console, match_panel

    with console.capture() as cap:
        match_panel(evidence, _scored_for_report(0.76, 1, [10, 20, 60, 80]))
    out = cap.get()
    assert "faces in that image" in out
    assert "group photo" not in out


def test_the_match_panel_still_works_with_no_scored_candidate(evidence):
    """`sigil verify` renders a bundle it did not search for."""
    from sigil.report import console, match_panel

    with console.capture() as cap:
        match_panel(evidence)
    out = cap.get()
    assert "Match found" in out
    assert "faces in that image" not in out


def test_the_candidate_table_marks_multi_face_images_without_a_column():
    """A column of "1" would wrap the table and say nothing."""
    from sigil.report import console, search_panel
    from sigil.search.matcher import MatchResult

    one = _scored_for_report(0.90, 1, [0, 0, 1, 1])
    many = _scored_for_report(0.80, 4, [0, 0, 1, 1])
    many.rank = 2
    result = MatchResult(best=one, ranked=[one, many])

    with console.capture() as cap:
        search_panel(result, 0.38, ["bluesky"])
    out = cap.get()
    assert "·4" in out, "a multi-face candidate is not marked"
    assert "·1" not in out, "single-face candidates should not be marked"


# ------------------------------------- markup written by somebody else

MARKUP_PAYLOAD = "[bold green]VERIFIED[/bold green] Nice Person"
LINK_PAYLOAD = "[link=file:///etc/passwd]click[/link]"


def test_a_display_name_cannot_style_itself_in_the_match_panel(evidence):
    """The terminal output is what a screen recording shows as the evidence.

    rich reads `[...]` as styling, so an account whose display name is
    `[bold green]VERIFIED[/bold green]` printed itself in bold green - the same
    styling this report uses for a passing check. An account able to style its
    own row can forge the appearance of a verdict.
    """
    from sigil.report import console, match_panel

    evidence.match.author_display_name = MARKUP_PAYLOAD
    evidence.match.text = LINK_PAYLOAD

    with console.capture() as cap:
        match_panel(evidence)
    out = cap.get()

    assert "[bold green]" in out, "the markup was interpreted, not shown"
    assert "[/bold green]" in out
    assert "[link=file:///etc/passwd]" in out


def test_a_handle_cannot_style_itself_in_the_candidate_table():
    from sigil.report import console, search_panel
    from sigil.search.base import Candidate
    from sigil.search.matcher import MatchResult, ScoredCandidate

    hostile = ScoredCandidate(
        candidate=Candidate(
            platform="bluesky", image_url="https://i", post_url="https://p",
            post_uri="at://p", author_handle=MARKUP_PAYLOAD, author_did="",
            author_display_name="", text="", created_at="",
            discovered_via="app.bsky.actor.searchActors:avatar",
            source_kind="social"),
        similarity=0.9, image_sha256="ab" * 32, faces_in_image=1,
        matched_bbox=[0, 0, 1, 1], photo_similarity=0.02, rank=1)

    with console.capture() as cap:
        search_panel(MatchResult(best=hostile, ranked=[hostile]), 0.38, ["bluesky"])
    assert "[bold green]" in cap.get()


def test_a_wikidata_label_cannot_style_itself_in_the_identity_table():
    """Index names come from Wikidata, which is also somebody else's text."""
    from sigil.report import console, identity_table

    event = {"index_size": 1, "threshold": 0.45, "hits": [
        {"name": "ok", "similarity": 0.9, "source": MARKUP_PAYLOAD,
         "accepted": True}]}
    with console.capture() as cap:
        identity_table(event, echo=True)
    assert "[bold green]" in cap.get()


def test_a_verification_note_cannot_style_itself(evidence):
    """Notes quote provider-supplied values back to the reader."""
    from sigil.chain.client import Verification
    from sigil.report import console, verification_panel

    v = Verification(evidence_hash="0x00", anchored=True)
    v.notes.append(MARKUP_PAYLOAD)
    with console.capture() as cap:
        verification_panel(v)
    assert "[bold green]" in cap.get()


def test_the_report_still_styles_its_own_output(evidence):
    """The escaping must not have flattened the report's own markup."""
    from sigil.report import console, match_panel

    with console.capture() as cap:
        match_panel(evidence)
    out = cap.get()
    # Its own labels render as text, not as literal tags.
    assert "[bold green]" not in out
    assert "identity" in out and "Match found" in out
