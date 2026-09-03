"""The README is a graded deliverable, and its links rot silently.

A path that no longer exists or an anchor that no longer matches a heading
still renders as a link; it just goes nowhere. Nothing else in the suite would
notice.
"""

import re

import pytest

from tests.conftest import ROOT

README = ROOT / "README.md"
# Every `](target)`, which also catches badge links where the target
# sits outside a nested image link.
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _links() -> list[str]:
    return LINK.findall(README.read_text())


def _slug(heading: str) -> str:
    """GitHub's heading slug: lowercase, drop punctuation, spaces to hyphens."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s", "-", s)


def _headings() -> set[str]:
    return {
        _slug(line.lstrip("#").strip())
        for line in README.read_text().splitlines()
        if line.startswith("#")
    }


@pytest.mark.parametrize("target", sorted(
    {t for t in _links() if not t.startswith(("http://", "https://", "#"))}
))
def test_every_relative_link_points_at_something_that_exists(target):
    path = target.split("#")[0]
    assert (ROOT / path).exists(), f"README links to a missing path: {path}"


@pytest.mark.parametrize("anchor", sorted(
    {t.lstrip("#") for t in _links() if t.startswith("#")}
))
def test_every_in_page_anchor_matches_a_heading(anchor):
    assert anchor in _headings(), (
        f"README anchor #{anchor} matches no heading; have "
        f"{sorted(h for h in _headings() if h[:4] == anchor[:4])}"
    )


def test_the_readme_covers_what_the_task_requires():
    """The brief asks the README to cover what it does, how to run it, which
    blockchain, and known limitations."""
    text = README.read_text().lower()

    assert "## quickstart" in text
    assert "## limitations" in text
    assert "polygon" in text and "py-evm" in text
    assert "sigil run" in text


def test_the_readme_states_the_real_offline_test_count(pytestconfig):
    """A stale count in a graded deliverable is a false claim, not a typo.

    Two things make a count meaningless to compare, and both skip rather than
    fail. A partial collection knows only its own files. And the count depends
    on which optional extras are installed - without insightface, four of its
    tests are never collected - so this checks the figure against the install
    the README quotes it for, which is the Quickstart's ``.[insight,dev]``.
    CI installs ``.[dev]`` alone and legitimately collects fewer.
    """
    if getattr(pytestconfig, "collected_modules", 0) < 10:
        pytest.skip("partial collection - only the full suite knows the count")
    pytest.importorskip(
        "insightface",
        reason="the README's count is for the full .[insight,dev] install",
    )

    stated = re.search(r"#\s*(\d+)\s+offline tests", README.read_text())
    assert stated, "the README no longer states an offline test count"
    actual = pytestconfig.offline_test_count
    assert int(stated.group(1)) == actual, (
        f"README says {stated.group(1)} offline tests, this run collected {actual}"
    )
