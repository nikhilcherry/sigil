"""Packaging metadata that is duplicated, and therefore drifts.

requirements.txt and pyproject.toml list the same runtime dependencies in two
places. CI installs from pyproject; a reader following the README may well
install from requirements.txt. When those disagree, the environment that was
tested is not the environment that was built.
"""

import pytest

from tests.conftest import ROOT

# tomllib is 3.11+, and this package supports 3.10. A test helper must not be
# what raises the floor, so it skips there rather than failing to import.
tomllib = pytest.importorskip("tomllib")

REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"


def _requirements() -> list[str]:
    return [
        line.strip()
        for line in REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def test_requirements_and_pyproject_list_the_same_runtime_dependencies():
    assert _requirements() == _pyproject()["project"]["dependencies"]


def test_the_console_script_points_at_something_importable():
    """A broken entry point only shows up after `pip install`, which no test
    does - so check the target resolves."""
    import importlib

    script = _pyproject()["project"]["scripts"]["sigil"]
    module, _, attr = script.partition(":")

    assert callable(getattr(importlib.import_module(module), attr))


def test_the_insight_extra_names_the_heavy_backend():
    """The README's quickstart installs `.[insight,dev]`; CI only installs
    `.[dev]`, so nothing else checks this extra exists at all."""
    extras = _pyproject()["project"]["optional-dependencies"]

    assert "insight" in extras
    assert any(d.startswith("insightface") for d in extras["insight"])
    assert "dev" in extras
    assert any(d.startswith("pytest") for d in extras["dev"])


def test_the_declared_version_matches_the_package():
    import sigil

    assert _pyproject()["project"]["version"] == sigil.__version__
