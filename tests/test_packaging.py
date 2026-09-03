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


# ------------------------------------------------------------ the lockfile


def _lock() -> dict[str, str]:
    """The pinned snapshot, as {name: version}."""
    path = ROOT / "requirements.lock"
    pins = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        pins[name.lower().replace("_", "-")] = version
    return pins


def test_the_lockfile_is_only_exact_pins():
    """A range in a lockfile is not a lock."""
    for line in (ROOT / "requirements.lock").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"not an exact pin: {line}"
        assert not any(c in line for c in "<>~!*"), f"not an exact pin: {line}"


def test_the_lockfile_carries_no_local_paths():
    """`uv pip freeze` emits an editable self-reference with an absolute path.

    Committing that would put whatever machine generated the file into a public
    repository, and it happened once while writing this.
    """
    # Only the pins. A path in a comment is an install instruction - the
    # header's own example contains `-e .` - and a path in a pin is the bug.
    pins = "\n".join(
        line for line in (ROOT / "requirements.lock").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    for marker in ("file://", "/home/", "/tmp/", "/Users/", "-e "):
        assert marker not in pins, f"local path leaked into the lockfile: {marker}"


def test_every_declared_runtime_dependency_is_pinned():
    """A dependency the lockfile does not mention is not actually locked."""
    lock = _lock()
    for spec in _pyproject()["project"]["dependencies"]:
        name = spec.split(">=")[0].split("[")[0].strip().lower().replace("_", "-")
        assert name in lock, f"{name} is a runtime dependency but is not pinned"


def test_the_pinned_versions_satisfy_the_declared_floors():
    """The snapshot has to be a legal resolution of what pyproject asks for.

    Otherwise the two files describe different projects and the lockfile
    reproduces something the package would refuse to install.
    """
    lock = _lock()
    for spec in _pyproject()["project"]["dependencies"]:
        if ">=" not in spec:
            continue
        name, _, floor = spec.partition(">=")
        name = name.split("[")[0].strip().lower().replace("_", "-")
        pinned = lock[name]

        def parts(v):
            out = []
            for chunk in v.split("."):
                digits = "".join(c for c in chunk if c.isdigit())
                out.append(int(digits) if digits else 0)
            return out

        assert parts(pinned) >= parts(floor.strip()), (
            f"{name} is pinned at {pinned}, below the declared floor {floor.strip()}"
        )
