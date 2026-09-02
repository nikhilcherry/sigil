"""Command-line surface, exercised through click's runner."""

from click.testing import CliRunner

from sigil.chain import ChainClient
from sigil.cli import cli


def test_chain_address_reports_without_deploying(cfg, monkeypatch):
    """`chain info` deploys the registry before it can report, which fails on an
    unfunded key - so there has to be a way to find out which address to fund
    without first trying to spend from it."""
    monkeypatch.setenv("SIGIL_CHAIN", "local")
    result = CliRunner().invoke(cli, ["chain", "address"])

    assert result.exit_code == 0, result.output
    assert ChainClient(cfg).address in result.output.replace("\n", "")
    assert ChainClient(cfg).deployed_address() is None, "reporting deployed a contract"


def test_chain_info_does_deploy(cfg, monkeypatch):
    """The contrast that makes the split worth having."""
    monkeypatch.setenv("SIGIL_CHAIN", "local")
    result = CliRunner().invoke(cli, ["chain", "info"])

    assert result.exit_code == 0, result.output
    assert ChainClient(cfg).deployed_address() is not None


def test_backends_reports_the_provider_in_use():
    result = CliRunner().invoke(cli, ["backends"])

    assert result.exit_code == 0, result.output
    assert "runs on" in result.output
    assert "CPU" in result.output or "CUDA" in result.output
