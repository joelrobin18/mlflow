import pytest
from click.testing import CliRunner

from mlflow.claude_code.cli import commands


@pytest.fixture
def runner():
    """Provide a CLI runner for tests."""
    return CliRunner()


def test_claude_help_command(runner):
    """Test that the main claude command shows help."""
    result = runner.invoke(commands, ["--help"])
    assert result.exit_code == 0
    assert "Commands for autologging with MLflow" in result.output
    assert "claude" in result.output
    assert "codex" in result.output


def test_trace_command_help(runner):
    """Test that the claude subcommand shows help."""
    result = runner.invoke(commands, ["claude", "--help"])
    assert result.exit_code == 0
    assert "Set up Claude Code tracing" in result.output
    assert "--tracking-uri" in result.output
    assert "--experiment-id" in result.output
    assert "--disable" in result.output
    assert "--status" in result.output


def test_trace_status_with_no_config(runner):
    """Test trace status when no config exists."""
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude", "--status"])
        assert result.exit_code == 0
        assert "❌ Claude tracing is not enabled" in result.output


def test_trace_disable_with_no_config(runner):
    """Test trace disable when no config exists."""
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["claude", "--disable"])
        assert result.exit_code == 0
        # Should handle gracefully even if no config exists


def test_codex_help_command(runner):
    """Test that the codex command shows help (alias for claude)."""
    result = runner.invoke(commands, ["codex", "--help"])
    assert result.exit_code == 0
    assert "Set up Codex" in result.output or "Set up Claude Code" in result.output
    assert "--tracking-uri" in result.output
    assert "--experiment-id" in result.output
    assert "--disable" in result.output
    assert "--status" in result.output


def test_codex_status_with_no_config(runner):
    """Test codex status when no config exists (alias for claude)."""
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["codex", "--status"])
        assert result.exit_code == 0
        assert "❌ Claude tracing is not enabled" in result.output


def test_codex_disable_with_no_config(runner):
    """Test codex disable when no config exists (alias for claude)."""
    with runner.isolated_filesystem():
        result = runner.invoke(commands, ["codex", "--disable"])
        assert result.exit_code == 0
        # Should handle gracefully even if no config exists
