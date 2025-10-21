"""Tests for Codex CLI commands."""

import pytest
from click.testing import CliRunner
from unittest import mock

from mlflow.claude_code.cli import commands as autolog_commands


@pytest.fixture
def runner():
    """Provide a CLI runner for tests."""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Mock the configuration module."""
    with mock.patch("mlflow.codex_cli.cli.setup_environment_config") as setup, \
         mock.patch("mlflow.codex_cli.cli.disable_tracing") as disable, \
         mock.patch("mlflow.codex_cli.cli.get_tracing_status") as get_status:
        yield {
            "setup": setup,
            "disable": disable,
            "get_status": get_status,
        }


def test_autolog_help_shows_codex_command(runner):
    """Test that the autolog group help shows the codex command."""
    result = runner.invoke(autolog_commands, ["--help"])
    assert result.exit_code == 0
    assert "autologging" in result.output.lower()
    assert "claude" in result.output
    assert "codex" in result.output


def test_codex_help_command(runner):
    """Test that the codex command shows help."""
    result = runner.invoke(autolog_commands, ["codex", "--help"])
    assert result.exit_code == 0
    assert "OpenAI Codex CLI" in result.output
    assert "--tracking-uri" in result.output
    assert "--experiment-id" in result.output
    assert "--sessions-dir" in result.output
    assert "--disable" in result.output
    assert "--status" in result.output
    assert "--trace-latest" in result.output


def test_codex_enable_minimal(runner, mock_config):
    """Test enabling Codex tracing with minimal options."""
    from mlflow.codex_cli.config import CodexTracingStatus

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=True,
        tracking_uri="file://~/.codex/mlflow",
    )

    result = runner.invoke(autolog_commands, ["codex"])

    assert result.exit_code == 0
    assert "✅ Codex CLI tracing enabled" in result.output
    mock_config["setup"].assert_called_once()


def test_codex_enable_with_all_options(runner, mock_config):
    """Test enabling with all configuration options."""
    from mlflow.codex_cli.config import CodexTracingStatus

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=True,
        tracking_uri="databricks",
        experiment_id="12345",
    )

    result = runner.invoke(autolog_commands, [
        "codex",
        "-u", "databricks",
        "-e", "12345",
        "-s", "/custom/sessions",
    ])

    assert result.exit_code == 0
    mock_config["setup"].assert_called_once_with(
        tracking_uri="databricks",
        experiment_id="12345",
        experiment_name=None,
        sessions_dir="/custom/sessions",
    )


def test_codex_status_flag(runner, mock_config):
    """Test the --status flag."""
    from mlflow.codex_cli.config import CodexTracingStatus

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=True,
        tracking_uri="file://./mlruns",
        experiment_name="test-experiment",
        sessions_dir="~/.codex/sessions",
    )

    result = runner.invoke(autolog_commands, ["codex", "--status"])

    assert result.exit_code == 0
    assert "Codex CLI Tracing Status" in result.output
    assert "✅ Status: ENABLED" in result.output
    mock_config["get_status"].assert_called()
    # Should not call setup when --status is used
    mock_config["setup"].assert_not_called()


def test_codex_status_disabled(runner, mock_config):
    """Test status when tracing is disabled."""
    from mlflow.codex_cli.config import CodexTracingStatus

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=False,
        reason="Not configured",
    )

    result = runner.invoke(autolog_commands, ["codex", "--status"])

    assert result.exit_code == 0
    assert "❌ Status: DISABLED" in result.output
    assert "mlflow autolog codex" in result.output


def test_codex_disable_flag_success(runner, mock_config):
    """Test the --disable flag when tracing is enabled."""
    mock_config["disable"].return_value = True

    result = runner.invoke(autolog_commands, ["codex", "--disable"])

    assert result.exit_code == 0
    assert "✅ Codex CLI tracing disabled" in result.output
    mock_config["disable"].assert_called_once()
    # Should not call setup when --disable is used
    mock_config["setup"].assert_not_called()


def test_codex_disable_flag_not_enabled(runner, mock_config):
    """Test the --disable flag when tracing is not enabled."""
    mock_config["disable"].return_value = False

    result = runner.invoke(autolog_commands, ["codex", "--disable"])

    assert result.exit_code == 0
    assert "❌ No Codex CLI configuration found" in result.output


def test_codex_trace_latest_flag_disabled(runner, mock_config):
    """Test --trace-latest flag when tracing is disabled."""
    from mlflow.codex_cli.config import CodexTracingStatus

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=False,
        reason="Not configured",
    )

    result = runner.invoke(autolog_commands, ["codex", "--trace-latest"])

    assert result.exit_code == 0
    assert "❌ Codex CLI tracing is not enabled" in result.output
    assert "mlflow autolog codex" in result.output


def test_codex_trace_latest_flag_success(runner, mock_config):
    """Test --trace-latest flag with successful trace creation."""
    from mlflow.codex_cli.config import CodexTracingStatus
    from mlflow.entities import TraceInfo

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=True,
        tracking_uri="file://./mlruns",
    )

    with mock.patch("mlflow.codex_cli.cli.process_latest_session") as process:
        # Create a minimal mock trace
        mock_trace = mock.MagicMock()
        mock_trace.info = TraceInfo(
            request_id="test-trace-123",
            experiment_id="0",
            timestamp_ms=1234567890,
            execution_time_ms=1000,
            status="OK",
            request_metadata={},
            tags={},
        )
        mock_trace.info.trace_metadata = {"mlflow.trace.session": "test-session"}
        process.return_value = mock_trace

        result = runner.invoke(autolog_commands, ["codex", "--trace-latest"])

        assert result.exit_code == 0
        assert "✅ Created trace" in result.output
        assert "test-trace-123" in result.output
        assert "test-session" in result.output
        process.assert_called_once()


def test_codex_trace_latest_flag_failure(runner, mock_config):
    """Test --trace-latest flag when trace creation fails."""
    from mlflow.codex_cli.config import CodexTracingStatus

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=True,
        tracking_uri="file://./mlruns",
    )

    with mock.patch("mlflow.codex_cli.cli.process_latest_session") as process:
        process.return_value = None

        result = runner.invoke(autolog_commands, ["codex", "--trace-latest"])

        assert result.exit_code == 0
        assert "❌ Failed to create trace" in result.output
        assert "codex_tracing.log" in result.output


def test_codex_combined_flags_prioritization(runner, mock_config):
    """Test that flags are prioritized correctly (status > disable > trace-latest > enable)."""
    from mlflow.codex_cli.config import CodexTracingStatus

    mock_config["get_status"].return_value = CodexTracingStatus(enabled=True)

    # Status flag takes precedence
    result = runner.invoke(autolog_commands, ["codex", "--status", "--disable"])
    assert "Codex CLI Tracing Status" in result.output
    mock_config["disable"].assert_not_called()
