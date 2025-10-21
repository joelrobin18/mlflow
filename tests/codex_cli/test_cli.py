import pytest
from click.testing import CliRunner
from unittest import mock

from mlflow.codex_cli.cli import commands


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


def test_codex_cli_help_command(runner):
    """Test that the main codex-cli command shows help."""
    result = runner.invoke(commands, ["--help"])
    assert result.exit_code == 0
    assert "OpenAI Codex CLI autologging" in result.output
    assert "enable" in result.output
    assert "disable" in result.output
    assert "status" in result.output
    assert "trace-latest" in result.output


def test_enable_command_help(runner):
    """Test that the enable command shows help."""
    result = runner.invoke(commands, ["enable", "--help"])
    assert result.exit_code == 0
    assert "Enable automatic tracing" in result.output
    assert "--tracking-uri" in result.output
    assert "--experiment-id" in result.output
    assert "--sessions-dir" in result.output


def test_enable_command_minimal(runner, mock_config):
    """Test enable command with minimal options."""
    result = runner.invoke(commands, ["enable"])

    assert result.exit_code == 0
    assert "✅ Codex CLI tracing enabled" in result.output
    mock_config["setup"].assert_called_once()


def test_enable_command_full_options(runner, mock_config):
    """Test enable command with all options."""
    result = runner.invoke(commands, [
        "enable",
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


def test_disable_command_success(runner, mock_config):
    """Test disable command when tracing is enabled."""
    mock_config["disable"].return_value = True

    result = runner.invoke(commands, ["disable"])

    assert result.exit_code == 0
    assert "✅ Codex CLI tracing disabled" in result.output
    mock_config["disable"].assert_called_once()


def test_disable_command_not_enabled(runner, mock_config):
    """Test disable command when tracing is not enabled."""
    mock_config["disable"].return_value = False

    result = runner.invoke(commands, ["disable"])

    assert result.exit_code == 0
    assert "❌ No Codex CLI configuration found" in result.output


def test_status_command(runner, mock_config):
    """Test status command."""
    from mlflow.codex_cli.config import CodexTracingStatus

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=True,
        tracking_uri="file://./mlruns",
        experiment_name="test-experiment",
    )

    result = runner.invoke(commands, ["status"])

    assert result.exit_code == 0
    assert "Codex CLI Tracing Status" in result.output
    mock_config["get_status"].assert_called()


def test_status_command_disabled(runner, mock_config):
    """Test status command when disabled."""
    from mlflow.codex_cli.config import CodexTracingStatus

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=False,
        reason="Not configured",
    )

    result = runner.invoke(commands, ["status"])

    assert result.exit_code == 0
    assert "DISABLED" in result.output


def test_trace_latest_command_disabled(runner, mock_config):
    """Test trace-latest command when tracing is disabled."""
    from mlflow.codex_cli.config import CodexTracingStatus

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=False,
        reason="Not configured",
    )

    result = runner.invoke(commands, ["trace-latest"])

    assert result.exit_code == 0
    assert "❌ Codex CLI tracing is not enabled" in result.output


def test_trace_latest_command_success(runner, mock_config):
    """Test trace-latest command with successful trace creation."""
    from mlflow.codex_cli.config import CodexTracingStatus
    from mlflow.entities import TraceInfo

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=True,
        tracking_uri="file://./mlruns",
    )

    # Mock process_latest_session
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

        result = runner.invoke(commands, ["trace-latest"])

        assert result.exit_code == 0
        assert "✅ Created trace" in result.output
        assert "test-trace-123" in result.output
        process.assert_called_once()


def test_trace_latest_command_failure(runner, mock_config):
    """Test trace-latest command when trace creation fails."""
    from mlflow.codex_cli.config import CodexTracingStatus

    mock_config["get_status"].return_value = CodexTracingStatus(
        enabled=True,
        tracking_uri="file://./mlruns",
    )

    with mock.patch("mlflow.codex_cli.cli.process_latest_session") as process:
        process.return_value = None

        result = runner.invoke(commands, ["trace-latest"])

        assert result.exit_code == 0
        assert "❌ Failed to create trace" in result.output
