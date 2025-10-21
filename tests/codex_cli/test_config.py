import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from mlflow.codex_cli.config import (
    DEFAULT_CODEX_DIR,
    CodexTracingStatus,
    disable_tracing,
    get_config_path,
    get_sessions_dir,
    get_tracing_status,
    is_tracing_enabled,
    load_config,
    save_config,
    setup_environment_config,
)


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / ".codex"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def mock_codex_dir(temp_config_dir):
    """Mock the default Codex directory."""
    with mock.patch("mlflow.codex_cli.config.DEFAULT_CODEX_DIR", temp_config_dir):
        with mock.patch("mlflow.codex_cli.config.DEFAULT_CONFIG_FILE", temp_config_dir / "mlflow_config.json"):
            yield temp_config_dir


def test_get_config_path(mock_codex_dir):
    """Test getting config path."""
    config_path = get_config_path()
    assert config_path == mock_codex_dir / "mlflow_config.json"


def test_load_config_empty(mock_codex_dir):
    """Test loading config when file doesn't exist."""
    config = load_config()
    assert config == {}


def test_save_and_load_config(mock_codex_dir):
    """Test saving and loading config."""
    test_config = {
        "environment": {
            "MLFLOW_CODEX_TRACING_ENABLED": "true",
            "MLFLOW_TRACKING_URI": "file://./mlruns",
        }
    }

    save_config(test_config)
    loaded_config = load_config()

    assert loaded_config == test_config


def test_setup_environment_config_minimal(mock_codex_dir):
    """Test setup with minimal configuration."""
    setup_environment_config()

    config = load_config()

    assert "environment" in config
    assert config["environment"]["MLFLOW_CODEX_TRACING_ENABLED"] == "true"
    assert "MLFLOW_TRACKING_URI" in config["environment"]


def test_setup_environment_config_full(mock_codex_dir):
    """Test setup with full configuration."""
    setup_environment_config(
        tracking_uri="databricks",
        experiment_id="12345",
        sessions_dir="/custom/sessions",
    )

    config = load_config()

    assert config["environment"]["MLFLOW_TRACKING_URI"] == "databricks"
    assert config["environment"]["MLFLOW_EXPERIMENT_ID"] == "12345"
    assert config["sessions_dir"] == "/custom/sessions"


def test_is_tracing_enabled_default(mock_codex_dir):
    """Test tracing enabled check when not configured."""
    assert not is_tracing_enabled()


def test_is_tracing_enabled_true(mock_codex_dir):
    """Test tracing enabled check when enabled."""
    setup_environment_config()
    assert is_tracing_enabled()


def test_get_tracing_status_disabled(mock_codex_dir):
    """Test getting status when disabled."""
    status = get_tracing_status()

    assert isinstance(status, CodexTracingStatus)
    assert not status.enabled
    assert status.reason is not None


def test_get_tracing_status_enabled(mock_codex_dir):
    """Test getting status when enabled."""
    setup_environment_config(
        tracking_uri="file://./mlruns",
        experiment_name="test-experiment",
    )

    status = get_tracing_status()

    assert status.enabled
    assert status.tracking_uri == "file://./mlruns"
    assert status.experiment_name == "test-experiment"


def test_disable_tracing_when_not_enabled(mock_codex_dir):
    """Test disabling when not enabled."""
    result = disable_tracing()
    assert not result


def test_disable_tracing_when_enabled(mock_codex_dir):
    """Test disabling when enabled."""
    setup_environment_config()
    assert get_config_path().exists()

    result = disable_tracing()
    assert result
    assert not get_config_path().exists()


def test_get_sessions_dir_default(mock_codex_dir):
    """Test getting default sessions directory."""
    sessions_dir = get_sessions_dir()
    assert sessions_dir == mock_codex_dir / "sessions"


def test_get_sessions_dir_custom(mock_codex_dir):
    """Test getting custom sessions directory."""
    custom_dir = "/custom/sessions"
    setup_environment_config(sessions_dir=custom_dir)

    sessions_dir = get_sessions_dir()
    assert str(sessions_dir) == custom_dir
