"""
Codex autologging - Configuration for Codex CLI tracing.

This module provides the autolog() function to enable Codex CLI tracing.
After enabling, run the tracer manually with: mlflow codex trace --experiment-id <id>
"""

import logging

from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "codex"


def autolog(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    experiment_id: str | None = None,
    disable: bool = False,
    silent: bool = False,
    codex_home: str | None = None,
):
    """
    Enable (or disable) autologging from Codex CLI to MLflow.

    This configures the Codex autologging settings. After enabling, run the
    tracer manually with: mlflow codex trace --experiment-id <id>

    Args:
        tracking_uri: The URI to the MLflow tracking server.
        experiment_name: The name of the experiment to log traces to.
        experiment_id: The ID of the experiment to log traces to.
        disable: If True, disables Codex autologging.
        silent: If True, suppress all event logs and warnings.
        codex_home: Path to Codex home directory.

    Example:
        .. code-block:: python

            import mlflow

            # Enable Codex autologging
            mlflow.codex.autolog()

            # Then run the tracer manually:
            # mlflow codex trace --experiment-id 1
    """
    try:
        if disable:
            if not silent:
                _logger.info("Codex autologging disabled.")
        else:
            if not silent:
                _logger.info("✅ Codex autologging enabled!")
                _logger.info(f"   Experiment: {experiment_name or experiment_id or 'default'}")
                _logger.info("\n📝 Next steps:")
                _logger.info(
                    f"  1. Start the tracer: mlflow codex trace --experiment-id "
                    f"{experiment_id or '<id>'}"
                )
                _logger.info("  2. Run 'codex' in another terminal")
                _logger.info("  3. View traces: mlflow ui")

        _record_event(
            AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": not disable, "disable": disable}
        )
    except Exception as e:
        if not silent:
            _logger.error(f"Failed to configure Codex autologging: {e}")
        raise
