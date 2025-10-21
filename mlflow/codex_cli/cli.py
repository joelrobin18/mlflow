"""MLflow CLI commands for Codex CLI integration."""

from pathlib import Path

import click

from mlflow.codex_cli.config import (
    disable_tracing,
    get_tracing_status,
    setup_environment_config,
)
from mlflow.codex_cli.tracing import process_latest_session


# This command will be registered under the autolog group in mlflow/claude_code/cli.py
@click.command("codex")
@click.option(
    "--tracking-uri",
    "-u",
    help="MLflow tracking URI (e.g., 'databricks' or 'file://mlruns')",
)
@click.option("--experiment-id", "-e", help="MLflow experiment ID")
@click.option("--experiment-name", "-n", help="MLflow experiment name")
@click.option(
    "--sessions-dir",
    "-s",
    help="Custom Codex sessions directory (default: ~/.codex/sessions)",
)
@click.option("--disable", is_flag=True, help="Disable Codex CLI tracing")
@click.option("--status", is_flag=True, help="Show current tracing status")
@click.option(
    "--trace-latest",
    is_flag=True,
    help="Process the latest Codex session and create a trace",
)
def codex_command(
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    sessions_dir: str | None,
    disable: bool,
    status: bool,
    trace_latest: bool,
) -> None:
    """Set up automatic tracing for OpenAI Codex CLI sessions.

    This command configures MLflow to automatically trace Codex CLI sessions.
    After setup, use the 'codex' command normally and process sessions with
    --trace-latest to create traces from session files in ~/.codex/sessions.

    Examples:

      # Enable tracing with local storage
      mlflow autolog codex

      # Enable with custom tracking URI
      mlflow autolog codex -u file://./custom-mlruns

      # Enable with Databricks
      mlflow autolog codex -u databricks -e 123456789

      # Check status
      mlflow autolog codex --status

      # Process latest session
      mlflow autolog codex --trace-latest

      # Disable tracing
      mlflow autolog codex --disable
    """
    # Handle status flag
    if status:
        _show_status()
        return

    # Handle disable flag
    if disable:
        _handle_disable()
        return

    # Handle trace-latest flag
    if trace_latest:
        _handle_trace_latest()
        return

    # Default: Enable tracing
    click.echo("Configuring Codex CLI tracing...")

    setup_environment_config(
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        sessions_dir=sessions_dir,
    )

    click.echo("✅ Codex CLI tracing enabled")
    _show_status()


def _handle_disable() -> None:
    """Handle the disable command."""
    if disable_tracing():
        click.echo("✅ Codex CLI tracing disabled")
    else:
        click.echo("❌ No Codex CLI configuration found - tracing was not enabled")


def _handle_trace_latest() -> None:
    """Handle the trace-latest command."""
    status_info = get_tracing_status()

    if not status_info.enabled:
        click.echo("❌ Codex CLI tracing is not enabled")
        click.echo(f"   Reason: {status_info.reason}")
        click.echo("\n💡 Enable tracing with: mlflow autolog codex")
        return

    click.echo("Processing latest Codex session...")

    trace = process_latest_session()

    if trace:
        click.echo(f"✅ Created trace: {trace.info.request_id}")
        click.echo(f"   Session: {trace.info.trace_metadata.get('mlflow.trace.session', 'N/A')}")

        if status_info.tracking_uri:
            click.echo(f"   Tracking URI: {status_info.tracking_uri}")

        click.echo("\n💡 View trace in MLflow UI:")
        # Show correct UI command based on tracking URI
        if status_info.tracking_uri:
            # Check if using default ./mlruns directory
            if status_info.tracking_uri.endswith("/mlruns"):
                # For default mlruns, just show 'mlflow ui'
                click.echo("   mlflow ui")
            elif status_info.tracking_uri == "databricks":
                click.echo("   View in your Databricks workspace")
            else:
                # For custom URIs, show full command
                click.echo(f"   mlflow ui --backend-store-uri {status_info.tracking_uri}")
        else:
            click.echo("   mlflow ui")
    else:
        click.echo("❌ Failed to create trace")
        click.echo("   Check ~/.codex/mlflow/codex_tracing.log for details")


def _show_status() -> None:
    """Show current tracing status."""
    click.echo("\n" + "=" * 50)
    click.echo("📊 Codex CLI Tracing Status")
    click.echo("=" * 50)

    status_info = get_tracing_status()

    if not status_info.enabled:
        click.echo("❌ Status: DISABLED")
        if status_info.reason:
            click.echo(f"   Reason: {status_info.reason}")
        click.echo("\n💡 Enable tracing with: mlflow autolog codex")
        return

    click.echo("✅ Status: ENABLED")

    if status_info.tracking_uri:
        click.echo(f"📊 Tracking URI: {status_info.tracking_uri}")

    if status_info.experiment_id:
        click.echo(f"🔬 Experiment ID: {status_info.experiment_id}")
    elif status_info.experiment_name:
        click.echo(f"🔬 Experiment Name: {status_info.experiment_name}")
    else:
        click.echo("🔬 Experiment: Default (experiment 0)")

    if status_info.sessions_dir:
        sessions_path = Path(status_info.sessions_dir)
        click.echo(f"📁 Sessions Directory: {sessions_path}")

        if sessions_path.exists():
            session_files = list(sessions_path.glob("*.jsonl"))
            click.echo(f"   Found {len(session_files)} session file(s)")
        else:
            click.echo("   ⚠️ Directory does not exist yet")

    click.echo("\n" + "=" * 30)
    click.echo("🚀 Usage:")
    click.echo("=" * 30)
    click.echo("1. Use Codex CLI normally:")
    click.echo("   codex 'write a function to parse JSON'")
    click.echo("\n2. Process latest session:")
    click.echo("   mlflow autolog codex --trace-latest")
    click.echo("\n3. View traces:")

    # Show correct UI command based on tracking URI
    if status_info.tracking_uri:
        # Check if using default ./mlruns directory
        if status_info.tracking_uri.endswith("/mlruns"):
            # For default mlruns, just show 'mlflow ui'
            click.echo("   mlflow ui")
        elif status_info.tracking_uri == "databricks":
            click.echo("   View traces in your Databricks workspace")
        else:
            # For custom URIs, show full command
            click.echo(f"   mlflow ui --backend-store-uri {status_info.tracking_uri}")
    else:
        click.echo("   mlflow ui")

    click.echo("\n🔧 To disable tracing:")
    click.echo("   mlflow autolog codex --disable")
