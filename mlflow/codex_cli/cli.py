"""MLflow CLI commands for Codex CLI integration."""

from pathlib import Path

import click

from mlflow.codex_cli.config import (
    disable_tracing,
    get_tracing_status,
    setup_environment_config,
)
from mlflow.codex_cli.tracing import process_latest_session


@click.group("codex-cli")
def commands():
    """Commands for OpenAI Codex CLI autologging with MLflow."""


@commands.command("enable")
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
def enable(
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    sessions_dir: str | None,
) -> None:
    """Enable automatic tracing for Codex CLI sessions.

    This command configures MLflow to automatically trace Codex CLI sessions.
    After enabling, use the 'codex' command normally and traces will be created
    from session files stored in ~/.codex/sessions.

    Examples:

      # Enable tracing with local storage
      mlflow codex-cli enable

      # Enable tracing with custom tracking URI
      mlflow codex-cli enable -u file://./custom-mlruns

      # Enable tracing with Databricks
      mlflow codex-cli enable -u databricks -e 123456789

      # Enable tracing with custom sessions directory
      mlflow codex-cli enable -s ~/my-codex-sessions
    """
    click.echo("Configuring Codex CLI tracing...")

    # Set up configuration
    setup_environment_config(
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        sessions_dir=sessions_dir,
    )

    click.echo("✅ Codex CLI tracing enabled")

    # Show status
    _show_status()


@commands.command("disable")
def disable() -> None:
    """Disable automatic tracing for Codex CLI sessions.

    Example:
      mlflow codex-cli disable
    """
    if disable_tracing():
        click.echo("✅ Codex CLI tracing disabled")
    else:
        click.echo("❌ No Codex CLI configuration found - tracing was not enabled")


@commands.command("status")
def status() -> None:
    """Show current Codex CLI tracing status.

    Example:
      mlflow codex-cli status
    """
    _show_status()


@commands.command("trace-latest")
def trace_latest() -> None:
    """Process the latest Codex session and create a trace.

    This command manually processes the most recent Codex CLI session file
    and creates an MLflow trace. Useful for testing or one-off trace creation.

    Example:
      mlflow codex-cli trace-latest
    """
    status_info = get_tracing_status()

    if not status_info.enabled:
        click.echo("❌ Codex CLI tracing is not enabled")
        click.echo(f"   Reason: {status_info.reason}")
        click.echo("\n💡 Enable tracing with: mlflow codex-cli enable")
        return

    click.echo("Processing latest Codex session...")

    trace = process_latest_session()

    if trace:
        click.echo(f"✅ Created trace: {trace.info.request_id}")
        click.echo(f"   Session: {trace.info.trace_metadata.get('mlflow.trace.session', 'N/A')}")

        if status_info.tracking_uri:
            click.echo(f"   Tracking URI: {status_info.tracking_uri}")

        click.echo("\n💡 View trace in MLflow UI:")
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
        click.echo("\n💡 Enable tracing with: mlflow codex-cli enable")
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
    click.echo("   mlflow codex-cli trace-latest")
    click.echo("\n3. View traces:")
    click.echo("   mlflow ui")
    click.echo("\n🔧 To disable tracing:")
    click.echo("   mlflow codex-cli disable")
