"""
CLI commands for Codex integration with MLflow.
"""

import click

import mlflow
import mlflow.codex


@click.group("codex", help="Codex CLI integration commands.")
def codex_commands():
    """Codex CLI integration commands."""


@codex_commands.command("trace", help="Start the Session Tracer for Codex CLI.")
@click.option(
    "--experiment-id",
    type=str,
    default=None,
    help="MLflow experiment ID to log traces to.",
)
@click.option(
    "--tracking-uri",
    type=str,
    default=None,
    help="MLflow tracking URI. Defaults to current tracking URI.",
)
@click.option(
    "--history-path",
    type=str,
    default=None,
    help="Path to sessions directory. Defaults to ~/.codex/sessions",
)
@click.option(
    "--process-existing",
    is_flag=True,
    default=False,
    help="Process existing sessions before watching for new entries.",
)
def trace_server(experiment_id, tracking_uri, history_path, process_existing):
    """
    Start the Session Tracer for Codex CLI.

    The tracer watches Codex's session files and creates comprehensive MLflow traces
    with complete inputs, outputs, tool calls, token usage, and all attributes.

    Examples:

        # Start with defaults
        mlflow codex trace

        # Start with specific experiment ID
        mlflow codex trace --experiment-id 1

        # Process existing sessions first
        mlflow codex trace --experiment-id 1 --process-existing

        # Custom tracking URI
        mlflow codex trace --tracking-uri http://localhost:5000

        # Custom sessions directory
        mlflow codex trace --history-path /path/to/sessions
    """
    try:
        from mlflow.codex.session_tracer import run_session_tracer

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        click.echo(click.style("\n🚀 Starting Codex Session Tracer...\n", fg="green"))
        run_session_tracer(
            experiment_id=experiment_id,
            tracking_uri=tracking_uri,
            sessions_dir=history_path,  # Using history_path param for sessions_dir
            process_existing=process_existing,
        )

    except KeyboardInterrupt:
        click.echo(click.style("\n\n✅ Tracer stopped.\n", fg="green"))
    except Exception as e:
        click.echo(click.style(f"\n❌ Error: {e}\n", fg="red"), err=True)
        raise click.Abort()


# Create autolog group for backward compatibility
@click.group("autolog", help="Configure autologging for various frameworks and tools.")
def autolog_commands():
    pass


@autolog_commands.command("codex", help="Enable or disable autologging for Codex CLI.")
@click.option(
    "--tracking-uri",
    type=str,
    default=None,
    help="The URI to the MLflow tracking server.",
)
@click.option(
    "--experiment-name",
    type=str,
    default=None,
    help="The name of the experiment to log traces to. Defaults to 'Codex CLI Traces'.",
)
@click.option(
    "--experiment-id",
    type=str,
    default=None,
    help="The ID of the experiment to log traces to. If provided, overrides experiment-name.",
)
@click.option(
    "--disable",
    is_flag=True,
    default=False,
    help="Disable Codex autologging.",
)
@click.option(
    "--codex-home",
    type=str,
    default=None,
    help="Path to Codex home directory. Defaults to ~/.codex",
)
def codex_autolog(tracking_uri, experiment_name, experiment_id, disable, codex_home):
    """
    Enable or disable autologging for Codex CLI.

    This configures the experiment for Codex tracing. After running this,
    start the history tracer to capture traces.

    Examples:

        # Enable Codex autologging with defaults
        mlflow autolog codex

        # Enable with custom experiment name
        mlflow autolog codex --experiment-name "My Codex Project"

        # Then start the tracer
        mlflow codex trace --experiment-id 1
    """
    try:
        mlflow.codex.autolog(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            disable=disable,
            silent=False,
            codex_home=codex_home,
        )
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise click.Abort()


# Export both command groups
commands = codex_commands
