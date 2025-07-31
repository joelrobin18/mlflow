from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
import logging
from mlflow.agno.autolog import _patched_agent_arun, _patched_agent_run, _patched_tool_aexecute, _patched_tool_execute, _patched_agent_print_response

FLAVOR_NAME = "agno"
_logger = logging.getLogger(__name__)

@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    disable: bool = False,
    silent: bool = False,
) -> None:

    try:
        import agno.agent  # type: ignore[attr-defined]
        import agno.team  # type: ignore[attr-defined]
        import agno.models.base  # type: ignore[attr-defined]
        import agno.tools.function  # type: ignore[attr-defined]
    except Exception as e:
        _logger.warning("Failed to import Agno modules for MLflow integration: %s", e)
        return

    # Patch Agent.run and Agent.arun
    safe_patch(
        FLAVOR_NAME,
        agno.agent.Agent,
        "run",
        _patched_agent_run,
        manage_run=False,
    )
    # async run
    safe_patch(
        FLAVOR_NAME,
        agno.agent.Agent,
        "arun",
        _patched_agent_arun,
        manage_run=False,
    )
    # Patch Team.run and Team.arun
    safe_patch(
        FLAVOR_NAME,
        agno.team.Team,
        "run",
        _patched_agent_run,
        manage_run=False,
    )
    safe_patch(
        FLAVOR_NAME,
        agno.team.Team,
        "arun",
        _patched_agent_arun,
        manage_run=False,
    )

    # Patch tool execution
    if hasattr(agno.tools.function, "FunctionCall"):
        if hasattr(agno.tools.function.FunctionCall, "execute"):
            safe_patch(
                FLAVOR_NAME,
                agno.tools.function.FunctionCall,
                "execute",
                _patched_tool_execute,
                manage_run=False,
            )
        if hasattr(agno.tools.function.FunctionCall, "aexecute"):
            safe_patch(
                FLAVOR_NAME,
                agno.tools.function.FunctionCall,
                "aexecute",
                _patched_tool_aexecute,
                manage_run=False,
            )

    # Patch print_response on Agent and Team to create top-level spans for user-facing calls
    if hasattr(agno.agent.Agent, "print_response"):
        safe_patch(
            FLAVOR_NAME,
            agno.agent.Agent,
            "print_response",
            _patched_agent_print_response,
            manage_run=False,
        )
    if hasattr(agno.team.Team, "print_response"):
        safe_patch(
            FLAVOR_NAME,
            agno.team.Team,
            "print_response",
            _patched_agent_print_response,
            manage_run=False,
        )

    # Integration is registered at runtime; autologging is now active.
    if not silent:
        _logger.info("MLflow Agno autologging is enabled.")
