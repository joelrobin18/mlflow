"""Messages for webhook destinations that expect their own format instead of MLflow's envelope."""

import json
from typing import Any
from urllib.parse import quote

from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.environment_variables import MLFLOW_WEBHOOK_LINK_BASE_URL
from mlflow.webhooks.types import WebhookPayload

# Slack rejects a section whose text is longer than 3,000 characters, so text taken from a payload
# is cut short of that.
_SLACK_SECTION_TEXT_LIMIT = 2900

# How much of a failure reason a message quotes. The run history carries the rest.
_ERROR_PREVIEW_LENGTH = 500


def _ui_url(route: str) -> str | None:
    """A link into the MLflow UI, or ``None`` when the server has no UI URL to link to."""
    if not (base := MLFLOW_WEBHOOK_LINK_BASE_URL.get()):
        return None
    # The UI routes with a hash, so the route follows ``#``.
    return f"{base.rstrip('/')}/#{route}"


def issue_url(experiment_id: str, issue_id: str) -> str | None:
    """A link to an issue on its experiment's Issues tab."""
    return _ui_url(
        f"/experiments/{quote(experiment_id, safe='')}/issues"
        f"?selectedIssueId={quote(issue_id, safe='')}"
    )


def detection_runs_url(experiment_id: str) -> str | None:
    """A link to an experiment's issue detection run history."""
    return _ui_url(f"/experiments/{quote(experiment_id, safe='')}/issues/runs")


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _slack_escape(text: str) -> str:
    # Slack reads &, < and > as markup, so text from a payload has to escape them or it could
    # forge a link or a mention.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _slack_link(url: str | None, label: str) -> str:
    return f"<{url}|{_slack_escape(label)}>" if url else _slack_escape(label)


def _slack_message(text: str, lines: list[str], context: list[str]) -> dict[str, Any]:
    """
    A Slack message: ``text`` is what notifications and clients without blocks show, the blocks
    what Slack renders.

    Args:
        text: Plain text, escaped here since Slack reads it as markup too.
        lines: The message body as Slack markup, already escaped.
        context: Small print under the body as Slack markup, already escaped.
    """
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": _truncate("\n".join(lines), _SLACK_SECTION_TEXT_LIMIT),
            },
        }
    ]
    if context:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": item} for item in context],
        })
    return {"text": _slack_escape(text), "blocks": blocks}


def _issue_created_message(payload: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(payload.get("experiment_id", ""))
    name = str(payload.get("name") or payload.get("issue_id", ""))
    severity = payload.get("severity")
    headline = f"New {severity} severity issue" if severity else "New issue"
    context = [f"Experiment {_slack_escape(experiment_id)}"]
    if categories := payload.get("categories"):
        context.append("Categories: " + _slack_escape(", ".join(map(str, categories))))
    if status := payload.get("status"):
        context.append(f"Status: {_slack_escape(str(status))}")
    return _slack_message(
        text=f"{headline} in experiment {experiment_id}: {name}",
        lines=[
            f"*{headline}*: "
            + _slack_link(issue_url(experiment_id, str(payload.get("issue_id", ""))), name)
        ],
        context=context,
    )


def _issue_detection_run_completed_message(payload: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(payload.get("experiment_id", ""))
    trigger = "Scheduled" if payload.get("trigger") == "schedule" else "Manual"
    runs_link = _slack_link(detection_runs_url(experiment_id), "run history")
    if payload.get("status") == "failed":
        error = _truncate(
            str(payload.get("error_message") or "unknown error"), _ERROR_PREVIEW_LENGTH
        )
        headline = f"{trigger} issue detection failed in experiment {experiment_id}"
        lines = [
            f"*{_slack_escape(headline)}*",
            f"`{_slack_escape(error)}`",
            f"See the {runs_link}.",
        ]
    else:
        new_issue_count = len(payload.get("new_issue_ids") or [])
        issue_count = payload.get("issue_count", 0)
        traces = payload.get("total_traces_analyzed", 0)
        headline = f"{trigger} issue detection finished in experiment {experiment_id}"
        lines = [
            f"*{_slack_escape(headline)}*",
            f"{new_issue_count} new of {issue_count} issues found across {traces} traces. "
            f"See the {runs_link}.",
        ]
    context = []
    if model := payload.get("model"):
        context.append(f"Model: {_slack_escape(str(model))}")
    if (cost := payload.get("total_cost_usd")) is not None:
        context.append(f"Cost: ${cost:.2f}")
    return _slack_message(text=headline, lines=lines, context=context)


def _generic_message(event: WebhookEvent, payload: WebhookPayload) -> dict[str, Any]:
    """Any other event, as its name and its payload as JSON."""
    rendered = _truncate(json.dumps(payload, indent=2, default=str), _SLACK_SECTION_TEXT_LIMIT - 20)
    return _slack_message(
        text=f"MLflow event: {event}",
        lines=[f"*MLflow event:* `{event}`", f"```{_slack_escape(rendered)}```"],
        context=[],
    )


def format_slack_message(event: WebhookEvent, payload: WebhookPayload) -> dict[str, Any]:
    """
    The Slack incoming-webhook message for ``event``.

    Issue events read as sentences that link to the issue and the run history when
    ``MLFLOW_WEBHOOK_LINK_BASE_URL`` is set; any other event is shown as its payload.
    """
    if event.entity == WebhookEntity.ISSUE and event.action == WebhookAction.CREATED:
        return _issue_created_message(dict(payload))
    if (
        event.entity == WebhookEntity.ISSUE_DETECTION_RUN
        and event.action == WebhookAction.COMPLETED
    ):
        return _issue_detection_run_completed_message(dict(payload))
    return _generic_message(event, payload)
