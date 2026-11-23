"""Filing a Jira ticket for each new issue, for a webhook whose destination is Jira."""

import base64
import logging
from typing import Any
from urllib.parse import quote

import requests

from mlflow.entities.webhook import (
    JiraDestinationConfig,
    Webhook,
    WebhookAction,
    WebhookEntity,
    WebhookEvent,
    WebhookTestResult,
)
from mlflow.environment_variables import MLFLOW_WEBHOOK_REQUEST_TIMEOUT
from mlflow.utils.validation import _validate_webhook_url
from mlflow.webhooks.formatters import detection_runs_url, issue_url
from mlflow.webhooks.types import WebhookPayload

_logger = logging.getLogger(__name__)

# Issue metadata key recording the ticket a Jira webhook filed for the issue. Per webhook, since two
# Jira webhooks can file into different projects.
JIRA_TICKET_METADATA_KEY_PREFIX = "mlflow.webhook.jiraTicket."

# Jira rejects a summary longer than this.
_MAX_SUMMARY_LENGTH = 255

# The label every ticket carries, so a Jira filter or automation can find what MLflow filed.
JIRA_TICKET_LABEL = "mlflow-issue"


def jira_ticket_metadata_key(webhook: Webhook) -> str:
    return JIRA_TICKET_METADATA_KEY_PREFIX + webhook.webhook_id


def _jira_config(webhook: Webhook) -> JiraDestinationConfig:
    if webhook.jira_config is None:
        raise ValueError(f"Jira webhook {webhook.webhook_id} has no jira_config")
    return webhook.jira_config


def _auth_headers(webhook: Webhook) -> dict[str, str]:
    """
    The credential header. Jira Cloud takes an API token with its account's email as basic auth;
    Jira Data Center takes a personal access token as a bearer token.
    """
    token = webhook.secret or ""
    if user_email := _jira_config(webhook).user_email:
        credentials = base64.b64encode(f"{user_email}:{token}".encode()).decode()
        return {"Authorization": f"Basic {credentials}"}
    return {"Authorization": f"Bearer {token}"}


def _api_url(webhook: Webhook, path: str) -> str:
    return f"{webhook.url.rstrip('/')}/rest/api/2/{path}"


def _ticket_description(payload: dict[str, Any]) -> str:
    """
    The ticket body: what the issue is and where to find it in MLflow.

    Built from the same fields as the ``issue.created`` payload, so the issue's description and
    root causes -- which detection writes from trace content -- stay in MLflow.
    """
    experiment_id = str(payload["experiment_id"])
    issue_id = str(payload["issue_id"])
    lines = [
        f"MLflow found an issue in experiment {experiment_id}.",
        "",
        f"Issue: {payload.get('name') or issue_id}",
        f"Issue ID: {issue_id}",
    ]
    if severity := payload.get("severity"):
        lines.append(f"Severity: {severity}")
    if categories := payload.get("categories"):
        lines.append("Categories: " + ", ".join(map(str, categories)))
    if source_run_id := payload.get("source_run_id"):
        lines.append(f"Detection run: {source_run_id}")
    if link := issue_url(experiment_id, issue_id):
        lines += ["", f"Open the issue in MLflow: {link}"]
    if runs := detection_runs_url(experiment_id):
        lines.append(f"Detection run history: {runs}")
    return "\n".join(lines)


def jira_ticket_fields(config: JiraDestinationConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """The ``fields`` of the Jira create-issue request for an ``issue.created`` payload."""
    summary = f"[MLflow] {payload.get('name') or payload['issue_id']}"
    if len(summary) > _MAX_SUMMARY_LENGTH:
        summary = summary[: _MAX_SUMMARY_LENGTH - 1] + "…"
    return {
        "project": {"key": config.project_key},
        "issuetype": {"name": config.issue_type},
        "summary": summary,
        "description": _ticket_description(payload),
        "labels": [JIRA_TICKET_LABEL],
    }


def _get_tracking_store():
    # Imported here: the handlers module imports the delivery module, which imports this one.
    from mlflow.server.handlers import _get_tracking_store

    return _get_tracking_store()


def file_jira_ticket(
    webhook: Webhook,
    event: WebhookEvent,
    payload: WebhookPayload,
    session: requests.Session,
) -> requests.Response | None:
    """
    File a Jira ticket for a new issue and record its key on the issue.

    Returns ``None`` without calling Jira when there is nothing to file: an event other than
    ``issue.created``, which a Jira webhook cannot subscribe to, or an issue this webhook already
    filed a ticket for -- a delivery retried after the ticket was filed, say. Recording the key is
    what makes that check work, so a ticket that was filed but could not be recorded is logged.
    """
    if event != WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED):
        _logger.debug("Jira webhook %s does not file tickets for %s", webhook.webhook_id, event)
        return None
    _validate_webhook_url(webhook.url)

    payload = dict(payload)
    issue_id = str(payload["issue_id"])
    metadata_key = jira_ticket_metadata_key(webhook)
    store = _get_tracking_store()
    if existing := (store.get_issue(issue_id).metadata or {}).get(metadata_key):
        _logger.info(
            "Jira webhook %s already filed %s for issue %s; not filing another",
            webhook.webhook_id,
            existing,
            issue_id,
        )
        return None

    response = session.post(
        _api_url(webhook, "issue"),
        json={"fields": jira_ticket_fields(_jira_config(webhook), payload)},
        headers=_auth_headers(webhook),
        timeout=MLFLOW_WEBHOOK_REQUEST_TIMEOUT.get(),
    )
    if response.status_code >= 400:
        # Jira explains a rejected ticket -- a missing field, an unknown issue type -- in the body.
        _logger.error(
            "Jira rejected the ticket for issue %s from webhook %s with status %s: %s",
            issue_id,
            webhook.webhook_id,
            response.status_code,
            response.text[:1000],
        )
        return response

    ticket_key = response.json().get("key")
    try:
        store.update_issue(issue_id, metadata={metadata_key: ticket_key})
    except Exception:
        _logger.warning(
            "Filed Jira ticket %s for issue %s but could not record it on the issue, so a retried "
            "delivery could file another",
            ticket_key,
            issue_id,
            exc_info=True,
        )
    return response


def check_jira_destination(webhook: Webhook, session: requests.Session) -> WebhookTestResult:
    """
    Check that a Jira webhook can file tickets: its site is reachable, its credential is accepted,
    and it can see its project. Files nothing, unlike testing any other webhook.
    """
    try:
        _validate_webhook_url(webhook.url)
        project_key = _jira_config(webhook).project_key
        response = session.get(
            _api_url(webhook, f"project/{quote(project_key, safe='')}"),
            headers=_auth_headers(webhook),
            timeout=MLFLOW_WEBHOOK_REQUEST_TIMEOUT.get(),
        )
    except Exception as e:
        return WebhookTestResult(success=False, error_message=f"Failed to reach Jira: {e!r}")
    return WebhookTestResult(
        success=response.status_code < 400,
        response_status=response.status_code,
        response_body=response.text,
    )
