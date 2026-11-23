"""Webhook events for MLflow issues."""

import logging
from collections.abc import Iterable

from mlflow.entities.issue import Issue
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.webhooks.delivery import deliver_webhook
from mlflow.webhooks.types import IssueCreatedPayload

_logger = logging.getLogger(__name__)

ISSUE_CREATED_EVENT = WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED)


def issue_created_payload(issue: Issue) -> IssueCreatedPayload:
    return IssueCreatedPayload(
        issue_id=issue.issue_id,
        experiment_id=issue.experiment_id,
        name=issue.name,
        status=str(issue.status),
        severity=str(issue.severity) if issue.severity is not None else None,
        categories=[str(category) for category in issue.categories or []],
        source_run_id=issue.source_run_id,
        created_by=issue.created_by,
        created_timestamp=issue.created_timestamp,
    )


def get_webhook_store() -> AbstractStore | None:
    """
    The model registry store webhooks are registered in, or ``None`` when there is none.

    Webhooks live in the registry store rather than the tracking store that issues live in, so an
    issue event has to look it up separately. A failure to build it skips the notification rather
    than failing the operation the event describes.
    """
    # Imported here: the handlers module imports this one to fire issue events.
    from mlflow.server.handlers import _get_model_registry_store

    try:
        return _get_model_registry_store()
    except Exception:
        _logger.debug("No model registry store to deliver issue webhooks from", exc_info=True)
        return None


def deliver_issue_created_webhooks(
    issues: Iterable[Issue],
    store: AbstractStore | None = None,
    *,
    wait: bool = False,
) -> None:
    """
    Fire ``issue.created`` for each issue.

    Args:
        issues: Issues that were just created. Callers pass only newly created issues: an issue
            that issue detection refreshed because it had been reported before is not new.
        store: The registry store to read webhooks from. Looked up when omitted.
        wait: Block until the deliveries finish, for callers in a process that exits right after.
    """
    issues = list(issues)
    if not issues:
        return
    if store is None and (store := get_webhook_store()) is None:
        return
    for issue in issues:
        deliver_webhook(
            event=ISSUE_CREATED_EVENT,
            payload=issue_created_payload(issue),
            store=store,
            wait=wait,
        )
