"""Webhook events for MLflow issues."""

import logging
from collections.abc import Iterable, Mapping

from mlflow.entities.issue import Issue
from mlflow.entities.run import Run
from mlflow.entities.run_status import RunStatus
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_ERROR,
    MLFLOW_ISSUE_DETECTION_TRIGGER,
    MLFLOW_ISSUE_DETECTION_TRIGGER_SCHEDULE,
    MLFLOW_ISSUE_DETECTION_WINDOW_END,
    MLFLOW_ISSUE_DETECTION_WINDOW_START,
)
from mlflow.webhooks.delivery import deliver_webhook
from mlflow.webhooks.types import IssueCreatedPayload, IssueDetectionRunCompletedPayload

_logger = logging.getLogger(__name__)

ISSUE_CREATED_EVENT = WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED)
ISSUE_DETECTION_RUN_COMPLETED_EVENT = WebhookEvent(
    WebhookEntity.ISSUE_DETECTION_RUN, WebhookAction.COMPLETED
)


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


def _int_tag(tags: Mapping[str, str], key: str) -> int | None:
    try:
        return int(tags[key])
    except (KeyError, TypeError, ValueError):
        return None


def issue_detection_run_completed_payload(
    run: Run,
    *,
    total_traces_analyzed: int,
    issue_count: int,
    new_issue_ids: list[str],
    total_cost_usd: float | None,
) -> IssueDetectionRunCompletedPayload:
    """
    The ``issue_detection_run.completed`` payload for a detection run that has ended.

    Read from the run rather than handed over by the caller, so that a manual and a scheduled run
    describe themselves the same way: the trigger, model and window are the tags the run was
    created with, and the failure reason is the tag the job records before failing it.

    Args:
        run: The detection run, already terminated.
        total_traces_analyzed: How many traces the run analyzed.
        issue_count: How many issues the run reported, counting issues it found again.
        new_issue_ids: The issues the run filed.
        total_cost_usd: What the run's model calls cost, or None when unknown.
    """
    tags = run.data.tags
    succeeded = run.info.status == RunStatus.to_string(RunStatus.FINISHED)
    scheduled = tags.get(MLFLOW_ISSUE_DETECTION_TRIGGER) == MLFLOW_ISSUE_DETECTION_TRIGGER_SCHEDULE
    return IssueDetectionRunCompletedPayload(
        run_id=run.info.run_id,
        experiment_id=run.info.experiment_id,
        status="succeeded" if succeeded else "failed",
        trigger="schedule" if scheduled else "manual",
        start_time=run.info.start_time,
        end_time=run.info.end_time,
        error_message=None if succeeded else tags.get(MLFLOW_ISSUE_DETECTION_ERROR),
        model=tags.get("model"),
        total_traces_analyzed=total_traces_analyzed,
        issue_count=issue_count,
        new_issue_ids=list(new_issue_ids),
        total_cost_usd=total_cost_usd,
        window_start=_int_tag(tags, MLFLOW_ISSUE_DETECTION_WINDOW_START),
        window_end=_int_tag(tags, MLFLOW_ISSUE_DETECTION_WINDOW_END),
    )


def deliver_issue_detection_run_completed_webhook(
    payload: IssueDetectionRunCompletedPayload,
    store: AbstractStore | None = None,
    *,
    wait: bool = False,
) -> None:
    """
    Fire ``issue_detection_run.completed``.

    Args:
        payload: The run's payload, from :py:func:`issue_detection_run_completed_payload`.
        store: The registry store to read webhooks from. Looked up when omitted.
        wait: Block until the deliveries finish, for callers in a process that exits right after.
    """
    if store is None and (store := get_webhook_store()) is None:
        return
    deliver_webhook(
        event=ISSUE_DETECTION_RUN_COMPLETED_EVENT, payload=payload, store=store, wait=wait
    )
