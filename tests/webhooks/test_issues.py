from unittest import mock

import pytest

from mlflow.entities import Run, RunData, RunInfo, RunTag
from mlflow.entities.issue import Issue, IssueCategory, IssueSeverity, IssueStatus
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_ERROR,
    MLFLOW_ISSUE_DETECTION_TRIGGER,
    MLFLOW_ISSUE_DETECTION_WINDOW_END,
    MLFLOW_ISSUE_DETECTION_WINDOW_START,
)
from mlflow.webhooks.issues import (
    ISSUE_CREATED_EVENT,
    ISSUE_DETECTION_RUN_COMPLETED_EVENT,
    deliver_issue_created_webhooks,
    deliver_issue_detection_run_completed_webhook,
    get_webhook_store,
    issue_created_payload,
    issue_detection_run_completed_payload,
)
from mlflow.webhooks.types import (
    IssueCreatedPayload,
    IssueDetectionRunCompletedPayload,
    get_payload_class_for_event,
)


def _issue(**overrides) -> Issue:
    fields = {
        "issue_id": "iss-1",
        "experiment_id": "7",
        "name": "Refund policy is invented",
        "description": "The agent quotes a refund window from the trace: card 4111...",
        "status": IssueStatus.PENDING,
        "created_timestamp": 1704067200000,
        "last_updated_timestamp": 1704067200000,
        "severity": IssueSeverity.HIGH,
        "root_causes": ["The prompt never states the real policy"],
        "source_run_id": "run-1",
        "categories": [IssueCategory.CORRECTNESS, "custom_category"],
        "created_by": None,
    }
    return Issue(**(fields | overrides))


def test_issue_created_event_has_its_payload_class():
    assert ISSUE_CREATED_EVENT == WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED)
    assert str(ISSUE_CREATED_EVENT) == "issue.created"
    assert get_payload_class_for_event(ISSUE_CREATED_EVENT) is IssueCreatedPayload


def test_issue_created_payload_carries_metadata_but_not_trace_derived_text():
    payload = issue_created_payload(_issue())

    assert payload == {
        "issue_id": "iss-1",
        "experiment_id": "7",
        "name": "Refund policy is invented",
        "status": "pending",
        "severity": "high",
        "categories": ["correctness", "custom_category"],
        "source_run_id": "run-1",
        "created_by": None,
        "created_timestamp": 1704067200000,
    }


def test_issue_created_payload_without_severity_or_categories():
    payload = issue_created_payload(_issue(severity=None, categories=None, created_by="alice"))

    assert payload["severity"] is None
    assert payload["categories"] == []
    assert payload["created_by"] == "alice"


def test_deliver_issue_created_webhooks_fires_once_per_issue():
    store = mock.Mock()
    first = _issue(issue_id="iss-1")
    second = _issue(issue_id="iss-2")

    with mock.patch("mlflow.webhooks.issues.deliver_webhook") as deliver:
        deliver_issue_created_webhooks([first, second], store, wait=True)

    assert deliver.call_args_list == [
        mock.call(
            event=ISSUE_CREATED_EVENT,
            payload=issue_created_payload(first),
            store=store,
            wait=True,
        ),
        mock.call(
            event=ISSUE_CREATED_EVENT,
            payload=issue_created_payload(second),
            store=store,
            wait=True,
        ),
    ]


def test_deliver_issue_created_webhooks_skips_the_store_lookup_without_issues():
    with (
        mock.patch("mlflow.webhooks.issues.get_webhook_store") as get_store,
        mock.patch("mlflow.webhooks.issues.deliver_webhook") as deliver,
    ):
        deliver_issue_created_webhooks([])

    get_store.assert_not_called()
    deliver.assert_not_called()


def test_deliver_issue_created_webhooks_skips_delivery_without_a_registry_store():
    with (
        mock.patch("mlflow.webhooks.issues.get_webhook_store", return_value=None),
        mock.patch("mlflow.webhooks.issues.deliver_webhook") as deliver,
    ):
        deliver_issue_created_webhooks([_issue()])

    deliver.assert_not_called()


def test_get_webhook_store_returns_none_when_the_registry_store_cannot_be_built():
    with mock.patch(
        "mlflow.server.handlers._get_model_registry_store", side_effect=RuntimeError("no store")
    ):
        assert get_webhook_store() is None


def _detection_run(status: str, tags: dict[str, str]) -> Run:
    return Run(
        run_info=RunInfo(
            run_id="run-1",
            experiment_id="7",
            user_id="",
            status=status,
            start_time=1000,
            end_time=2000,
            lifecycle_stage="active",
        ),
        run_data=RunData(tags=[RunTag(key, value) for key, value in tags.items()]),
    )


def test_issue_detection_run_completed_event_has_its_payload_class():
    assert str(ISSUE_DETECTION_RUN_COMPLETED_EVENT) == "issue_detection_run.completed"
    assert (
        get_payload_class_for_event(ISSUE_DETECTION_RUN_COMPLETED_EVENT)
        is IssueDetectionRunCompletedPayload
    )


def test_issue_detection_run_completed_payload_for_a_successful_manual_run():
    run = _detection_run("FINISHED", {"model": "openai:/gpt-4o"})

    payload = issue_detection_run_completed_payload(
        run,
        total_traces_analyzed=40,
        issue_count=3,
        new_issue_ids=["iss-1"],
        total_cost_usd=0.25,
    )

    assert payload == {
        "run_id": "run-1",
        "experiment_id": "7",
        "status": "succeeded",
        "trigger": "manual",
        "start_time": 1000,
        "end_time": 2000,
        "error_message": None,
        "model": "openai:/gpt-4o",
        "total_traces_analyzed": 40,
        "issue_count": 3,
        "new_issue_ids": ["iss-1"],
        "total_cost_usd": 0.25,
        "window_start": None,
        "window_end": None,
    }


def test_issue_detection_run_completed_payload_for_a_failed_scheduled_run():
    run = _detection_run(
        "FAILED",
        {
            MLFLOW_ISSUE_DETECTION_TRIGGER: "schedule",
            MLFLOW_ISSUE_DETECTION_ERROR: "MlflowException: invalid API key",
            MLFLOW_ISSUE_DETECTION_WINDOW_START: "100",
            MLFLOW_ISSUE_DETECTION_WINDOW_END: "900",
        },
    )

    payload = issue_detection_run_completed_payload(
        run, total_traces_analyzed=0, issue_count=0, new_issue_ids=[], total_cost_usd=None
    )

    assert payload["status"] == "failed"
    assert payload["trigger"] == "schedule"
    assert payload["error_message"] == "MlflowException: invalid API key"
    assert payload["window_start"] == 100
    assert payload["window_end"] == 900
    assert payload["model"] is None


@pytest.mark.parametrize("raw_bound", ["", "not-a-number"])
def test_issue_detection_run_completed_payload_ignores_an_unreadable_window(raw_bound):
    run = _detection_run("FINISHED", {MLFLOW_ISSUE_DETECTION_WINDOW_START: raw_bound})

    payload = issue_detection_run_completed_payload(
        run, total_traces_analyzed=0, issue_count=0, new_issue_ids=[], total_cost_usd=0.0
    )

    assert payload["window_start"] is None


def test_a_successful_run_carries_no_error_message_even_with_a_stale_error_tag():
    run = _detection_run("FINISHED", {MLFLOW_ISSUE_DETECTION_ERROR: "from an earlier attempt"})

    payload = issue_detection_run_completed_payload(
        run, total_traces_analyzed=1, issue_count=0, new_issue_ids=[], total_cost_usd=0.0
    )

    assert payload["error_message"] is None


def test_deliver_issue_detection_run_completed_webhook():
    store = mock.Mock()
    payload = IssueDetectionRunCompletedPayload.example()

    with mock.patch("mlflow.webhooks.issues.deliver_webhook") as deliver:
        deliver_issue_detection_run_completed_webhook(payload, store, wait=True)

    deliver.assert_called_once_with(
        event=ISSUE_DETECTION_RUN_COMPLETED_EVENT, payload=payload, store=store, wait=True
    )


def test_deliver_issue_detection_run_completed_webhook_without_a_registry_store():
    with (
        mock.patch("mlflow.webhooks.issues.get_webhook_store", return_value=None),
        mock.patch("mlflow.webhooks.issues.deliver_webhook") as deliver,
    ):
        deliver_issue_detection_run_completed_webhook(IssueDetectionRunCompletedPayload.example())

    deliver.assert_not_called()
