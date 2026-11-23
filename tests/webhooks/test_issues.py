from unittest import mock

from mlflow.entities.issue import Issue, IssueCategory, IssueSeverity, IssueStatus
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.webhooks.issues import (
    ISSUE_CREATED_EVENT,
    deliver_issue_created_webhooks,
    get_webhook_store,
    issue_created_payload,
)
from mlflow.webhooks.types import IssueCreatedPayload, get_payload_class_for_event


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
