import json
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from urllib3.util.retry import Retry

from mlflow.entities.webhook import (
    JiraDestinationConfig,
    Webhook,
    WebhookAction,
    WebhookDestinationType,
    WebhookEntity,
    WebhookEvent,
)
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.store.model_registry.file_store import FileStore
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from mlflow.webhooks.delivery import (
    _create_webhook_session,
    _send_webhook_request,
    deliver_webhook,
)
from mlflow.webhooks.delivery import test_webhook as send_test_webhook
from mlflow.webhooks.formatters import format_slack_message


@pytest.fixture
def file_store(tmp_path: Path) -> FileStore:
    pytest.skip("FileStore is no longer supported.")
    return FileStore(str(tmp_path))


@pytest.fixture
def sql_store(tmp_path: Path) -> SqlAlchemyStore:
    db_file = tmp_path / "test.db"
    db_uri = f"sqlite:///{db_file}"
    return SqlAlchemyStore(db_uri)


@pytest.fixture
def webhook_event() -> WebhookEvent:
    return WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)


@pytest.fixture
def webhook_payload() -> dict[str, str]:
    return {"name": "test_model", "description": "Test model"}


@pytest.fixture
def webhook(webhook_event: WebhookEvent) -> Webhook:
    return Webhook(
        webhook_id="wh-1",
        name="test",
        url="https://example.com/hook",
        events=[webhook_event],
        creation_timestamp=0,
        last_updated_timestamp=0,
        workspace="team-a",
    )


def test_deliver_webhook_exits_early_for_file_store(
    file_store: FileStore, webhook_event: WebhookEvent, webhook_payload: dict[str, str]
) -> None:
    pytest.skip("FileStore is no longer supported.")
    with patch("mlflow.webhooks.delivery._deliver_webhook_impl") as mock_impl:
        deliver_webhook(
            event=webhook_event,
            payload=webhook_payload,
            store=file_store,
        )

        # _deliver_webhook_impl should not be called for FileStore
        mock_impl.assert_not_called()


def test_deliver_webhook_calls_impl_for_sql_store(
    sql_store: SqlAlchemyStore, webhook_event: WebhookEvent, webhook_payload: dict[str, str]
) -> None:
    with patch("mlflow.webhooks.delivery._deliver_webhook_impl") as mock_impl:
        deliver_webhook(
            event=webhook_event,
            payload=webhook_payload,
            store=sql_store,
        )

        # _deliver_webhook_impl should be called for SqlAlchemyStore
        mock_impl.assert_called_once_with(
            event=webhook_event,
            payload=webhook_payload,
            store=sql_store,
        )


def test_deliver_webhook_handles_exception_for_sql_store(
    sql_store: SqlAlchemyStore, webhook_event: WebhookEvent, webhook_payload: dict[str, str]
) -> None:
    with (
        patch("mlflow.webhooks.delivery._deliver_webhook_impl", side_effect=Exception("Test")),
        patch("mlflow.webhooks.delivery._logger") as mock_logger,
    ):
        # This should not raise an exception
        deliver_webhook(
            event=webhook_event,
            payload=webhook_payload,
            store=sql_store,
        )

        # Verify that the error was logged
        mock_logger.error.assert_called_once()
        assert "Failed to deliver webhook for event" in str(mock_logger.error.call_args)


def test_deliver_webhook_no_exception_for_file_store(
    file_store: FileStore, webhook_event: WebhookEvent, webhook_payload: dict[str, str]
) -> None:
    pytest.skip("FileStore is no longer supported.")
    with (
        patch(
            "mlflow.webhooks.delivery._deliver_webhook_impl", side_effect=Exception("Test")
        ) as mock_impl,
        patch("mlflow.webhooks.delivery._logger") as mock_logger,
    ):
        # This should not raise an exception and should return early
        deliver_webhook(
            event=webhook_event,
            payload=webhook_payload,
            store=file_store,
        )

        # _deliver_webhook_impl should not be called, so no error should be logged
        mock_impl.assert_not_called()
        mock_logger.error.assert_not_called()


@pytest.mark.parametrize(
    ("urllib3_version", "expected_kwargs"),
    [
        ("1.26.18", {}),
        ("2.0.0", {"backoff_max": 60.0, "backoff_jitter": 1.0}),
    ],
)
def test_create_webhook_session_retry_kwargs_depend_on_urllib3_version(
    urllib3_version: str, expected_kwargs: dict[str, float]
):
    with (
        patch("mlflow.webhooks.delivery.urllib3.__version__", urllib3_version),
        patch("mlflow.webhooks.delivery.Retry", return_value=Retry(total=0)) as mock_retry,
    ):
        _create_webhook_session()

    mock_retry.assert_called_once()
    kwargs = mock_retry.call_args.kwargs
    assert {k: kwargs[k] for k in ("backoff_max", "backoff_jitter") if k in kwargs} == (
        expected_kwargs
    )


def test_deliver_webhook_does_not_create_session_without_webhooks(
    sql_store: SqlAlchemyStore, webhook_event: WebhookEvent, webhook_payload: dict[str, str]
):
    with patch("mlflow.webhooks.delivery._get_or_create_webhook_session") as mock_get_session:
        deliver_webhook(event=webhook_event, payload=webhook_payload, store=sql_store)

    mock_get_session.assert_not_called()


def test_deliver_webhook_with_wait_returns_after_the_delivery_finishes(
    sql_store: SqlAlchemyStore,
    webhook: Webhook,
    webhook_event: WebhookEvent,
    webhook_payload: dict[str, str],
):
    delivered = threading.Event()

    def slow_send(*args, **kwargs):
        time.sleep(0.2)
        delivered.set()

    with (
        patch("mlflow.webhooks.delivery._get_cached_webhooks_by_event", return_value=[webhook]),
        patch("mlflow.webhooks.delivery._send_webhook_with_error_handling", side_effect=slow_send),
    ):
        deliver_webhook(event=webhook_event, payload=webhook_payload, store=sql_store, wait=True)

        assert delivered.is_set()


def test_test_webhook_rejects_private_ip():
    event = WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)
    webhook = Webhook(
        webhook_id="wh-1",
        name="test",
        url="https://localhost/hook",
        events=[event],
        creation_timestamp=0,
        last_updated_timestamp=0,
    )

    result = send_test_webhook(webhook)

    assert result.success is False
    assert "must not resolve to a non-public" in result.error_message


def test_send_webhook_request_includes_workspace_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    webhook: Webhook,
    webhook_event: WebhookEvent,
    webhook_payload: dict[str, str],
) -> None:
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    session = Mock()
    session.post.return_value = Mock(status_code=200, text="ok")

    with patch("mlflow.webhooks.delivery._validate_webhook_url"):
        _send_webhook_request(webhook, webhook_payload, webhook_event, session)

    actual_payload = json.loads(session.post.call_args.kwargs["data"].decode("utf-8"))
    assert actual_payload["entity"] == webhook_event.entity.value
    assert actual_payload["action"] == webhook_event.action.value
    assert actual_payload["workspace"] == webhook.workspace
    assert actual_payload["data"] == webhook_payload
    assert "timestamp" in actual_payload


def test_send_webhook_request_omits_workspace_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    webhook: Webhook,
    webhook_event: WebhookEvent,
    webhook_payload: dict[str, str],
) -> None:
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    session = Mock()
    session.post.return_value = Mock(status_code=200, text="ok")

    with patch("mlflow.webhooks.delivery._validate_webhook_url"):
        _send_webhook_request(webhook, webhook_payload, webhook_event, session)

    actual_payload = json.loads(session.post.call_args.kwargs["data"].decode("utf-8"))
    assert "workspace" not in actual_payload


def test_deliver_webhook_skips_webhooks_limited_to_other_experiments(sql_store: SqlAlchemyStore):
    event = WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED)

    def issue_webhook(webhook_id, experiment_ids):
        return Webhook(
            webhook_id=webhook_id,
            name=webhook_id,
            url="https://example.com/hook",
            events=[event],
            creation_timestamp=0,
            last_updated_timestamp=0,
            experiment_ids=experiment_ids,
        )

    webhooks = [
        issue_webhook("everywhere", []),
        issue_webhook("this-experiment", ["7"]),
        issue_webhook("other-experiment", ["8"]),
    ]
    with (
        patch("mlflow.webhooks.delivery._get_cached_webhooks_by_event", return_value=webhooks),
        patch("mlflow.webhooks.delivery._send_webhook_with_error_handling") as send,
    ):
        deliver_webhook(event=event, payload={"experiment_id": "7"}, store=sql_store, wait=True)

    assert sorted(call.args[0].webhook_id for call in send.call_args_list) == [
        "everywhere",
        "this-experiment",
    ]


def test_send_webhook_request_posts_a_slack_message_to_a_slack_webhook(
    webhook_event: WebhookEvent,
    webhook_payload: dict[str, str],
) -> None:
    slack_webhook = Webhook(
        webhook_id="wh-slack",
        name="slack",
        url="https://hooks.slack.com/services/T/B/X",
        events=[webhook_event],
        creation_timestamp=0,
        last_updated_timestamp=0,
        destination_type=WebhookDestinationType.SLACK,
    )
    session = Mock()
    session.post.return_value = Mock(status_code=200, text="ok")

    with patch("mlflow.webhooks.delivery._validate_webhook_url"):
        _send_webhook_request(slack_webhook, webhook_payload, webhook_event, session)

    body = json.loads(session.post.call_args.kwargs["data"].decode("utf-8"))
    assert body == format_slack_message(webhook_event, webhook_payload)
    assert "data" not in body


def _jira_webhook(event):
    return Webhook(
        webhook_id="wh-jira",
        name="jira",
        url="https://example.atlassian.net",
        events=[event],
        creation_timestamp=0,
        last_updated_timestamp=0,
        secret="api-token",
        destination_type=WebhookDestinationType.JIRA,
        jira_config=JiraDestinationConfig(project_key="OPS", issue_type="Bug"),
    )


def test_send_webhook_request_files_a_ticket_for_a_jira_webhook() -> None:
    event = WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED)
    webhook = _jira_webhook(event)
    session = Mock()

    with patch("mlflow.webhooks.delivery.file_jira_ticket") as file_ticket:
        _send_webhook_request(webhook, {"issue_id": "iss-1"}, event, session)

    file_ticket.assert_called_once_with(webhook, event, {"issue_id": "iss-1"}, session)
    session.post.assert_not_called()


def test_testing_a_jira_webhook_checks_its_project_instead_of_filing() -> None:
    webhook = _jira_webhook(WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED))

    with (
        patch("mlflow.webhooks.delivery.check_jira_destination") as check,
        patch("mlflow.webhooks.delivery.file_jira_ticket") as file_ticket,
    ):
        result = send_test_webhook(webhook)

    assert result is check.return_value
    file_ticket.assert_not_called()
