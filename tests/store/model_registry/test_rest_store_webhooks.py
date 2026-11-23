from pathlib import Path
from typing import Iterator

import pytest
from cryptography.fernet import Fernet

from mlflow.entities.webhook import (
    JiraDestinationConfig,
    WebhookAction,
    WebhookDestinationType,
    WebhookEntity,
    WebhookEvent,
    WebhookStatus,
)
from mlflow.environment_variables import MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY
from mlflow.exceptions import MlflowException
from mlflow.server import handlers
from mlflow.server.fastapi_app import app
from mlflow.server.handlers import initialize_backend_stores
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.utils.rest_utils import MlflowHostCreds

from tests.helper_functions import get_safe_port
from tests.tracking.integration_test_utils import ServerThread


@pytest.fixture
def store(tmp_path: Path, db_uri: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[RestStore]:
    """Set up a local MLflow server with proper webhook encryption key support."""
    # Set up encryption key for webhooks using monkeypatch
    encryption_key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv(MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY.name, encryption_key)

    # Force-reset backend stores before each test
    handlers._tracking_store = None
    handlers._model_registry_store = None
    initialize_backend_stores(db_uri, default_artifact_root=tmp_path.as_uri())

    # Start server and return RestStore
    with ServerThread(app, get_safe_port()) as url:
        yield RestStore(lambda: MlflowHostCreds(url))


def test_create_webhook(store: RestStore):
    webhook = store.create_webhook(
        name="test_webhook",
        url="https://example.com/webhook",
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
    )
    assert webhook.name == "test_webhook"
    assert webhook.url == "https://example.com/webhook"
    assert webhook.secret is None
    assert webhook.events == [WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)]

    webhook = store.get_webhook(webhook.webhook_id)
    assert webhook.name == "test_webhook"
    assert webhook.url == "https://example.com/webhook"
    assert webhook.secret is None

    # With secret
    webhook_with_secret = store.create_webhook(
        name="test_webhook_with_secret",
        url="https://example.com/webhook_with_secret",
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
        secret="my_secret",
    )
    assert webhook_with_secret.name == "test_webhook_with_secret"
    assert webhook_with_secret.url == "https://example.com/webhook_with_secret"
    assert webhook_with_secret.secret is None
    assert webhook_with_secret.events == [
        WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)
    ]

    # Multiple events
    webhook_multiple_events = store.create_webhook(
        name="test_webhook_multiple_events",
        url="https://example.com/webhook_multiple_events",
        events=[
            WebhookEvent(WebhookEntity.MODEL_VERSION_ALIAS, WebhookAction.CREATED),
            WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED),
        ],
    )
    assert webhook_multiple_events.name == "test_webhook_multiple_events"
    assert webhook_multiple_events.url == "https://example.com/webhook_multiple_events"
    assert sorted(
        webhook_multiple_events.events, key=lambda e: (e.entity.value, e.action.value)
    ) == [
        WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED),
        WebhookEvent(WebhookEntity.MODEL_VERSION_ALIAS, WebhookAction.CREATED),
    ]
    assert webhook_multiple_events.secret is None


def test_get_webhook(store: RestStore):
    events = [WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)]
    created_webhook = store.create_webhook(
        name="test_webhook", url="https://example.com/webhook", events=events
    )
    retrieved_webhook = store.get_webhook(created_webhook.webhook_id)
    assert retrieved_webhook.webhook_id == created_webhook.webhook_id
    assert retrieved_webhook.name == "test_webhook"
    assert retrieved_webhook.url == "https://example.com/webhook"
    assert retrieved_webhook.events == events


def test_get_webhook_not_found(store: RestStore):
    with pytest.raises(MlflowException, match="Webhook with ID nonexistent not found"):
        store.get_webhook("nonexistent")


def test_list_webhooks(store: RestStore):
    # Create more webhooks than max_results
    for i in range(5):
        store.create_webhook(
            name=f"webhook{i}",
            url=f"https://example.com/{i}",
            events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
        )
    # Test pagination with max_results=2
    webhooks_page = store.list_webhooks(max_results=2)
    assert len(webhooks_page) == 2
    assert webhooks_page.token is not None
    # Get next page
    next_webhooks_page = store.list_webhooks(max_results=2, page_token=webhooks_page.token)
    assert len(next_webhooks_page) == 2
    assert next_webhooks_page.token is not None
    # Verify we don't get duplicates
    first_page_ids = {w.webhook_id for w in webhooks_page}
    second_page_ids = {w.webhook_id for w in next_webhooks_page}
    assert first_page_ids.isdisjoint(second_page_ids)


def test_update_webhook(store: RestStore):
    events = [WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)]
    webhook = store.create_webhook(
        name="original_name", url="https://example.com/original", events=events
    )
    # Update webhook
    new_events = [
        WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED),
        WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED),
    ]
    updated_webhook = store.update_webhook(
        webhook_id=webhook.webhook_id,
        name="updated_name",
        url="https://example.com/updated",
        events=new_events,
        description="Updated description",
        secret="new_secret",
        status=WebhookStatus.DISABLED,
    )
    assert updated_webhook.webhook_id == webhook.webhook_id
    assert updated_webhook.name == "updated_name"
    assert updated_webhook.url == "https://example.com/updated"
    assert updated_webhook.events == new_events
    assert updated_webhook.description == "Updated description"
    assert updated_webhook.status == WebhookStatus.DISABLED
    assert updated_webhook.last_updated_timestamp > webhook.last_updated_timestamp


def test_update_webhook_partial(store: RestStore):
    events = [WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)]
    webhook = store.create_webhook(
        name="original_name",
        url="https://example.com/original",
        events=events,
        description="Original description",
    )
    # Update only the name
    updated_webhook = store.update_webhook(
        webhook_id=webhook.webhook_id,
        name="updated_name",
    )
    assert updated_webhook.name == "updated_name"
    assert updated_webhook.url == "https://example.com/original"
    assert updated_webhook.events == events
    assert updated_webhook.description == "Original description"


def test_update_webhook_not_found(store: RestStore):
    with pytest.raises(MlflowException, match="Webhook with ID nonexistent not found"):
        store.update_webhook(webhook_id="nonexistent", name="new_name")


@pytest.mark.parametrize(
    ("invalid_url", "expected_match"),
    [
        ("   ", r"Webhook URL cannot be empty or just whitespace"),
        ("ftp://example.com", r"Invalid webhook URL scheme"),
        ("http://[invalid", r"Invalid webhook URL"),
    ],
)
def test_update_webhook_invalid_urls(store, invalid_url, expected_match):
    # Create a valid webhook first
    webhook = store.create_webhook(
        name="test_webhook",
        url="https://example.com/webhook",
        events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
    )
    with pytest.raises(MlflowException, match=expected_match):
        store.update_webhook(webhook_id=webhook.webhook_id, url=invalid_url)


def test_delete_webhook(store: RestStore):
    events = [WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)]
    webhook = store.create_webhook(
        name="test_webhook",
        url="https://example.com/webhook",
        events=events,
    )
    store.delete_webhook(webhook.webhook_id)
    with pytest.raises(MlflowException, match=r"Webhook with ID .* not found"):
        store.get_webhook(webhook.webhook_id)
    webhooks_page = store.list_webhooks()
    webhook_ids = {w.webhook_id for w in webhooks_page}
    assert webhook.webhook_id not in webhook_ids


def test_delete_webhook_not_found(store: RestStore):
    with pytest.raises(MlflowException, match="Webhook with ID nonexistent not found"):
        store.delete_webhook("nonexistent")


def test_webhook_experiments_round_trip_and_clear(store: RestStore):
    event = WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED)
    webhook = store.create_webhook(
        name="issue_webhook",
        url="https://example.com/webhook",
        events=[event],
        experiment_ids=["7", "12"],
    )
    assert webhook.experiment_ids == ["12", "7"]
    assert store.get_webhook(webhook.webhook_id).experiment_ids == ["12", "7"]

    assert store.update_webhook(webhook.webhook_id, name="renamed").experiment_ids == ["12", "7"]
    # An empty list has to reach the server to remove the limit, though protobuf JSON drops it.
    assert store.update_webhook(webhook.webhook_id, experiment_ids=[]).experiment_ids == []


def test_create_webhook_rejects_experiments_for_registry_events(store: RestStore):
    with pytest.raises(MlflowException, match="can be limited to experiments"):
        store.create_webhook(
            name="model_webhook",
            url="https://example.com/webhook",
            events=[WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)],
            experiment_ids=["7"],
        )


def test_webhook_destination_type_round_trips(store: RestStore):
    webhook = store.create_webhook(
        name="slack_webhook",
        url="https://hooks.slack.com/services/T/B/X",
        events=[WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED)],
        destination_type=WebhookDestinationType.SLACK,
    )
    assert webhook.destination_type == WebhookDestinationType.SLACK

    updated = store.update_webhook(
        webhook.webhook_id, destination_type=WebhookDestinationType.GENERIC
    )
    assert updated.destination_type == WebhookDestinationType.GENERIC


def test_jira_webhook_round_trips(store: RestStore):
    config = JiraDestinationConfig(project_key="OPS", issue_type="Bug", user_email="bot@x.com")
    webhook = store.create_webhook(
        name="jira_webhook",
        url="https://example.atlassian.net",
        events=[WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED)],
        secret="api-token",
        destination_type=WebhookDestinationType.JIRA,
        jira_config=config,
    )
    assert webhook.jira_config == config
    # The token is the webhook's secret, which no response carries.
    assert webhook.secret is None

    new_config = JiraDestinationConfig(project_key="SUP", issue_type="Task")
    assert store.update_webhook(webhook.webhook_id, jira_config=new_config).jira_config == (
        new_config
    )
