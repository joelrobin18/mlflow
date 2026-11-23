import base64
from unittest import mock

import pytest

from mlflow.entities.issue import Issue, IssueStatus
from mlflow.entities.webhook import (
    JiraDestinationConfig,
    Webhook,
    WebhookAction,
    WebhookDestinationType,
    WebhookEntity,
    WebhookEvent,
)
from mlflow.environment_variables import MLFLOW_WEBHOOK_LINK_BASE_URL
from mlflow.webhooks.jira import (
    JIRA_TICKET_LABEL,
    check_jira_destination,
    file_jira_ticket,
    jira_ticket_fields,
    jira_ticket_metadata_key,
)
from mlflow.webhooks.types import IssueCreatedPayload

ISSUE_CREATED = WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED)
CONFIG = JiraDestinationConfig(project_key="OPS", issue_type="Bug", user_email="bot@example.com")


def _jira_webhook(config=CONFIG, secret="api-token"):
    return Webhook(
        webhook_id="wh-jira",
        name="jira",
        url="https://example.atlassian.net/",
        events=[ISSUE_CREATED],
        creation_timestamp=0,
        last_updated_timestamp=0,
        secret=secret,
        destination_type=WebhookDestinationType.JIRA,
        jira_config=config,
    )


class _IssueStore:
    """An in-memory stand-in for the tracking store, which owns the issue a ticket is filed for."""

    def __init__(self, metadata=None):
        self.metadata = dict(metadata or {})

    def get_issue(self, issue_id):
        return Issue(
            issue_id=issue_id,
            experiment_id="1",
            name="n",
            description="d",
            status=IssueStatus.PENDING,
            created_timestamp=0,
            last_updated_timestamp=0,
            metadata=self.metadata,
        )

    def update_issue(self, issue_id, *, metadata):
        self.metadata |= metadata


@pytest.fixture(autouse=True)
def no_link_base(monkeypatch):
    monkeypatch.delenv(MLFLOW_WEBHOOK_LINK_BASE_URL.name, raising=False)


@pytest.fixture
def store():
    store = _IssueStore()
    with mock.patch("mlflow.webhooks.jira._get_tracking_store", return_value=store):
        yield store


@pytest.fixture
def session():
    session = mock.Mock()
    session.post.return_value = mock.Mock(status_code=201, json=lambda: {"key": "OPS-7"})
    return session


def test_jira_ticket_fields_describe_the_issue_without_trace_text():
    fields = jira_ticket_fields(CONFIG, dict(IssueCreatedPayload.example()))

    assert fields["project"] == {"key": "OPS"}
    assert fields["issuetype"] == {"name": "Bug"}
    assert fields["summary"] == "[MLflow] Tool call omits a required argument"
    assert fields["labels"] == [JIRA_TICKET_LABEL]
    assert fields["description"] == (
        "MLflow found an issue in experiment 1.\n"
        "\n"
        "Issue: Tool call omits a required argument\n"
        "Issue ID: iss-abc123\n"
        "Severity: high\n"
        "Categories: tool_use\n"
        "Detection run: abcd1234abcd5678"
    )


def test_jira_ticket_fields_link_back_to_mlflow(monkeypatch):
    monkeypatch.setenv(MLFLOW_WEBHOOK_LINK_BASE_URL.name, "https://mlflow.example.com")

    description = jira_ticket_fields(CONFIG, dict(IssueCreatedPayload.example()))["description"]

    assert description.endswith(
        "Open the issue in MLflow: https://mlflow.example.com/#/experiments/1/issues"
        "?selectedIssueId=iss-abc123\n"
        "Detection run history: https://mlflow.example.com/#/experiments/1/issues/runs"
    )


def test_jira_ticket_summary_stays_within_jiras_limit():
    payload = dict(IssueCreatedPayload.example()) | {"name": "x" * 400}

    assert len(jira_ticket_fields(CONFIG, payload)["summary"]) == 255


def test_file_jira_ticket_files_and_records_the_ticket(store, session):
    payload = IssueCreatedPayload.example()

    with mock.patch("mlflow.webhooks.jira._validate_webhook_url"):
        file_jira_ticket(_jira_webhook(), ISSUE_CREATED, payload, session)

    (call,) = session.post.call_args_list
    assert call.args == ("https://example.atlassian.net/rest/api/2/issue",)
    assert call.kwargs["json"] == {"fields": jira_ticket_fields(CONFIG, dict(payload))}
    expected = base64.b64encode(b"bot@example.com:api-token").decode()
    assert call.kwargs["headers"] == {"Authorization": f"Basic {expected}"}
    assert store.metadata == {jira_ticket_metadata_key(_jira_webhook()): "OPS-7"}


def test_file_jira_ticket_sends_a_bearer_token_without_an_account_email(store, session):
    webhook = _jira_webhook(config=JiraDestinationConfig(project_key="OPS", issue_type="Task"))

    with mock.patch("mlflow.webhooks.jira._validate_webhook_url"):
        file_jira_ticket(webhook, ISSUE_CREATED, IssueCreatedPayload.example(), session)

    assert session.post.call_args.kwargs["headers"] == {"Authorization": "Bearer api-token"}


def test_file_jira_ticket_does_not_file_a_second_ticket_for_the_same_issue(session):
    store = _IssueStore({jira_ticket_metadata_key(_jira_webhook()): "OPS-1"})

    with (
        mock.patch("mlflow.webhooks.jira._get_tracking_store", return_value=store),
        mock.patch("mlflow.webhooks.jira._validate_webhook_url"),
    ):
        response = file_jira_ticket(
            _jira_webhook(), ISSUE_CREATED, IssueCreatedPayload.example(), session
        )

    assert response is None
    session.post.assert_not_called()


def test_file_jira_ticket_leaves_a_rejected_ticket_unrecorded(store, session):
    session.post.return_value = mock.Mock(status_code=400, text='{"errors":{"issuetype":"bad"}}')

    with mock.patch("mlflow.webhooks.jira._validate_webhook_url"):
        response = file_jira_ticket(
            _jira_webhook(), ISSUE_CREATED, IssueCreatedPayload.example(), session
        )

    assert response.status_code == 400
    assert store.metadata == {}


def test_file_jira_ticket_files_nothing_for_other_events(store, session):
    event = WebhookEvent(WebhookEntity.ISSUE_DETECTION_RUN, WebhookAction.COMPLETED)

    assert file_jira_ticket(_jira_webhook(), event, {"experiment_id": "1"}, session) is None
    session.post.assert_not_called()


@pytest.mark.parametrize(("status_code", "success"), [(200, True), (404, False), (401, False)])
def test_check_jira_destination_looks_up_the_project(status_code, success):
    session = mock.Mock()
    session.get.return_value = mock.Mock(status_code=status_code, text="body")

    with mock.patch("mlflow.webhooks.jira._validate_webhook_url"):
        result = check_jira_destination(_jira_webhook(), session)

    assert session.get.call_args.args == ("https://example.atlassian.net/rest/api/2/project/OPS",)
    assert result.success is success
    assert result.response_status == status_code
    session.post.assert_not_called()


def test_check_jira_destination_reports_an_unreachable_site():
    session = mock.Mock()
    session.get.side_effect = ConnectionError("no route to host")

    with mock.patch("mlflow.webhooks.jira._validate_webhook_url"):
        result = check_jira_destination(_jira_webhook(), session)

    assert result.success is False
    assert "no route to host" in result.error_message
