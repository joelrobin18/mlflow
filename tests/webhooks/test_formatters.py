import pytest

from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.environment_variables import MLFLOW_WEBHOOK_LINK_BASE_URL
from mlflow.webhooks.formatters import detection_runs_url, format_slack_message, issue_url
from mlflow.webhooks.types import IssueCreatedPayload, IssueDetectionRunCompletedPayload

ISSUE_CREATED = WebhookEvent(WebhookEntity.ISSUE, WebhookAction.CREATED)
RUN_COMPLETED = WebhookEvent(WebhookEntity.ISSUE_DETECTION_RUN, WebhookAction.COMPLETED)


def _section_text(message):
    return message["blocks"][0]["text"]["text"]


def _context(message):
    return [element["text"] for element in message["blocks"][1]["elements"]]


@pytest.fixture
def link_base(monkeypatch):
    monkeypatch.setenv(MLFLOW_WEBHOOK_LINK_BASE_URL.name, "https://mlflow.example.com/")


def test_ui_links_need_a_base_url(monkeypatch):
    monkeypatch.delenv(MLFLOW_WEBHOOK_LINK_BASE_URL.name, raising=False)

    assert issue_url("7", "iss-1") is None
    assert detection_runs_url("7") is None


def test_ui_links_point_at_the_issues_tab(link_base):
    assert (
        issue_url("7", "iss/1")
        == "https://mlflow.example.com/#/experiments/7/issues?selectedIssueId=iss%2F1"
    )
    assert detection_runs_url("7") == "https://mlflow.example.com/#/experiments/7/issues/runs"


def test_issue_created_message_links_to_the_issue(link_base):
    message = format_slack_message(ISSUE_CREATED, IssueCreatedPayload.example())

    assert message["text"] == (
        "New high severity issue in experiment 1: Tool call omits a required argument"
    )
    assert _section_text(message) == (
        "*New high severity issue*: <https://mlflow.example.com/#/experiments/1/issues"
        "?selectedIssueId=iss-abc123|Tool call omits a required argument>"
    )
    assert _context(message) == ["Experiment 1", "Categories: tool_use", "Status: pending"]


def test_issue_created_message_without_a_base_url_names_the_issue(monkeypatch):
    monkeypatch.delenv(MLFLOW_WEBHOOK_LINK_BASE_URL.name, raising=False)
    payload = IssueCreatedPayload.example() | {"severity": None, "categories": []}

    message = format_slack_message(ISSUE_CREATED, payload)

    assert _section_text(message) == "*New issue*: Tool call omits a required argument"
    assert _context(message) == ["Experiment 1", "Status: pending"]


def test_issue_created_message_escapes_slack_markup(monkeypatch):
    monkeypatch.delenv(MLFLOW_WEBHOOK_LINK_BASE_URL.name, raising=False)
    payload = IssueCreatedPayload.example() | {"name": "Ping <!channel> & <https://evil|here>"}

    message = format_slack_message(ISSUE_CREATED, payload)

    assert "<!channel>" not in message["text"]
    assert _section_text(message).endswith("Ping &lt;!channel&gt; &amp; &lt;https://evil|here&gt;")


def test_run_completed_message_summarizes_a_successful_run(link_base):
    message = format_slack_message(RUN_COMPLETED, IssueDetectionRunCompletedPayload.example())

    assert message["text"] == "Scheduled issue detection finished in experiment 1"
    assert _section_text(message) == (
        "*Scheduled issue detection finished in experiment 1*\n"
        "1 new of 3 issues found across 100 traces. See the "
        "<https://mlflow.example.com/#/experiments/1/issues/runs|run history>."
    )
    assert _context(message) == ["Model: openai:/gpt-4o", "Cost: $0.42"]


def test_run_completed_message_quotes_why_a_run_failed(monkeypatch):
    monkeypatch.delenv(MLFLOW_WEBHOOK_LINK_BASE_URL.name, raising=False)
    payload = IssueDetectionRunCompletedPayload.example() | {
        "status": "failed",
        "trigger": "manual",
        "error_message": "MlflowException: " + "x" * 600,
        "total_cost_usd": None,
    }

    message = format_slack_message(RUN_COMPLETED, payload)

    assert message["text"] == "Manual issue detection failed in experiment 1"
    lines = _section_text(message).split("\n")
    assert lines[0] == "*Manual issue detection failed in experiment 1*"
    assert lines[1].startswith("`MlflowException: xxx")
    assert len(lines[1]) <= 502
    assert lines[2] == "See the run history."
    assert _context(message) == ["Model: openai:/gpt-4o"]


def test_other_events_are_shown_as_their_payload():
    event = WebhookEvent(WebhookEntity.PROMPT, WebhookAction.CREATED)

    message = format_slack_message(event, {"name": "my-prompt", "tags": {}, "description": None})

    assert message["text"] == "MLflow event: prompt.created"
    assert _section_text(message).startswith("*MLflow event:* `prompt.created`\n```{")
    assert '"name": "my-prompt"' in _section_text(message)
    assert len(message["blocks"]) == 1


def test_a_large_payload_stays_within_slacks_section_limit():
    event = WebhookEvent(WebhookEntity.PROMPT, WebhookAction.CREATED)

    message = format_slack_message(event, {"description": "y" * 10_000})

    assert len(_section_text(message)) <= 3000
