import { useState } from 'react';
import { useForm, Controller, FormProvider } from 'react-hook-form';
import {
  Alert,
  Checkbox,
  FormUI,
  Modal,
  RHFControlledComponents,
  SimpleSelect,
  SimpleSelectOption,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { WebhooksApi } from './webhooksApi';
import type { Webhook, WebhookEvent } from './webhooksApi';
import {
  DESTINATION_TYPES,
  JIRA_EVENT_KEY,
  VALID_EVENTS,
  WEBHOOK_NAME_REGEX,
  destinationTypeLabels,
  eventKey,
  eventLabels,
  isExperimentScopedEventKey,
  parseExperimentIds,
} from './webhookConstants';
import type { WebhookDestinationType } from './webhookConstants';

interface WebhookFormData {
  name: string;
  url: string;
  description: string;
  secret: string;
  status: boolean;
  events: string[];
  experimentIds: string;
  destinationType: WebhookDestinationType;
  jiraProjectKey: string;
  jiraIssueType: string;
  jiraUserEmail: string;
}

interface WebhookFormModalProps {
  visible: boolean;
  editingWebhook: Webhook | null;
  onClose: () => void;
  onSaved: () => void;
  /** If set, only show events matching this entity */
  eventFilter?: string;
}

const WebhookFormModal = ({ visible, editingWebhook, onClose, onSaved, eventFilter }: WebhookFormModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const filteredEvents = eventFilter ? VALID_EVENTS.filter((e) => e.entity === eventFilter) : VALID_EVENTS;

  // When filtered to a single event (e.g. budget page), pre-select it for new webhooks
  const defaultEvents =
    !editingWebhook && eventFilter && filteredEvents.length === 1
      ? filteredEvents.map((e) => eventKey(e.entity, e.action))
      : (editingWebhook?.events.map((e) => eventKey(e.entity, e.action)) ?? []);

  const form = useForm<WebhookFormData>({
    defaultValues: {
      name: editingWebhook?.name ?? '',
      url: editingWebhook?.url ?? '',
      description: editingWebhook?.description ?? '',
      secret: '',
      status: editingWebhook ? editingWebhook.status === 'ACTIVE' : true,
      events: defaultEvents,
      experimentIds: editingWebhook?.experiment_ids?.join(', ') ?? '',
      destinationType: editingWebhook?.destination_type ?? 'GENERIC',
      jiraProjectKey: editingWebhook?.jira_config?.project_key ?? '',
      jiraIssueType: editingWebhook?.jira_config?.issue_type ?? 'Bug',
      jiraUserEmail: editingWebhook?.jira_config?.user_email ?? '',
    },
  });

  // A Jira webhook files one ticket per new issue, so it subscribes to that event alone.
  const isJira = form.watch('destinationType') === 'JIRA';
  const displayedEvents = isJira
    ? filteredEvents.filter((e) => eventKey(e.entity, e.action) === JIRA_EVENT_KEY)
    : filteredEvents;

  // Only a webhook whose events all happen in an experiment can be limited to experiments: any other
  // event carries no experiment to match, so the server rejects the combination.
  const selectedEvents = form.watch('events');
  const canLimitToExperiments = selectedEvents.length > 0 && selectedEvents.every(isExperimentScopedEventKey);

  const formatEventLabel = (entity: string, action: string) => {
    const key = eventKey(entity, action);
    const descriptor = eventLabels[key as keyof typeof eventLabels];
    return descriptor ? intl.formatMessage(descriptor) : key;
  };

  const handleSubmit = async (values: WebhookFormData) => {
    const events: WebhookEvent[] = values.events.map((key) => {
      const [entity, action] = key.split('.', 2);
      return { entity, action };
    });

    setIsSaving(true);
    setSubmitError(null);

    const payload = {
      name: values.name.trim(),
      url: values.url.trim(),
      events,
      description: values.description.trim() || undefined,
      secret: values.secret.trim() || undefined,
      status: values.status ? ('ACTIVE' as const) : ('DISABLED' as const),
      // Sent even when empty, so editing a limited webhook to drop its experiments, or to add an event
      // that cannot be limited, removes the limit instead of keeping it.
      experiment_ids: canLimitToExperiments ? parseExperimentIds(values.experimentIds) : [],
      destination_type: values.destinationType,
      jira_config:
        values.destinationType === 'JIRA'
          ? {
              project_key: values.jiraProjectKey.trim(),
              issue_type: values.jiraIssueType.trim(),
              user_email: values.jiraUserEmail.trim() || undefined,
            }
          : undefined,
    };

    try {
      if (editingWebhook) {
        await WebhooksApi.updateWebhook(editingWebhook.webhook_id, payload);
      } else {
        await WebhooksApi.createWebhook(payload);
      }
      onSaved();
    } catch (e: any) {
      setSubmitError(
        e?.message ??
          intl.formatMessage({
            defaultMessage: 'Failed to save webhook',
            description: 'Generic error message informing the user that webhook saving failed',
          }),
      );
    } finally {
      setIsSaving(false);
    }
  };

  const nameValidationMessage = intl.formatMessage({
    defaultMessage:
      'Name must start and end with a letter or digit, be less than 63 characters long, and contain only letters, digits, dots (.), underscores (_), and hyphens (-).',
    description: 'Webhook name validation error',
  });

  return (
    <FormProvider {...form}>
      <Modal
        componentId="mlflow.settings.webhooks.form-modal"
        title={
          editingWebhook
            ? intl.formatMessage({ defaultMessage: 'Edit webhook', description: 'Edit webhook modal title' })
            : intl.formatMessage({ defaultMessage: 'Create webhook', description: 'Create webhook modal title' })
        }
        visible={visible}
        onCancel={onClose}
        onOk={form.handleSubmit(handleSubmit)}
        okText={
          editingWebhook
            ? intl.formatMessage({ defaultMessage: 'Save', description: 'Save webhook button' })
            : intl.formatMessage({ defaultMessage: 'Create', description: 'Create webhook confirm button' })
        }
        cancelText={intl.formatMessage({ defaultMessage: 'Cancel', description: 'Cancel webhook form button' })}
        confirmLoading={isSaving}
        size="wide"
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {submitError && (
            <Alert componentId="mlflow.settings.webhooks.form-error-alert" type="error" message={submitError} />
          )}

          <div>
            <FormUI.Label htmlFor="mlflow.settings.webhooks.name-input">
              <FormattedMessage defaultMessage="Name" description="Webhook name field label" /> *
            </FormUI.Label>
            <RHFControlledComponents.Input
              name="name"
              control={form.control}
              id="mlflow.settings.webhooks.name-input"
              componentId="mlflow.settings.webhooks.name-input"
              placeholder={intl.formatMessage({
                defaultMessage: 'My webhook',
                description: 'Webhook name placeholder',
              })}
              rules={{
                required: intl.formatMessage({
                  defaultMessage: 'Name is required',
                  description: 'Form validation error message informing the user that the webhook name is required',
                }),
                validate: (value) => {
                  const trimmed = String(value).trim();
                  if (trimmed.length > 63 || !WEBHOOK_NAME_REGEX.test(trimmed)) {
                    return nameValidationMessage;
                  }
                  return true;
                },
              }}
              validationState={form.formState.errors.name ? 'error' : undefined}
            />
            {form.formState.errors.name && <FormUI.Message type="error" message={form.formState.errors.name.message} />}
          </div>

          <div>
            <FormUI.Label htmlFor="mlflow.settings.webhooks.url-input">
              <FormattedMessage defaultMessage="URL" description="Webhook URL field label" /> *
            </FormUI.Label>
            <RHFControlledComponents.Input
              name="url"
              control={form.control}
              id="mlflow.settings.webhooks.url-input"
              componentId="mlflow.settings.webhooks.url-input"
              placeholder={intl.formatMessage({
                defaultMessage: 'https://example.com/webhook',
                description: 'Webhook URL placeholder',
              })}
              rules={{
                required: intl.formatMessage({
                  defaultMessage: 'URL is required',
                  description: 'Form validation error message informing the user that the webhook URL is required',
                }),
              }}
              validationState={form.formState.errors.url ? 'error' : undefined}
            />
            {form.formState.errors.url && <FormUI.Message type="error" message={form.formState.errors.url.message} />}
          </div>

          <div>
            <FormUI.Label htmlFor="mlflow.settings.webhooks.destination-type-select">
              <FormattedMessage defaultMessage="Destination" description="Webhook destination type field label" />
            </FormUI.Label>
            <FormUI.Hint>
              <FormattedMessage
                defaultMessage="A Slack incoming webhook gets a formatted Slack message. An HTTP endpoint gets the event as signed JSON."
                description="Webhook destination type field description"
              />
            </FormUI.Hint>
            <Controller
              name="destinationType"
              control={form.control}
              render={({ field }) => (
                <SimpleSelect
                  id="mlflow.settings.webhooks.destination-type-select"
                  componentId="mlflow.settings.webhooks.destination-type-select"
                  value={field.value}
                  onChange={({ target }) => {
                    const destinationType = target.value as WebhookDestinationType;
                    field.onChange(destinationType);
                    if (destinationType === 'JIRA') {
                      form.setValue('events', [JIRA_EVENT_KEY]);
                    }
                  }}
                >
                  {DESTINATION_TYPES.map((destinationType) => (
                    <SimpleSelectOption key={destinationType} value={destinationType}>
                      {intl.formatMessage(destinationTypeLabels[destinationType])}
                    </SimpleSelectOption>
                  ))}
                </SimpleSelect>
              )}
            />
          </div>

          {isJira && (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage="Enter your Jira site as the URL, such as https://example.atlassian.net, and the API token or personal access token as the secret."
                  description="Explains which webhook fields a Jira destination uses"
                />
              </FormUI.Hint>
              <div>
                <FormUI.Label htmlFor="mlflow.settings.webhooks.jira-project-input">
                  <FormattedMessage defaultMessage="Jira project key" description="Jira project key field label" /> *
                </FormUI.Label>
                <RHFControlledComponents.Input
                  name="jiraProjectKey"
                  control={form.control}
                  id="mlflow.settings.webhooks.jira-project-input"
                  componentId="mlflow.settings.webhooks.jira-project-input"
                  placeholder="OPS"
                  rules={{
                    validate: (value) =>
                      !isJira ||
                      String(value).trim().length > 0 ||
                      intl.formatMessage({
                        defaultMessage: 'Jira project key is required',
                        description: 'Validation error for a missing Jira project key',
                      }),
                  }}
                  validationState={form.formState.errors.jiraProjectKey ? 'error' : undefined}
                />
                {form.formState.errors.jiraProjectKey && (
                  <FormUI.Message type="error" message={form.formState.errors.jiraProjectKey.message} />
                )}
              </div>
              <div>
                <FormUI.Label htmlFor="mlflow.settings.webhooks.jira-issue-type-input">
                  <FormattedMessage defaultMessage="Jira issue type" description="Jira issue type field label" /> *
                </FormUI.Label>
                <RHFControlledComponents.Input
                  name="jiraIssueType"
                  control={form.control}
                  id="mlflow.settings.webhooks.jira-issue-type-input"
                  componentId="mlflow.settings.webhooks.jira-issue-type-input"
                  placeholder="Bug"
                  rules={{
                    validate: (value) =>
                      !isJira ||
                      String(value).trim().length > 0 ||
                      intl.formatMessage({
                        defaultMessage: 'Jira issue type is required',
                        description: 'Validation error for a missing Jira issue type',
                      }),
                  }}
                  validationState={form.formState.errors.jiraIssueType ? 'error' : undefined}
                />
                {form.formState.errors.jiraIssueType && (
                  <FormUI.Message type="error" message={form.formState.errors.jiraIssueType.message} />
                )}
              </div>
              <div>
                <FormUI.Label htmlFor="mlflow.settings.webhooks.jira-email-input">
                  <FormattedMessage defaultMessage="Jira account email" description="Jira account email field label" />
                </FormUI.Label>
                <FormUI.Hint>
                  <FormattedMessage
                    defaultMessage="For Jira Cloud, the account the API token belongs to. Leave blank for a Jira Data Center personal access token."
                    description="Jira account email field description"
                  />
                </FormUI.Hint>
                <RHFControlledComponents.Input
                  name="jiraUserEmail"
                  control={form.control}
                  id="mlflow.settings.webhooks.jira-email-input"
                  componentId="mlflow.settings.webhooks.jira-email-input"
                />
              </div>
            </div>
          )}

          <div>
            <FormUI.Label htmlFor="mlflow.settings.webhooks.description-input">
              <FormattedMessage defaultMessage="Description" description="Webhook description field label" />
            </FormUI.Label>
            <RHFControlledComponents.TextArea
              name="description"
              control={form.control}
              id="mlflow.settings.webhooks.description-input"
              componentId="mlflow.settings.webhooks.description-input"
              placeholder={intl.formatMessage({
                defaultMessage: 'Optional description',
                description: 'Webhook description placeholder',
              })}
              rows={2}
            />
          </div>

          <div>
            <FormUI.Label htmlFor="mlflow.settings.webhooks.secret-input">
              <FormattedMessage defaultMessage="Secret" description="Webhook secret field label" />
            </FormUI.Label>
            <FormUI.Hint>
              <FormattedMessage
                defaultMessage="Used for HMAC signature verification of incoming webhook requests."
                description="Webhook secret field description"
              />
            </FormUI.Hint>
            <RHFControlledComponents.Password
              name="secret"
              control={form.control}
              id="mlflow.settings.webhooks.secret-input"
              placeholder={
                editingWebhook
                  ? intl.formatMessage({
                      defaultMessage: 'Leave blank to keep existing secret',
                      description: 'Webhook secret placeholder when editing',
                    })
                  : intl.formatMessage({
                      defaultMessage: 'Optional secret key',
                      description: 'Webhook secret placeholder when creating',
                    })
              }
            />
          </div>

          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <FormUI.Label htmlFor="mlflow.settings.webhooks.status-switch">
              <FormattedMessage defaultMessage="Active" description="Webhook status field label" />
            </FormUI.Label>
            <RHFControlledComponents.Switch
              name="status"
              control={form.control}
              id="mlflow.settings.webhooks.status-switch"
              componentId="mlflow.settings.webhooks.status-switch"
              activeLabel={intl.formatMessage({ defaultMessage: 'Active', description: 'Webhook active label' })}
              inactiveLabel={intl.formatMessage({ defaultMessage: 'Disabled', description: 'Webhook disabled label' })}
            />
          </div>

          <div>
            <FormUI.Label>
              <FormattedMessage defaultMessage="Events" description="Webhook events field label" /> *
            </FormUI.Label>
            <FormUI.Hint>
              <FormattedMessage
                defaultMessage="Select the events that will trigger this webhook."
                description="Webhook events field description"
              />
            </FormUI.Hint>
            <Controller
              name="events"
              control={form.control}
              rules={{
                validate: (value: string[]) =>
                  value.length > 0 ||
                  intl.formatMessage({
                    defaultMessage: 'At least one event must be selected',
                    description:
                      'Form validation error message informing the user that at least one event must be selected',
                  }),
              }}
              render={({ field }) => (
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  {displayedEvents.map((event) => {
                    const key = eventKey(event.entity, event.action);
                    return (
                      <Checkbox
                        key={key}
                        componentId="mlflow.settings.webhooks.event-checkbox"
                        isChecked={field.value.includes(key)}
                        onChange={(checked) => {
                          const next = checked ? [...field.value, key] : field.value.filter((k: string) => k !== key);
                          field.onChange(next);
                        }}
                      >
                        {formatEventLabel(event.entity, event.action)}
                      </Checkbox>
                    );
                  })}
                </div>
              )}
            />
            {form.formState.errors.events && (
              <FormUI.Message type="error" message={form.formState.errors.events.message} />
            )}
          </div>

          {canLimitToExperiments && (
            <div>
              <FormUI.Label htmlFor="mlflow.settings.webhooks.experiment-ids-input">
                <FormattedMessage defaultMessage="Experiments" description="Webhook experiment IDs field label" />
              </FormUI.Label>
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage="IDs of the experiments to send events from, separated by commas. Leave blank to send events from every experiment."
                  description="Webhook experiment IDs field description"
                />
              </FormUI.Hint>
              <RHFControlledComponents.Input
                name="experimentIds"
                control={form.control}
                id="mlflow.settings.webhooks.experiment-ids-input"
                componentId="mlflow.settings.webhooks.experiment-ids-input"
                placeholder={intl.formatMessage({
                  defaultMessage: 'All experiments',
                  description: 'Webhook experiment IDs placeholder',
                })}
              />
            </div>
          )}
        </div>
      </Modal>
    </FormProvider>
  );
};

export default WebhookFormModal;
