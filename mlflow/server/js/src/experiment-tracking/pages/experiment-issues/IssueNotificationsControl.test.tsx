import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { PermissionError } from '@databricks/web-shared/errors';
import { renderWithIntl } from '../../../common/utils/TestUtils.react18';
import { WebhooksApi } from '../../../settings/webhooksApi';
import type { Webhook } from '../../../settings/webhooksApi';
import { IssueNotificationsControl } from './IssueNotificationsControl';

jest.mock('../../../settings/webhooksApi', () => ({
  WebhooksApi: {
    listWebhooks: jest.fn(() => Promise.resolve({ webhooks: [] })),
    createWebhook: jest.fn(() => Promise.resolve({})),
    updateWebhook: jest.fn(() => Promise.resolve({})),
    deleteWebhook: jest.fn(() => Promise.resolve()),
    testWebhook: jest.fn(() => Promise.resolve({ result: { success: true, response_status: 200 } })),
  },
}));

const mockListWebhooks = jest.mocked(WebhooksApi.listWebhooks);
const mockCreateWebhook = jest.mocked(WebhooksApi.createWebhook);

const webhook = (name: string, overrides: Partial<Webhook>): Webhook => ({
  webhook_id: name,
  name,
  url: `https://example.com/${name}`,
  events: [{ entity: 'ISSUE', action: 'CREATED' }],
  status: 'ACTIVE',
  creation_timestamp: '1700000000000',
  last_updated_timestamp: '1700000000000',
  ...overrides,
});

describe('IssueNotificationsControl', () => {
  const renderControl = () =>
    renderWithIntl(
      <DesignSystemProvider>
        <IssueNotificationsControl experimentId="7" />
      </DesignSystemProvider>,
    );

  const openNotifications = async () => {
    await userEvent.click(screen.getByRole('button', { name: 'Issue notifications' }));
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockListWebhooks.mockResolvedValue({ webhooks: [] });
  });

  it('reads the webhooks only once it is opened', async () => {
    renderControl();

    expect(mockListWebhooks).not.toHaveBeenCalled();
    await openNotifications();

    await waitFor(() => {
      expect(screen.getByText('No notifications are set up for this experiment yet.')).toBeInTheDocument();
    });
    expect(mockListWebhooks).toHaveBeenCalledTimes(1);
  });

  it("lists only this experiment's issue webhooks", async () => {
    mockListWebhooks.mockResolvedValue({
      webhooks: [
        webhook('this-experiment', { experiment_ids: ['7'], destination_type: 'SLACK' }),
        webhook('another-experiment', { experiment_ids: ['8'] }),
        webhook('every-experiment', {}),
        webhook('prompt-events', {
          events: [{ entity: 'PROMPT', action: 'CREATED' }],
          experiment_ids: ['7'],
        }),
      ],
    });

    renderControl();
    await openNotifications();

    await waitFor(() => {
      expect(screen.getByText('this-experiment')).toBeInTheDocument();
    });
    expect(screen.queryByText('another-experiment')).not.toBeInTheDocument();
    expect(screen.queryByText('every-experiment')).not.toBeInTheDocument();
    expect(screen.queryByText('prompt-events')).not.toBeInTheDocument();
  });

  it('creates a webhook limited to this experiment for its issue events', async () => {
    renderControl();
    await openNotifications();
    await waitFor(() => {
      expect(screen.getByText('Create webhook')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Create webhook'));
    await userEvent.type(screen.getByPlaceholderText('My webhook'), 'team-issues');
    await userEvent.type(screen.getByPlaceholderText('https://example.com/webhook'), 'https://example.com/hook');

    // Only issue events are offered, the first pre-selected, and the experiment is fixed.
    expect(screen.queryByText('Prompt created')).not.toBeInTheDocument();
    expect(screen.getByText('Issue detection run completed')).toBeInTheDocument();
    expect(screen.queryByPlaceholderText('All experiments')).not.toBeInTheDocument();
    const createButtons = screen.getAllByText('Create');
    await userEvent.click(createButtons[createButtons.length - 1]);

    await waitFor(() => {
      expect(mockCreateWebhook).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'team-issues',
          events: [{ entity: 'ISSUE', action: 'CREATED' }],
          experiment_ids: ['7'],
        }),
      );
    });
  });

  it('explains when only admins can manage notifications', async () => {
    mockListWebhooks.mockRejectedValue(new PermissionError({}));

    renderControl();
    await openNotifications();

    await waitFor(() => {
      expect(screen.getByText(/Only admins can manage notifications on this server/)).toBeInTheDocument();
    });
  });
});
