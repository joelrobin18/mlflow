import { useState } from 'react';
import { Button, Modal, NotificationIcon, Tooltip } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import WebhooksSettings from '../../../settings/WebhooksSettings';

const COMPONENT_ID = 'mlflow.experiment-issues.notifications';

/** The events that tell someone about an experiment's issues. */
const ISSUE_EVENT_ENTITIES = ['ISSUE', 'ISSUE_DETECTION_RUN'];

export interface IssueNotificationsControlProps {
  experimentId: string;
}

/**
 * Where the people who own an experiment set up who hears about its issues: the webhooks limited to
 * this experiment, which can post to Slack, file Jira tickets, or call an HTTP endpoint.
 */
export const IssueNotificationsControl = ({ experimentId }: IssueNotificationsControlProps) => {
  const intl = useIntl();
  const [isOpen, setIsOpen] = useState(false);
  const label = intl.formatMessage({
    defaultMessage: 'Issue notifications',
    description: 'Experiment issues tab > Label of the toolbar button opening the issue notification settings',
  });

  return (
    <>
      <Tooltip componentId={`${COMPONENT_ID}.tooltip`} content={label}>
        <Button
          componentId={`${COMPONENT_ID}.open`}
          icon={<NotificationIcon />}
          aria-label={label}
          onClick={() => setIsOpen(true)}
        />
      </Tooltip>
      {/* Mounted only while open, so the list is read fresh each time rather than on every page load. */}
      {isOpen && (
        <Modal
          componentId={`${COMPONENT_ID}.modal`}
          title={label}
          visible
          onCancel={() => setIsOpen(false)}
          footer={null}
          size="wide"
        >
          <WebhooksSettings
            experimentId={experimentId}
            eventFilter={ISSUE_EVENT_ENTITIES}
            showTitle={false}
            description={
              <FormattedMessage
                defaultMessage="Tell your team about new issues and detection runs in this experiment: post to a Slack channel, file a Jira ticket for each new issue, or call your own endpoint. Webhooks for every experiment are managed in Settings."
                description="Experiment issues tab > Description of the issue notification settings"
              />
            }
            emptyDescription={
              <FormattedMessage
                defaultMessage="No notifications are set up for this experiment yet."
                description="Experiment issues tab > Empty state of the issue notification settings"
              />
            }
          />
        </Modal>
      )}
    </>
  );
};
