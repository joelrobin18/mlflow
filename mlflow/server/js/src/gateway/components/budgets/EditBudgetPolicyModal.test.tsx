import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { EditBudgetPolicyModal } from './EditBudgetPolicyModal';
import { useUpdateBudgetPolicy } from '../../hooks/useUpdateBudgetPolicy';
import type { BudgetPolicy } from '../../types';

jest.mock('../../hooks/useUpdateBudgetPolicy');
jest.mock('../../../experiment-tracking/hooks/useServerInfo', () => ({
  getWorkspacesEnabledSync: () => false,
}));

const mockMutateAsync = jest.fn().mockReturnValue(Promise.resolve());

const mockPolicy: BudgetPolicy = {
  budget_policy_id: 'bp-1',
  budget_unit: 'USD',
  budget_amount: 200,
  duration: { unit: 'WEEKS', value: 1 },
  target_scope: 'GLOBAL',
  budget_action: 'ALERT',
  created_at: Date.now() / 1000,
  last_updated_at: Date.now() / 1000,
};

const mockUserPolicy: BudgetPolicy = {
  ...mockPolicy,
  budget_policy_id: 'bp-user',
  target_scope: 'USER',
  budget_action: 'REJECT',
  principal: 'alice',
};

describe('EditBudgetPolicyModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useUpdateBudgetPolicy).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);
  });

  test('renders nothing when policy is null', () => {
    const { container } = renderWithDesignSystem(<EditBudgetPolicyModal open policy={null} onClose={jest.fn()} />);

    expect(container.innerHTML).toBe('');
  });

  test('renders form with policy data populated', () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockPolicy} onClose={jest.fn()} />);

    expect(screen.getByText('Edit Budget Policy')).toBeInTheDocument();
    expect(screen.getByText('Budget amount (USD)')).toBeInTheDocument();
    expect(screen.getByDisplayValue('200')).toBeInTheDocument();
  });

  test('submits with correct payload mapping duration preset', async () => {
    const onClose = jest.fn();
    const onSuccess = jest.fn();

    mockMutateAsync.mockReturnValue(Promise.resolve());

    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockPolicy} onClose={onClose} onSuccess={onSuccess} />);

    const saveButton = screen.getByRole('button', { name: 'Save Changes' });
    await userEvent.click(saveButton);

    expect(mockMutateAsync).toHaveBeenCalledWith({
      budget_policy_id: 'bp-1',
      budget_unit: 'USD',
      budget_amount: 200,
      duration: { unit: 'WEEKS', value: 1 },
      target_scope: 'GLOBAL',
      budget_action: 'ALERT',
    });
  });

  test('renders principal field populated for a USER-scoped policy', () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockUserPolicy} onClose={jest.fn()} />);

    expect(screen.getByText('Applies to user')).toBeInTheDocument();
    expect(screen.getByDisplayValue('alice')).toBeInTheDocument();
  });

  test('does not render principal field for a non-USER policy', () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockPolicy} onClose={jest.fn()} />);

    expect(screen.queryByText('Applies to user')).not.toBeInTheDocument();
  });

  test('preserves USER scope and submits edited principal', async () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockUserPolicy} onClose={jest.fn()} />);

    const principalInput = screen.getByDisplayValue('alice');
    await userEvent.clear(principalInput);
    await userEvent.type(principalInput, 'bob');

    await userEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    expect(mockMutateAsync).toHaveBeenCalledWith({
      budget_policy_id: 'bp-user',
      budget_unit: 'USD',
      budget_amount: 200,
      duration: { unit: 'WEEKS', value: 1 },
      target_scope: 'USER',
      budget_action: 'REJECT',
      principal: 'bob',
    });
  });

  test('disables save when the principal of a USER policy is cleared', async () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockUserPolicy} onClose={jest.fn()} />);

    await userEvent.clear(screen.getByDisplayValue('alice'));

    expect(screen.getByRole('button', { name: 'Save Changes' })).toBeDisabled();
    expect(mockMutateAsync).not.toHaveBeenCalled();
  });

  test('displays error message on mutation failure', () => {
    jest.mocked(useUpdateBudgetPolicy).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isLoading: false,
      error: new Error('Update failed'),
      reset: jest.fn(),
    } as any);

    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockPolicy} onClose={jest.fn()} />);

    expect(screen.getByText('Update failed')).toBeInTheDocument();
  });
});
