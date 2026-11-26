import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { BudgetPolicyWindow, ListBudgetWindowsResponse } from '../types';
import { GatewayQueryKeys } from './queryKeys';

const MIN_REFETCH_INTERVAL_MS = 5_000;
const MAX_REFETCH_INTERVAL_MS = 5 * 60_000;
// Gives the server a moment to start the next window before the list asks again.
const WINDOW_END_GRACE_MS = 1_000;
// While the request fails (e.g. the server is restarting), the windows only look ended, so ask
// less often than the minimum instead of every 5 s.
const FAILED_REFETCH_INTERVAL_MS = 30_000;

/**
 * Refetch just after the earliest window ends, so an open page doesn't keep showing a window
 * (and its spend) after it has ended. The bounds keep short windows from flooding the server
 * and keep spend from going stale while every window is long.
 */
export const getBudgetWindowsRefetchInterval = (
  data: ListBudgetWindowsResponse | undefined,
  failed: boolean = false,
  now: number = Date.now(),
): number | false => {
  const windowEnds = (data?.windows ?? []).map((w) => Number(w.window_end_ms));
  if (!windowEnds.length) {
    return false;
  }
  if (failed) {
    return FAILED_REFETCH_INTERVAL_MS;
  }
  const untilFirstEnd = Math.min(...windowEnds) - now + WINDOW_END_GRACE_MS;
  return Math.min(Math.max(untilFirstEnd, MIN_REFETCH_INTERVAL_MS), MAX_REFETCH_INTERVAL_MS);
};

export const useBudgetWindowsQuery = () => {
  const queryResult = useQuery<
    ListBudgetWindowsResponse,
    Error,
    ListBudgetWindowsResponse,
    typeof GatewayQueryKeys.budgetWindows
  >(GatewayQueryKeys.budgetWindows, {
    queryFn: () => GatewayApi.getBudgetWindows(),
    retry: false,
    refetchInterval: (data, query) => getBudgetWindowsRefetchInterval(data, query.state.status === 'error'),
  });

  const windowsByPolicyId: Record<string, BudgetPolicyWindow> = {};
  for (const w of queryResult.data?.windows ?? []) {
    windowsByPolicyId[w.budget_policy_id] = w;
  }

  return {
    data: windowsByPolicyId,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
  };
};
