import { describe, test, expect } from '@jest/globals';
import { getBudgetWindowsRefetchInterval } from './useBudgetWindowsQuery';

const NOW = Date.parse('2026-03-01T08:00:00Z');

const windowsEndingAt = (...windowEnds: number[]) => ({
  windows: windowEnds.map((windowEnd, i) => ({
    budget_policy_id: `bp-${i}`,
    window_start_ms: NOW - 60_000,
    window_end_ms: windowEnd,
    current_spend: 0,
  })),
});

describe('getBudgetWindowsRefetchInterval', () => {
  test('does not refetch when there are no windows', () => {
    expect(getBudgetWindowsRefetchInterval(undefined, false, NOW)).toBe(false);
    expect(getBudgetWindowsRefetchInterval({ windows: [] }, false, NOW)).toBe(false);
  });

  test('refetches 1 second after the earliest window ends', () => {
    expect(getBudgetWindowsRefetchInterval(windowsEndingAt(NOW + 3_600_000, NOW + 30_000), false, NOW)).toBe(31_000);
  });

  test('waits at least 5 seconds, also for a window that has already ended', () => {
    expect(getBudgetWindowsRefetchInterval(windowsEndingAt(NOW + 1_000), false, NOW)).toBe(5_000);
    expect(getBudgetWindowsRefetchInterval(windowsEndingAt(NOW - 60_000), false, NOW)).toBe(5_000);
  });

  test('refetches at least every 5 minutes, so spend does not go stale while every window is long', () => {
    expect(getBudgetWindowsRefetchInterval(windowsEndingAt(NOW + 30 * 86_400_000), false, NOW)).toBe(300_000);
  });

  test('waits 30 seconds between attempts while the request fails', () => {
    // The windows look ended only because no fresh data arrives, so the 5 s minimum would poll a down server.
    expect(getBudgetWindowsRefetchInterval(windowsEndingAt(NOW - 60_000), true, NOW)).toBe(30_000);
    expect(getBudgetWindowsRefetchInterval(windowsEndingAt(NOW + 3_600_000), true, NOW)).toBe(30_000);
  });

  test('does not refetch after a failure when there were no windows', () => {
    expect(getBudgetWindowsRefetchInterval(undefined, true, NOW)).toBe(false);
  });
});
