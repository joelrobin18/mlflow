import { describe, test, expect, beforeEach, jest } from '@jest/globals';
import { initUIConfig, getUIConfig, getSupportPageUrl } from './uiConfig';

const DEFAULT_SUPPORT_PAGE_URL =
  'https://github.com/mlflow/mlflow/issues/new?template=ui_bug_report_template.yaml';

describe('uiConfig', () => {
  beforeEach(() => {
    // Reset fetch mock before each test
    global.fetch = jest.fn() as jest.MockedFunction<typeof global.fetch>;
  });

  describe('getSupportPageUrl', () => {
    test('returns default URL when not initialized', () => {
      const url = getSupportPageUrl();
      expect(url).toBe(DEFAULT_SUPPORT_PAGE_URL);
    });
  });

  describe('getUIConfig', () => {
    test('returns default config when not initialized', () => {
      const config = getUIConfig();
      expect(config.supportPageUrl).toBe(DEFAULT_SUPPORT_PAGE_URL);
    });
  });

  describe('initUIConfig', () => {
    test('fetches config from server and updates cached value', async () => {
      const customUrl = 'https://example.com/support';
      (global.fetch as jest.MockedFunction<typeof global.fetch>).mockResolvedValue({
        ok: true,
        json: async () => ({ support_page_url: customUrl }),
      } as Response);

      const config = await initUIConfig();

      expect(global.fetch).toHaveBeenCalledWith('ajax-api/3.0/mlflow/ui-config', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      expect(config.supportPageUrl).toBe(customUrl);
      expect(getSupportPageUrl()).toBe(customUrl);
    });

    test('returns default config when fetch fails', async () => {
      (global.fetch as jest.MockedFunction<typeof global.fetch>).mockRejectedValue(
        new Error('Network error'),
      );

      const config = await initUIConfig();

      expect(config.supportPageUrl).toBe(DEFAULT_SUPPORT_PAGE_URL);
    });

    test('returns default config when server returns error', async () => {
      (global.fetch as jest.MockedFunction<typeof global.fetch>).mockResolvedValue({
        ok: false,
        status: 500,
      } as Response);

      const config = await initUIConfig();

      expect(config.supportPageUrl).toBe(DEFAULT_SUPPORT_PAGE_URL);
    });
  });
});
