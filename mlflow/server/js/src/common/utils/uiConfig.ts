/**
 * UI Configuration service for fetching server-side configuration.
 * This module fetches configurable UI settings from the MLflow server,
 * such as the support page URL for error pages.
 */

// Default support page URL (MLflow GitHub issues)
const DEFAULT_SUPPORT_PAGE_URL =
  'https://github.com/mlflow/mlflow/issues/new?template=ui_bug_report_template.yaml';

interface UIConfig {
  supportPageUrl: string;
}

// Cached config - starts with default and is updated after fetch
let cachedConfig: UIConfig = {
  supportPageUrl: DEFAULT_SUPPORT_PAGE_URL,
};

// Promise to track if config is being fetched
let fetchPromise: Promise<UIConfig> | null = null;

/**
 * Fetches UI configuration from the server.
 * Uses a relative URL to support reverse proxy setups.
 */
async function fetchUIConfig(): Promise<UIConfig> {
  try {
    const response = await fetch('ajax-api/3.0/mlflow/ui-config', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      console.warn(`[UIConfig] Failed to fetch config: ${response.status}, using defaults`);
      return cachedConfig;
    }

    const data = await response.json();
    return {
      supportPageUrl: data.support_page_url || DEFAULT_SUPPORT_PAGE_URL,
    };
  } catch (error) {
    console.warn('[UIConfig] Failed to fetch config, using defaults:', error);
    return cachedConfig;
  }
}

/**
 * Initializes the UI config by fetching from the server.
 * This should be called early in the application lifecycle.
 * The config is cached and subsequent calls return the cached value.
 */
export async function initUIConfig(): Promise<UIConfig> {
  if (fetchPromise) {
    return fetchPromise;
  }

  fetchPromise = fetchUIConfig().then((config) => {
    cachedConfig = config;
    return config;
  });

  return fetchPromise;
}

/**
 * Gets the current UI configuration.
 * Returns the cached config (default values if not yet fetched).
 */
export function getUIConfig(): UIConfig {
  return cachedConfig;
}

/**
 * Gets the support page URL.
 * This is a convenience function for the most commonly used config value.
 */
export function getSupportPageUrl(): string {
  return cachedConfig.supportPageUrl;
}
