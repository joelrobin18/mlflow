/**
 * Histogram Chart Card Component
 * Displays 3D histogram visualization in a card format
 */

import React, { useState, useEffect, useMemo } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartsHistogramCardConfig } from '../../runs-charts.types';
import { RunsHistogram3DPlot, type HistogramData } from '../RunsHistogram3DPlot';
import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export interface RunsChartsHistogramChartCardProps {
  config: RunsChartsHistogramCardConfig;
  chartRunData: RunsChartsRunData[];
}

/**
 * Fetches histogram data from artifacts for a given run and histogram key
 */
const fetchHistogramData = async (runId: string, histogramKey: string): Promise<HistogramData[]> => {
  try {
    // Sanitize key for filename (replace / with #)
    const sanitizedKey = histogramKey.replace(/\//g, '#');
    const artifactPath = `histograms/${sanitizedKey}.json`;

    // Fetch from artifacts endpoint
    const response = await fetch(
      `/ajax-api/2.0/mlflow/artifacts/list?run_id=${runId}&path=histograms`,
    );

    if (!response.ok) {
      console.warn(`Failed to list artifacts for run ${runId}`);
      return [];
    }

    const artifactsList = await response.json();

    // Check if histogram file exists
    const histogramFile = artifactsList.files?.find((f: any) =>
      f.path.includes(sanitizedKey),
    );

    if (!histogramFile) {
      console.warn(`Histogram ${histogramKey} not found for run ${runId}`);
      return [];
    }

    // Download the histogram data
    const downloadResponse = await fetch(
      `/get-artifact?path=${encodeURIComponent(artifactPath)}&run_uuid=${runId}`,
    );

    if (!downloadResponse.ok) {
      console.warn(`Failed to download histogram ${histogramKey} for run ${runId}`);
      return [];
    }

    const histogramData = await downloadResponse.json();

    // Handle both single histogram object and array of histograms
    return Array.isArray(histogramData) ? histogramData : [histogramData];
  } catch (error) {
    console.error(`Error fetching histogram data for run ${runId}:`, error);
    return [];
  }
};

export const RunsChartsHistogramChartCard: React.FC<RunsChartsHistogramChartCardProps> = ({
  config,
  chartRunData,
}) => {
  const { theme } = useDesignSystemTheme();
  const [histograms, setHistograms] = useState<HistogramData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Determine which runs to fetch data for
  const selectedRuns = useMemo(() => {
    if (config.selectedRunUuids && config.selectedRunUuids.length > 0) {
      return chartRunData.filter((run) =>
        config.selectedRunUuids.includes(run.uuid),
      );
    }
    // Default to first run if none selected
    return chartRunData.slice(0, 1);
  }, [chartRunData, config.selectedRunUuids]);

  // Fetch histogram data
  useEffect(() => {
    const fetchData = async () => {
      if (config.histogramKeys.length === 0 || selectedRuns.length === 0) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        // For now, fetch data for the first histogram key and first run
        // TODO: Support multiple histograms and runs
        const firstKey = config.histogramKeys[0];
        const firstRun = selectedRuns[0];

        const data = await fetchHistogramData(firstRun.uuid, firstKey);
        setHistograms(data);
      } catch (err) {
        console.error('Error loading histogram data:', err);
        setError('Failed to load histogram data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [config.histogramKeys, selectedRuns]);

  const title = useMemo(() => {
    if (config.displayName) {
      return config.displayName;
    }
    if (config.histogramKeys.length > 0) {
      return `Histogram: ${config.histogramKeys[0]}`;
    }
    return 'Histogram Distribution';
  }, [config.displayName, config.histogramKeys]);

  if (loading) {
    return (
      <div
        style={{
          padding: theme.spacing.md,
          height: 600,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.backgroundPrimary,
          borderRadius: theme.borders.borderRadiusMd,
          border: `1px solid ${theme.colors.borderDecorative}`,
        }}
      >
        <FormattedMessage
          defaultMessage="Loading histogram data..."
          description="Loading message for histogram chart"
        />
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          padding: theme.spacing.md,
          height: 600,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.backgroundPrimary,
          borderRadius: theme.borders.borderRadiusMd,
          border: `1px solid ${theme.colors.borderDecorative}`,
          color: theme.colors.textValidationDanger,
        }}
      >
        <div style={{ marginBottom: theme.spacing.sm }}>{error}</div>
        <div style={{ fontSize: 12, color: theme.colors.textSecondary }}>
          Make sure histogram data has been logged using mlflow.log_histogram()
        </div>
      </div>
    );
  }

  if (config.histogramKeys.length === 0) {
    return (
      <div
        style={{
          padding: theme.spacing.md,
          height: 600,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme.colors.backgroundPrimary,
          borderRadius: theme.borders.borderRadiusMd,
          border: `1px solid ${theme.colors.borderDecorative}`,
          color: theme.colors.textSecondary,
        }}
      >
        <FormattedMessage
          defaultMessage="No histogram selected. Configure the chart to select histogram keys."
          description="Empty state message for histogram chart"
        />
      </div>
    );
  }

  return (
    <div
      style={{
        padding: theme.spacing.md,
        backgroundColor: theme.colors.backgroundPrimary,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.borderDecorative}`,
      }}
    >
      <RunsHistogram3DPlot
        histograms={histograms}
        title={title}
        height={600}
        logScale={false}
      />
    </div>
  );
};

export default RunsChartsHistogramChartCard;
