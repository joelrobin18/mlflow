/**
 * 3D Histogram Plot Component
 * Displays histogram distributions over time in a 3D surface plot
 */

import React, { useMemo } from 'react';
import type { Layout, Data } from 'plotly.js';
import { LazyPlot } from '../../LazyPlot';

export interface HistogramData {
  name: string;
  step: number;
  timestamp: number;
  bin_edges: number[];
  counts: number[];
  min_value?: number;
  max_value?: number;
}

export interface RunsHistogram3DPlotProps {
  /**
   * Array of histogram data points to display
   */
  histograms: HistogramData[];
  /**
   * Title for the chart
   */
  title?: string;
  /**
   * Height of the plot in pixels
   */
  height?: number;
  /**
   * Whether to use log scale for count axis
   */
  logScale?: boolean;
}

/**
 * Converts histogram data into a 3D surface plot format
 */
const convertHistogramsTo3DSurface = (histograms: HistogramData[], logScale: boolean = false) => {
  if (histograms.length === 0) {
    return {
      x: [],
      y: [],
      z: [],
    };
  }

  // Sort histograms by step
  const sortedHistograms = [...histograms].sort((a, b) => a.step - b.step);

  // Get unique steps
  const steps = sortedHistograms.map((h) => h.step);

  // Get bin centers from first histogram (assuming all have same binning)
  const firstHist = sortedHistograms[0];
  const binCenters = firstHist.bin_edges.slice(0, -1).map((edge, i) => {
    return (edge + firstHist.bin_edges[i + 1]) / 2;
  });

  // Build Z matrix (counts at each step and bin)
  const zMatrix = sortedHistograms.map((hist) => {
    const counts = hist.counts.map((count) => (logScale && count > 0 ? Math.log10(count + 1) : count));
    return counts;
  });

  return {
    x: binCenters, // bin centers (values)
    y: steps, // training steps
    z: zMatrix, // counts matrix
  };
};

/**
 * 3D Histogram Plot Component
 *
 * Visualizes how histogram distributions evolve over training steps.
 * Uses a 3D surface plot where:
 * - X axis: Bin values (e.g., weight values)
 * - Y axis: Training step
 * - Z axis: Count/frequency
 */
export const RunsHistogram3DPlot: React.FC<RunsHistogram3DPlotProps> = ({
  histograms,
  title = 'Histogram Distribution Over Time',
  height = 600,
  logScale = false,
}) => {
  const plotData = useMemo(() => {
    const { x, y, z } = convertHistogramsTo3DSurface(histograms, logScale);

    if (x.length === 0 || y.length === 0 || z.length === 0) {
      return [];
    }

    const trace: Partial<Data> = {
      type: 'surface',
      x,
      y,
      z,
      colorscale: 'Viridis',
      showscale: true,
      colorbar: {
        title: logScale ? 'log10(Count + 1)' : 'Count',
        titleside: 'right',
      },
      hovertemplate: 'Value: %{x:.3f}<br>Step: %{y}<br>Count: %{z:.2f}<extra></extra>',
    };

    return [trace];
  }, [histograms, logScale]);

  const layout = useMemo<Partial<Layout>>(() => {
    return {
      title: {
        text: title,
        font: { size: 16 },
      },
      height,
      autosize: true,
      scene: {
        xaxis: {
          title: 'Value',
          gridcolor: '#e0e0e0',
          showbackground: true,
          backgroundcolor: '#f8f8f8',
        },
        yaxis: {
          title: 'Step',
          gridcolor: '#e0e0e0',
          showbackground: true,
          backgroundcolor: '#f8f8f8',
        },
        zaxis: {
          title: logScale ? 'log10(Count + 1)' : 'Count',
          gridcolor: '#e0e0e0',
          showbackground: true,
          backgroundcolor: '#f8f8f8',
        },
        camera: {
          eye: { x: 1.5, y: 1.5, z: 1.3 },
        },
      },
      margin: {
        l: 0,
        r: 0,
        t: 50,
        b: 0,
      },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
    };
  }, [title, height, logScale]);

  const config = useMemo(
    () => ({
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['sendDataToCloud', 'autoScale2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: 'histogram_3d',
        height: 800,
        width: 1200,
        scale: 2,
      },
    }),
    [],
  );

  if (histograms.length === 0) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#999',
          fontSize: 14,
        }}
      >
        No histogram data available
      </div>
    );
  }

  return <LazyPlot data={plotData} layout={layout} config={config} />;
};

export default RunsHistogram3DPlot;
