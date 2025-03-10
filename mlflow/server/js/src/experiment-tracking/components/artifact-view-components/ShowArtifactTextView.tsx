import React, { Component } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { coy as style, atomDark as darkStyle } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { getLanguage } from '../../../common/utils/FileUtils';
import {
  getArtifactContent,
  getArtifactLocationUrl,
  getLoggedModelArtifactLocationUrl,
} from '../../../common/utils/ArtifactUtils';
import './ShowArtifactTextView.css';
import { DesignSystemHocProps, WithDesignSystemThemeHoc, ToggleButton } from '@databricks/design-system';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';

const LARGE_ARTIFACT_SIZE = 100 * 1024;

type Props = DesignSystemHocProps & {
  runUuid: string;
  path: string;
  size?: number;
  getArtifact?: (...args: any[]) => any;
} & LoggedModelArtifactViewerProps;

type State = {
  loading?: boolean;
  error?: Error;
  text?: string;
  path?: string;
  autoRefresh: boolean;
};

class ShowArtifactTextView extends Component<Props, State> {
  private autoRefreshInterval?: number;

  constructor(props: Props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
  }

  static defaultProps = {
    getArtifact: getArtifactContent,
  };

  state = {
    loading: true,
    error: undefined,
    text: undefined,
    path: undefined,
    autoRefresh: false, // auto-refresh is off by default
  };

  componentDidMount() {
    this.fetchArtifacts();
    if (this.state.autoRefresh) {
      this.startAutoRefresh();
    }
  }

  componentDidUpdate(prevProps: Props, prevState: State) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.fetchArtifacts();
    }
    if (prevState.autoRefresh !== this.state.autoRefresh) {
      if (this.state.autoRefresh) {
        this.startAutoRefresh();
      } else {
        this.stopAutoRefresh();
      }
    }
  }

  componentWillUnmount() {
    this.stopAutoRefresh();
  }

  startAutoRefresh() {
    if (!this.autoRefreshInterval) {
      // Set auto-refresh interval (every 5 seconds)
      this.autoRefreshInterval = window.setInterval(this.fetchArtifacts, 5000);
    }
  }

  stopAutoRefresh() {
    if (this.autoRefreshInterval) {
      clearInterval(this.autoRefreshInterval);
      this.autoRefreshInterval = undefined;
    }
  }

  handleAutoRefreshToggle = (pressed: boolean) => {
    this.setState({ autoRefresh: pressed });
  };

  render() {
    if (this.state.loading || this.state.path !== this.props.path) {
      return <ArtifactViewSkeleton className="artifact-text-view-loading" />;
    }
    if (this.state.error) {
      return <ArtifactViewErrorState className="artifact-text-view-error" />;
    } else {
      const isLargeFile = (this.props.size || 0) > LARGE_ARTIFACT_SIZE;
      const language = isLargeFile ? 'text' : getLanguage(this.props.path);
      const { theme } = this.props.designSystemThemeApi;

      const overrideStyles = {
        fontFamily: 'Source Code Pro,Menlo,monospace',
        fontSize: theme.typography.fontSizeMd,
        overflow: 'auto',
        marginTop: '0',
        width: '100%',
        height: '100%',
        padding: theme.spacing.xs,
        borderColor: theme.colors.borderDecorative,
        border: 'none',
      };
      const renderedContent = this.state.text ? prettifyArtifactText(language, this.state.text) : this.state.text;

      const syntaxStyle = theme.isDarkMode ? darkStyle : style;

      return (
        <div className="ShowArtifactPage">
          {/* auto-refresh toggle button */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '8px', gap: '8px' }}>
            <ToggleButton
              componentId="auto-refresh-toggle"
              pressed={this.state.autoRefresh}
              onPressedChange={this.handleAutoRefreshToggle}
            >
              Auto-refresh
            </ToggleButton>
          </div>
          <div className="text-area-border-box">
            <SyntaxHighlighter language={language} style={syntaxStyle} customStyle={overrideStyles}>
              {renderedContent ?? ''}
            </SyntaxHighlighter>
          </div>
        </div>
      );
    }
  }

  /** Fetches artifacts and updates component state with the result */
  fetchArtifacts() {
    this.setState({ loading: true });
    const { isLoggedModelsMode, loggedModelId, path, runUuid } = this.props;

    const artifactLocation =
      isLoggedModelsMode && loggedModelId
        ? getLoggedModelArtifactLocationUrl(path, loggedModelId)
        : getArtifactLocationUrl(path, runUuid);

    this.props
      .getArtifact?.(artifactLocation)
      .then((text: string) => {
        this.setState({ text: text, loading: false });
      })
      .catch((error: Error) => {
        this.setState({ error: error, loading: false });
      });
    this.setState({ path: this.props.path });
  }
}

export function prettifyArtifactText(language: string, rawText: string) {
  if (language === 'json') {
    try {
      const parsedJson = JSON.parse(rawText);
      return JSON.stringify(parsedJson, null, 2);
    } catch (e) {
      // No-op
    }
    return rawText;
  }
  return rawText;
}
export default React.memo(WithDesignSystemThemeHoc(ShowArtifactTextView));
