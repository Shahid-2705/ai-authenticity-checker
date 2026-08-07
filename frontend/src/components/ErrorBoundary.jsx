import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('ErrorBoundary caught:', error, info.componentStack);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div className="flex flex-col items-center justify-center min-h-[300px] p-8 text-center">
        <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-4 bg-risk-criticalDim border border-[rgba(251,113,133,0.20)]">
          <AlertTriangle size={24} className="text-risk-critical" />
        </div>
        <h2 className="font-display text-lg font-bold mb-1 text-text-1">
          Something went wrong
        </h2>
        <p className="text-sm mb-4 max-w-md text-text-2">
          {this.state.error?.message || 'An unexpected error occurred.'}
        </p>
        <button onClick={this.handleReset} className="btn-primary text-sm">
          <RefreshCw size={14} />
          Try Again
        </button>
      </div>
    );
  }
}
