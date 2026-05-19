import React, { useEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Menu } from 'lucide-react';
import Sidebar from './Sidebar';
import NeuralBackground from './NeuralBackground';
import ErrorBoundary from './ErrorBoundary';
import Breadcrumbs from './Breadcrumbs';
import Toaster from './Toaster';
import useForensicStore from '../store/useForensicStore';

export default function Layout() {
  const { fetchStatus } = useForensicStore();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return (
    <div className="flex min-h-screen overflow-hidden relative bg-bg-void">
      {/* Skip-to-content */}
      <a href="#main-content" className="skip-nav">Skip to content</a>

      <NeuralBackground />

      {/* Mobile hamburger */}
      <button
        className="md:hidden fixed top-3 left-3 z-[60] p-2 rounded-lg bg-[rgba(12,15,22,0.9)] border border-border-dim text-text-1"
        onClick={() => setSidebarOpen((prev) => !prev)}
        aria-label="Toggle navigation menu"
      >
        <Menu size={18} />
      </button>

      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <main
        id="main-content"
        className="flex-1 overflow-y-auto h-screen relative z-10 md:ml-[200px]"
        style={{ padding: '24px 20px', paddingTop: 'max(24px, env(safe-area-inset-top))' }}
      >
        <div className="max-w-[1280px] mx-auto pt-10 md:pt-0">
          <Breadcrumbs />
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </div>
      </main>

      <Toaster />
    </div>
  );
}
