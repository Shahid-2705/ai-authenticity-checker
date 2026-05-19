import React, { Suspense, useEffect, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import ErrorBoundary from './components/ErrorBoundary';
import useAuthStore from './store/useAuthStore';
import { isAuthEnabled } from './services/supabase';

// Auth pages — small, keep eager for instant paint
import Login from './pages/Login';
import Signup from './pages/Signup';
import ForgotPassword from './pages/ForgotPassword';

// App pages — lazy-loaded to reduce initial bundle
const Dashboard = lazy(() => import('./pages/Dashboard'));
const ImageAnalysis = lazy(() => import('./pages/ImageAnalysis'));
const VideoAnalysis = lazy(() => import('./pages/VideoAnalysis'));
const AudioAnalysis = lazy(() => import('./pages/AudioAnalysis'));
const Multimodal = lazy(() => import('./pages/Multimodal'));
const History = lazy(() => import('./pages/History'));
const SystemStatus = lazy(() => import('./pages/SystemStatus'));

function PageLoader() {
  return (
    <div className="flex items-center justify-center py-24">
      <div className="w-5 h-5 border-2 rounded-full animate-spin border-accent border-t-transparent" />
    </div>
  );
}

function ProtectedRoute({ children }) {
  const { user, isLoading } = useAuthStore();

  if (!isAuthEnabled()) return children;
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg-void">
        <div className="flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 rounded-full animate-spin border-accent border-t-transparent" />
          <span className="text-sm text-text-2">Loading...</span>
        </div>
      </div>
    );
  }
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

function App() {
  const { initialize } = useAuthStore();

  useEffect(() => {
    initialize();
  }, [initialize]);

  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          {/* Public auth routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />

          {/* Protected app routes */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Suspense fallback={<PageLoader />}><Dashboard /></Suspense>} />
            <Route path="image" element={<Suspense fallback={<PageLoader />}><ImageAnalysis /></Suspense>} />
            <Route path="video" element={<Suspense fallback={<PageLoader />}><VideoAnalysis /></Suspense>} />
            <Route path="audio" element={<Suspense fallback={<PageLoader />}><AudioAnalysis /></Suspense>} />
            <Route path="multimodal" element={<Suspense fallback={<PageLoader />}><Multimodal /></Suspense>} />
            <Route path="history" element={<Suspense fallback={<PageLoader />}><History /></Suspense>} />
            <Route path="system" element={<Suspense fallback={<PageLoader />}><SystemStatus /></Suspense>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
