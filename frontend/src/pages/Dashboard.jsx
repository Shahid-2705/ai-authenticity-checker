import React, { useEffect, useState, useCallback, useRef } from 'react';
import { Film, Mic, Layers, Clock, Cpu, Upload, FileSearch, ArrowUpRight } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import useForensicStore from '../store/useForensicStore';
import EmptyDashboard from '../components/EmptyDashboard';
import { getRiskColorRaw, getRiskBg, normalizeScore } from '../utils/risk';
import { formatRelativeTime, detectMediaRoute } from '../utils/format';
import { Image } from 'lucide-react';

const MEDIA_ICONS = { image: Image, video: Film, audio: Mic, multimodal: Layers };
const ANALYSIS_CARDS = [
  { to: '/image', label: 'Image', desc: 'Synthetic face & pixel manipulation', icon: Image },
  { to: '/video', label: 'Video', desc: 'Temporal consistency & frame analysis', icon: Film },
  { to: '/audio', label: 'Audio', desc: 'Voice cloning & spectrogram forensics', icon: Mic },
  { to: '/multimodal', label: 'Multimodal', desc: 'Cross-modal fusion confidence matrix', icon: Layers },
];
const STAT_DEFS = (totals, models, cleared) => [
  { label: 'Total Scans', value: totals, icon: FileSearch, iconClass: 'text-accent' },
  { label: 'Avg Risk', value: null, icon: Image, iconClass: 'text-risk-caution' },
  { label: 'Models Active', value: models, icon: Cpu, iconClass: 'text-accent' },
  { label: 'Cleared', value: cleared, icon: Image, iconClass: 'text-risk-clear' },
];

export default function Dashboard() {
  const { history, historyTotal, fetchHistory, systemStatus, fetchStatus, isStatusLoading, setPendingFile } = useForensicStore();
  const navigate = useNavigate();
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleQuickFile = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPendingFile(file);
    navigate(detectMediaRoute(file));
  }, [navigate, setPendingFile]);

  useEffect(() => { fetchHistory(20); fetchStatus(); }, [fetchHistory, fetchStatus]);

  const modelsOnline = systemStatus.loaded_models?.length || 0;
  const totalModels = modelsOnline + (systemStatus.missing_models?.length || 0);
  const riskCounts = history.reduce((acc, item) => {
    const pct = normalizeScore(item.risk_score);
    if (pct > 70) acc.critical += 1;
    else if (pct > 40) acc.caution += 1;
    else acc.clear += 1;
    return acc;
  }, { clear: 0, caution: 0, critical: 0 });
  const avgRisk = historyTotal > 0
    ? Math.round(history.reduce((s, i) => s + normalizeScore(i.risk_score), 0) / history.length) : 0;

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    setPendingFile(file);
    navigate(detectMediaRoute(file));
  }, [navigate, setPendingFile]);

  /* Empty state */
  if (historyTotal === 0 && !isStatusLoading) {
    return (
      <div className="space-y-5">
        <div>
          <h1 className="font-display text-xl font-bold tracking-tight gradient-text">Forensics Command Center</h1>
          <p className="text-xs mt-0.5 text-text-3">AI-powered deepfake detection & media authentication</p>
        </div>
        <EmptyDashboard />
      </div>
    );
  }

  const barTotal = riskCounts.clear + riskCounts.caution + riskCounts.critical || 1;
  const recentScans = history.slice(0, 5);
  const stats = STAT_DEFS(historyTotal, `${modelsOnline}/${totalModels}`, riskCounts.clear);
  stats[1].value = `${avgRisk}%`;

  return (
    <div className="space-y-5"
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)} onDrop={handleDrop}>

      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="font-display text-xl font-bold tracking-tight gradient-text">Forensics Command Center</h1>
          <p className="text-xs mt-0.5 text-text-3">AI-powered deepfake detection & media authentication</p>
        </div>
        <span className="text-xs font-mono text-text-3">{modelsOnline}/{totalModels} models online</span>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {stats.map(({ label, value, icon, iconClass }) => {
          const Icon = icon;
          return (
            <div key={label} className="card">
              <div className="flex items-center gap-2 mb-2">
                <Icon size={14} className={iconClass} />
                <span className="label-tag">{label}</span>
              </div>
              <p className="font-display text-lg font-bold text-text-1">{value}</p>
            </div>
          );
        })}
      </div>

      {/* Risk breakdown bar */}
      {historyTotal > 0 && (
        <div className="card">
          <span className="label-tag mb-3 block">Risk Distribution</span>
          <div className="flex gap-1 h-3 rounded-full overflow-hidden">
            <div className="rounded-l-full bg-risk-clear" style={{ width: `${(riskCounts.clear / barTotal) * 100}%` }} />
            <div className="bg-risk-caution" style={{ width: `${(riskCounts.caution / barTotal) * 100}%` }} />
            <div className="rounded-r-full bg-risk-critical" style={{ width: `${(riskCounts.critical / barTotal) * 100}%` }} />
          </div>
          <div className="flex gap-6 mt-2">
            {[{ l: 'cleared', c: riskCounts.clear, cls: 'bg-risk-clear' },
              { l: 'caution', c: riskCounts.caution, cls: 'bg-risk-caution' },
              { l: 'critical', c: riskCounts.critical, cls: 'bg-risk-critical' }].map((b) => (
              <span key={b.l} className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${b.cls}`} />
                <span className="text-xs font-mono text-text-3">{b.c} {b.l}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Quick upload + analysis cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div
          className={`card flex flex-col items-center justify-center text-center cursor-pointer min-h-[160px] border-dashed ${dragOver ? 'border-accent' : ''}`}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*,audio/*"
            onChange={handleQuickFile}
            className="hidden"
            aria-label="Quick analyze file"
          />
          <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-3 bg-accent-dim">
            <Upload size={18} className="text-accent" />
          </div>
          <p className="text-sm font-semibold text-text-1">Quick Analyze</p>
          <p className="text-xs mt-1 text-text-3">Drop any file or click to browse</p>
        </div>
        <div className="lg:col-span-2 grid grid-cols-2 gap-3">
          {ANALYSIS_CARDS.map(({ to, label, desc, icon }) => {
            const Icon = icon;
            return (
              <Link key={to} to={to} className="group card card-hover flex flex-col no-underline">
                <div className="flex items-center justify-between mb-2">
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-accent-dim">
                    <Icon size={15} className="text-accent" />
                  </div>
                  <ArrowUpRight size={12} className="opacity-0 group-hover:opacity-100 transition-opacity text-accent" />
                </div>
                <span className="text-sm font-semibold text-text-1">{label}</span>
                <span className="text-xs mt-0.5 text-text-3">{desc}</span>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Recent scans table */}
      {recentScans.length > 0 && (
        <div className="card overflow-hidden !p-0">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border-dim">
            <div className="flex items-center gap-2">
              <Clock size={13} className="text-accent" />
              <span className="label-tag">Recent Activity</span>
            </div>
            <Link to="/history" className="text-xs font-medium text-accent no-underline">View all</Link>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-text-3 text-xs">
                <th className="px-4 py-2 font-medium">Time</th>
                <th className="px-4 py-2 font-medium">Type</th>
                <th className="px-4 py-2 font-medium">File</th>
                <th className="px-4 py-2 font-medium text-right">Risk</th>
              </tr>
            </thead>
            <tbody>
              {recentScans.map((item) => {
                const pct = normalizeScore(item.risk_score);
                const Icon = MEDIA_ICONS[item.media_type] || FileSearch;
                return (
                  <tr key={item.id} className="border-t border-border-dim cursor-pointer table-row-hover"
                    onClick={() => navigate('/history')}>
                    <td className="px-4 py-2 text-xs font-mono text-text-3">{formatRelativeTime(item.created_at || item.timestamp)}</td>
                    <td className="px-4 py-2"><Icon size={14} className="text-text-2" /></td>
                    <td className="px-4 py-2 truncate max-w-[200px] text-text-1">{item.file_name || `${item.media_type} analysis`}</td>
                    <td className="px-4 py-2 text-right">
                      <span className="text-xs font-bold font-mono px-2 py-0.5 rounded" style={{ color: getRiskColorRaw(pct), background: getRiskBg(pct) }}>
                        {pct.toFixed(0)}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
