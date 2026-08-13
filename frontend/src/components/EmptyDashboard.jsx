import React, { useRef, useCallback } from 'react';
import { Shield, Upload, Cpu, CheckCircle, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { detectMediaRoute } from '../utils/format';
import useForensicStore from '../store/useForensicStore';

const STEPS = [
  { icon: Upload, title: 'Upload Media', desc: 'Image, video, or audio file' },
  { icon: Cpu, title: 'AI Analysis', desc: 'Multi-model forensic ensemble' },
  { icon: CheckCircle, title: 'Get Verdict', desc: 'Risk score & detailed report' },
];

export default function EmptyDashboard() {
  const navigate = useNavigate();
  const { setPendingFile } = useForensicStore();
  const fileInputRef = useRef(null);

  const handleFile = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPendingFile(file);
    navigate(detectMediaRoute(file));
  }, [navigate, setPendingFile]);

  return (
    <div className="flex flex-col items-center justify-center py-12 px-4">
      {/* Shield icon */}
      <div className="mb-6">
        <div className="w-16 h-16 rounded-2xl flex items-center justify-center bg-accent-dim border border-accent/20">
          <Shield size={30} className="text-accent" />
        </div>
      </div>

      <h2 className="font-display text-2xl font-bold gradient-text mb-1">
        System Ready
      </h2>
      <p className="text-sm mb-8 text-text-3">
        No scans yet. Start your first forensic analysis in three steps.
      </p>

      {/* Step cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full max-w-xl mb-8">
        {STEPS.map((step, i) => {
          const Icon = step.icon;
          return (
            <div key={step.title} className="card p-5 flex flex-col items-center text-center">
              <div className="flex items-center gap-2 mb-3">
                <span className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold bg-accent-dim text-accent border border-accent/20">
                  {i + 1}
                </span>
                <Icon size={15} className="text-accent" />
              </div>
              <p className="text-sm font-semibold text-text-1">{step.title}</p>
              <p className="text-xs mt-0.5 text-text-3">{step.desc}</p>
            </div>
          );
        })}
      </div>

      <div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,video/*,audio/*"
          onChange={handleFile}
          className="hidden"
          aria-label="Upload media file"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="btn-primary px-8 py-3"
        >
          Start Your First Scan
          <ArrowRight size={15} />
        </button>
      </div>
    </div>
  );
}
