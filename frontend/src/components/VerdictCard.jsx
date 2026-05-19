import React from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert } from 'lucide-react';

/**
 * Risk level tiers with explicit keyword lists.
 * Order matters: critical is checked first, then caution, then fallback to clear.
 *
 * The `riskScore` prop (0-100) is used as the primary signal when available.
 * Keyword matching on the verdict string is a fallback only.
 */
const CRITICAL_KEYWORDS = ['critical', 'fake', 'manipulat', 'deepfake', 'synthetic', 'forged', 'altered'];
const CAUTION_KEYWORDS = ['medium', 'suspicious', 'caution', 'uncertain', 'inconclusive', 'partial'];
const CLEAR_KEYWORDS = ['authentic', 'genuine', 'real', 'clear', 'low'];

const TIERS = {
  critical: {
    Icon: ShieldAlert,
    bgClass: 'bg-risk-criticalDim border-[rgba(251,113,133,0.15)]',
    accentClass: 'text-risk-critical',
  },
  caution: {
    Icon: AlertTriangle,
    bgClass: 'bg-risk-cautionDim border-[rgba(251,191,36,0.15)]',
    accentClass: 'text-risk-caution',
  },
  clear: {
    Icon: CheckCircle,
    bgClass: 'bg-risk-clearDim border-[rgba(52,211,153,0.15)]',
    accentClass: 'text-risk-clear',
  },
};

function classifyVerdict(verdict, riskScore) {
  // Primary signal: numeric risk score when available
  if (riskScore != null && typeof riskScore === 'number') {
    if (riskScore > 70) return 'critical';
    if (riskScore > 40) return 'caution';
    return 'clear';
  }

  // Fallback: keyword matching on verdict text
  const lower = (verdict || '').toLowerCase();
  if (CRITICAL_KEYWORDS.some((kw) => lower.includes(kw))) return 'critical';
  if (CAUTION_KEYWORDS.some((kw) => lower.includes(kw))) return 'caution';
  if (CLEAR_KEYWORDS.some((kw) => lower.includes(kw))) return 'clear';

  // Default to caution when unable to classify
  return 'caution';
}

export default function VerdictCard({ verdict, riskScore }) {
  if (!verdict) return null;

  const tier = classifyVerdict(verdict, riskScore);
  const { Icon, bgClass, accentClass } = TIERS[tier];

  return (
    <div className={`rounded-lg p-4 mt-3 border ${bgClass}`}>
      <div className={`flex items-center gap-2 mb-1.5 ${accentClass}`}>
        <Icon size={14} />
        <span className="label-tag">Verdict</span>
      </div>
      <p className="text-sm font-medium leading-snug text-text-1">
        {verdict}
      </p>
    </div>
  );
}
