/**
 * Shared risk-level utilities.
 *
 * Centralizes the color/label/background logic that was duplicated
 * across Dashboard, History, ScoreBar, RiskGauge, and VerdictCard.
 */

/** Thresholds (percentage 0-100). */
const HIGH_THRESHOLD = 70;
const MEDIUM_THRESHOLD = 40;

/**
 * Normalize a score that may be 0-1 or 0-100 into 0-100 percentage.
 *
 * Scores <= 1 are treated as fractional (0.0–1.0) and scaled to 0–100.
 * Scores > 1 are already in percentage form and clamped to 100.
 */
export function normalizeScore(score) {
  if (score == null) return 0;
  const n = Number(score);
  if (Number.isNaN(n)) return 0;
  if (n < 0) return 0;
  if (n <= 1) return n * 100;
  return Math.min(n, 100);
}

export function getRiskColor(pct) {
  if (pct > HIGH_THRESHOLD) return 'var(--risk-critical)';
  if (pct > MEDIUM_THRESHOLD) return 'var(--risk-caution)';
  return 'var(--risk-clear)';
}

export function getRiskColorRaw(pct) {
  if (pct > HIGH_THRESHOLD) return '#FB7185';
  if (pct > MEDIUM_THRESHOLD) return '#FBBF24';
  return '#34D399';
}

export function getRiskBg(pct) {
  if (pct > HIGH_THRESHOLD) return 'rgba(251,113,133,0.10)';
  if (pct > MEDIUM_THRESHOLD) return 'rgba(251,191,36,0.10)';
  return 'rgba(52,211,153,0.10)';
}

export function getRiskLabel(pct) {
  if (pct > HIGH_THRESHOLD) return 'Deepfake Detected';
  if (pct > MEDIUM_THRESHOLD) return 'Suspicious';
  return 'Authentic';
}

export function getRiskLevel(pct) {
  if (pct > HIGH_THRESHOLD) return 'High Risk';
  if (pct > MEDIUM_THRESHOLD) return 'Medium Risk';
  return 'Low Risk';
}

export function getRiskGlow(pct) {
  if (pct > HIGH_THRESHOLD) return 'rgba(251,113,133,0.4)';
  if (pct > MEDIUM_THRESHOLD) return 'rgba(251,191,36,0.4)';
  return 'rgba(52,211,153,0.4)';
}
