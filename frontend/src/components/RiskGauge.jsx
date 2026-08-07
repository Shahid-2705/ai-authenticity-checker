import React, { useEffect, useState } from 'react';
import { getRiskColorRaw, getRiskLevel } from '../utils/risk';

export default function RiskGauge({ percentage, label = 'AI Risk Score', size = 160 }) {
  const [animated, setAnimated] = useState(false);

  const radius = size * 0.38;
  const circumference = 2 * Math.PI * radius;
  const strokeWidth = 6;
  const color = getRiskColorRaw(percentage);
  const offset = animated ? ((100 - percentage) / 100) * circumference : circumference;

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 120);
    return () => clearTimeout(t);
  }, [percentage]);

  return (
    <div className="flex flex-col items-center justify-center py-3">
      <div
        className="relative"
        style={{
          width: size,
          height: size,
          transform: animated ? 'scale(1)' : 'scale(0.95)',
          opacity: animated ? 1 : 0,
          transition: 'transform 0.5s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.3s ease',
        }}
      >
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: 'rotate(-90deg)' }}>
          <circle
            cx={size / 2} cy={size / 2} r={radius}
            fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth={strokeWidth}
          />
          <circle
            cx={size / 2} cy={size / 2} r={radius}
            fill="none" stroke={color} strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 1s cubic-bezier(0.22, 1, 0.36, 1)' }}
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-display font-bold tracking-tight text-2xl leading-none text-text-1">
            {percentage.toFixed(1)}<span className="text-xs">%</span>
          </span>
          <span className="text-xs mt-1 text-text-3">
            {label}
          </span>
        </div>
      </div>

      <div
        className="mt-2 px-3 py-1 rounded-full text-xs font-medium"
        style={{ background: `${color}10`, color, border: `1px solid ${color}26` }}
      >
        {getRiskLevel(percentage)}
      </div>
    </div>
  );
}
