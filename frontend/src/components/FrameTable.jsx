import React from 'react';

const COLUMNS = [
  { key: 0, label: 'Frame', title: 'Frame number' },
  { key: 1, label: 'Time',  title: 'Timestamp in video' },
  { key: 2, label: 'Risk',  title: 'Aggregated risk score', isRisk: true },
  { key: 3, label: 'Pred',  title: 'Prediction label' },
  { key: 4, label: 'Face',  title: 'Face detected', center: true },
  { key: 5, label: 'ViT',   title: 'Vision Transformer score' },
  { key: 6, label: 'Freq',  title: 'Frequency analysis score' },
  { key: 7, label: 'Forns', title: 'Forensics model score' },
  { key: 8, label: 'FaceM', title: 'Face manipulation score' },
  { key: 9, label: 'DINO',  title: 'DINOv2 model score' },
  { key: 10, label: 'Eff',  title: 'EfficientNet score' },
];

function getRiskColor(value) {
  const risk = parseFloat(value) || 0;
  if (risk > 0.70) return '#FB7185';
  if (risk > 0.40) return '#FBBF24';
  return '#34D399';
}

function getRiskGlow(value) {
  const risk = parseFloat(value) || 0;
  if (risk > 0.70) return '0 0 6px rgba(251,113,133,0.3)';
  if (risk > 0.40) return '0 0 6px rgba(251,191,36,0.3)';
  return 'none';
}

export default function FrameTable({ framesRawStr }) {
  if (!framesRawStr) return null;

  const lines = framesRawStr.split('\n');
  if (lines.length < 3) return <pre className="text-xs text-text-3">{framesRawStr}</pre>;

  const rows = [];
  for (let i = 2; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const cols = line.split(/\s+/);
    if (cols.length >= 8) {
      rows.push(cols);
    }
  }

  return (
    <div className="w-full rounded-[10px] overflow-hidden">
      {/* Mobile: card layout */}
      <div className="md:hidden space-y-2">
        {rows.map((row, idx) => (
          <div
            key={idx}
            className="inset-panel p-3"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-text-1">
                Frame {row[0]}
              </span>
              <span
                className="text-xs font-mono font-bold px-2 py-0.5 rounded"
                style={{
                  color: getRiskColor(row[2]),
                  background: `${getRiskColor(row[2])}18`,
                }}
              >
                Risk: {row[2]}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
              <span className="text-text-3">Time</span>
              <span className="font-mono text-text-2">{row[1]}</span>
              <span className="text-text-3">Prediction</span>
              <span className="text-text-2">{row[3]}</span>
              <span className="text-text-3">Face</span>
              <span className="text-text-2">{row[4]}</span>
              {row[5] && (
                <>
                  <span className="text-text-3">ViT</span>
                  <span className="font-mono text-text-2">{row[5]}</span>
                </>
              )}
              {row[6] && (
                <>
                  <span className="text-text-3">Freq</span>
                  <span className="font-mono text-text-2">{row[6]}</span>
                </>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Desktop: scrollable table */}
      <div className="hidden md:block table-scroll">
        <table className="w-full text-left text-xs min-w-[700px]">
          <thead>
            <tr className="bg-gradient-to-br from-[rgba(59,130,246,0.08)] to-[rgba(6,182,212,0.04)] border-b border-[rgba(59,130,246,0.10)]">
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  title={col.title}
                  className={`px-3 py-2.5 text-[11px] uppercase tracking-wide whitespace-nowrap text-text-3 ${col.center ? 'text-center' : ''}`}
                >
                  <span className="cursor-help border-b border-dotted border-current">
                    {col.label}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr
                key={idx}
                className="table-row-hover border-b border-white/[0.04]"
              >
                {COLUMNS.map((col) => {
                  const value = row[col.key];
                  if (col.isRisk) {
                    return (
                      <td
                        key={col.key}
                        className="px-3 py-2 font-mono font-bold"
                        style={{ color: getRiskColor(value), textShadow: getRiskGlow(value) }}
                      >
                        {value}
                      </td>
                    );
                  }
                  return (
                    <td
                      key={col.key}
                      className={`px-3 py-2 font-mono ${col.center ? 'text-center' : ''} ${col.key === 0 ? 'text-text-1' : 'text-text-2'}`}
                    >
                      {value}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
