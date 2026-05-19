import React from 'react';

/**
 * Simplified ambient background.
 *
 * Reduced from 3 animated blurred orbs + scanline overlay to a single
 * static gradient. Eliminates continuous GPU compositing cost from
 * filter: blur(80px) animations. The subtle gradient still gives depth
 * without competing with content.
 */
export default function NeuralBackground() {
  return (
    <div
      className="fixed inset-0 pointer-events-none z-0"
      aria-hidden="true"
      style={{
        background: `
          radial-gradient(ellipse 60% 50% at 15% 10%, rgba(59,130,246,0.07) 0%, transparent 60%),
          radial-gradient(ellipse 50% 40% at 85% 90%, rgba(6,182,212,0.04) 0%, transparent 60%)
        `,
      }}
    />
  );
}
