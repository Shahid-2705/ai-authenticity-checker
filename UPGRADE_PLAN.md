# ProofyX — "Neon Obsidian" Premium UI/UX Upgrade Plan

## Problem Analysis

The current "Arctic Obsidian" theme is functional but **visually flat**:
- Background grid at 0.03 opacity + 0.12 canvas opacity = practically invisible
- Monochromatic blue accent (#4B9EFF) used uniformly = bland
- No color gradients, no glow, no depth = feels like a skeleton theme
- Cards are simple flat boxes with dim borders = no dimensionality
- No ambient lighting effects = dead space around content
- Sidebar is purely functional = no visual character

## Design Philosophy: "Neon Obsidian Intelligence"

**Inspired by**: [Darktrace Cyber AI](https://darktrace.com), [Palantir Foundry](https://palantir.com), [CrowdStrike Falcon](https://crowdstrike.com), Linear App, Vercel Dashboard

**Core Principle**: A **living, breathing interface** that feels like a military-grade cyber intelligence platform — not a flat Bootstrap template.

### What We Add:
- **Dual-accent gradient system** — electric blue to violet creates depth and richness
- **Ambient glow effects** — cards and elements emit soft light halos
- **Particle field + grid mesh** — layered 3D background with floating particles above the grid
- **Glass morphism with depth** — cards have subtle inner glow and graduated backgrounds
- **Animated accent borders** — hover states reveal animated gradient borders
- **Color-rich risk system** — risks use saturated neon colors with pulsing indicators
- **Layered shadows** — multi-layer box-shadows create floating depth

---

## New Color System: "Neon Obsidian"

| Token | Value | Change |
|-------|-------|--------|
| `--bg-void` | `#06070A` | Deeper, richer black |
| `--bg-base` | `#0B0D12` | Slightly darker for more contrast |
| `--bg-card` | `#10131A` | Darker base to make glow pop |
| `--bg-elevated` | `#161A24` | More visible elevation |
| `--bg-inset` | `#080A0F` | Deep inset for contrast |
| `--accent` | `#6366F1` | Indigo-violet (richer than flat blue) |
| `--accent-2` | `#06B6D4` | Cyan secondary accent |
| `--accent-gradient` | `linear-gradient(135deg, #6366F1, #06B6D4)` | Signature gradient |
| `--accent-dim` | `rgba(99,102,241,0.12)` | Tinted backgrounds |
| `--accent-glow` | `rgba(99,102,241,0.25)` | Visible glow halos |
| `--accent-2-glow` | `rgba(6,182,212,0.20)` | Cyan glow |
| `--text-1` | `#F1F5F9` | Brighter primary text |
| `--text-2` | `#94A3B8` | Warmer secondary |
| `--text-3` | `#475569` | More visible muted text |
| `--risk-clear` | `#10B981` | Richer emerald |
| `--risk-caution` | `#F59E0B` | Warmer amber |
| `--risk-critical` | `#EF4444` | True red (more alarming) |
| `--border-dim` | `rgba(148,163,184,0.06)` | Slightly more visible |
| `--border-mid` | `rgba(148,163,184,0.12)` | Hover borders visible |
| `--border-glow` | `rgba(99,102,241,0.3)` | Glowing border on focus |

---

## 3D Background: Layered Particle Grid System

Replace the barely-visible wireframe with a **three-layer** depth system:

### Layer 1: Perspective Grid Floor (existing, enhanced)
- Increase opacity from 0.03 to 0.06
- Add a second color (cyan) for grid intersections
- Subtle pulse animation on grid lines

### Layer 2: Floating Particles
- 200-400 small Points floating slowly upward
- Size: 1-3px, color: accent gradient (indigo → cyan)
- Depth-based opacity (closer = brighter)
- Mouse repulsion effect (particles move away from cursor)

### Layer 3: Ambient Light Orbs
- 3-5 large, soft gradient spheres (like bokeh)
- Very low opacity (0.04-0.08)
- Slow drift animation
- Colors: accent, accent-2, risk-critical (subtle variety)

### Canvas opacity: 0.35 (up from 0.12)

---

## Card System: "Floating Glass"

### Base Card
```css
.card {
  background: linear-gradient(135deg, rgba(16,19,26,0.9), rgba(11,13,18,0.95));
  border: 1px solid rgba(148,163,184,0.06);
  border-radius: 12px;
  box-shadow:
    0 0 0 1px rgba(148,163,184,0.03),
    0 2px 8px rgba(0,0,0,0.4),
    0 8px 32px rgba(0,0,0,0.3),
    inset 0 1px 0 rgba(255,255,255,0.03);
  backdrop-filter: blur(8px);
}
```

### Hover Card (animated gradient border)
```css
.card-hover:hover {
  border-color: transparent;
  background-clip: padding-box;
  box-shadow:
    0 0 0 1px rgba(99,102,241,0.3),
    0 4px 16px rgba(99,102,241,0.1),
    0 8px 32px rgba(0,0,0,0.4),
    inset 0 1px 0 rgba(255,255,255,0.05);
  transform: translateY(-2px);
}
```

---

## Sidebar: "Command Dock"

- **Gradient accent line** on active nav (vertical left bar with gradient, not solid)
- **Glassmorphism** background: `backdrop-filter: blur(16px)` with semi-transparent bg
- **Icon glow** on active state: tiny accent halo behind active icon
- **Logo area**: Animated gradient text for "PROOFYX" brand
- **Status indicator**: Pulsing green dot with ring animation

---

## Typography Upgrade

| Role | Font | Notes |
|------|------|-------|
| Display (H1) | **Space Grotesk** | Geometric, futuristic, techy |
| Body | **Inter** | Keep for readability |
| Data/Code | **JetBrains Mono** | Keep for data values |
| Labels | **Inter** 600 | Uppercase, wide tracking |

Space Grotesk gives headings a distinctive, futuristic character without sacrificing readability.

---

## Micro-Interactions & Effects

### 1. Gradient Border Animation
Cards on hover get a rotating gradient border (conic-gradient animation):
```css
@keyframes borderRotate {
  0% { --border-angle: 0deg; }
  100% { --border-angle: 360deg; }
}
```

### 2. Glow Pulse on Risk Indicators
Critical alerts pulse with a subtle red glow:
```css
@keyframes riskPulse {
  0%, 100% { box-shadow: 0 0 4px var(--risk-critical); }
  50% { box-shadow: 0 0 12px var(--risk-critical); }
}
```

### 3. Data Count-Up Animation
Numbers animate from 0 to value on mount (already implemented, keep).

### 4. Staggered Card Entrance
Cards fade up with staggered delays (existing framer-motion, enhance easing).

### 5. Hover Lift + Glow
All interactive elements: translateY(-2px) + accent shadow on hover.

### 6. Loading Skeleton Shimmer
Gradient sweep animation on loading states (existing, enhance colors).

---

## Dashboard Enhancements

### Hero Section
- Add a **gradient orb** behind the title (large, soft, accent-colored blur)
- Animated typing effect for subtitle or threat count

### Stat Cards
- Each card gets a **colored top border** (2px gradient matching its theme)
- Subtle **inner glow** on the icon area
- Number uses **tabular-nums** for proper alignment

### Risk Distribution
- Replace flat bar with **glowing segmented bar**
- Each segment has a subtle inner glow matching its color
- Add percentage labels that appear on hover

### Activity Feed
- Left side: **colored vertical timeline line** (gradient from top to bottom)
- Each item has a **risk-colored dot** on the timeline
- Hover reveals full details with slide-right animation

---

## Page-Specific Enhancements

### Analysis Pages
- Upload zone: **Animated dashed border** (dash-array animation on SVG border)
- Progress: **Gradient progress bar** with glow effect
- Results: Cards with **accent-colored top accent line**

### Login/Signup
- **Floating orb** behind the form (animated gradient sphere)
- Input focus: **Glow border** animation
- Submit button: **Gradient background** with shimmer on hover

---

## Files to Modify

| # | File | Changes |
|---|------|---------|
| 1 | `tailwind.config.js` | New color tokens, gradient utilities, glow shadows |
| 2 | `index.css` | Updated design tokens, card glow classes, gradient borders |
| 3 | `index.html` | Add Space Grotesk font |
| 4 | `NeuralBackground.jsx` | Three-layer system: enhanced grid + particles + light orbs |
| 5 | `Sidebar.jsx` | Glassmorphism, gradient active bar, icon glow |
| 6 | `Layout.jsx` | Ambient gradient orbs in corners |
| 7 | `Dashboard.jsx` | Gradient orb hero, enhanced stat cards, timeline feed |
| 8 | `ImageAnalysis.jsx` | Animated upload border, glow progress |
| 9 | `VideoAnalysis.jsx` | Same pattern as image |
| 10 | `AudioAnalysis.jsx` | Same pattern as image |
| 11 | `Multimodal.jsx` | Same pattern |
| 12 | `History.jsx` | Timeline accent line, hover glow |
| 13 | `Login.jsx` | Floating orb, glow inputs |
| 14 | `Signup.jsx` | Same as login |
| 15 | `RiskGauge.jsx` | Glow filter on arc, pulsing critical |
| 16 | `ScoreBar.jsx` | Gradient fill with glow |
| 17 | `VerdictCard.jsx` | Top accent border, inner glow |
| 18 | `UploadZone.jsx` | Animated dashed border |
| 19 | `HeatmapViewer.jsx` | Enhanced contrast |

---

## Execution Order

1. **Foundation** — `index.html` (fonts) + `tailwind.config.js` + `index.css` (tokens + utilities)
2. **3D Background** — `NeuralBackground.jsx` (three-layer particle system)
3. **Shell** — `Layout.jsx` (ambient orbs) + `Sidebar.jsx` (glassmorphism)
4. **Dashboard** — Full enhancement with gradient orbs and timeline
5. **Shared Components** — UploadZone, RiskGauge, ScoreBar, VerdictCard
6. **Analysis Pages** — Image, Video, Audio, Multimodal
7. **Utility Pages** — History, Login, Signup
8. **Polish** — Final glow tuning, performance check

---

## Performance Safeguards

- Particles use `THREE.Points` (GPU instanced, not individual meshes)
- Glow effects use `box-shadow` (GPU composited) not `filter: blur()` where possible
- `backdrop-filter` only on sidebar (one element, not cards)
- `will-change: transform` on animated elements
- `prefers-reduced-motion` disables all particle effects and reduces animations
- Canvas renders at `Math.min(devicePixelRatio, 1.5)` to save GPU

---

## Inspiration Sources

- [Darktrace — Cyber AI Threat Visualizer](https://darktrace.com)
- [CrowdStrike Falcon — SOC Dashboard](https://crowdstrike.com)
- [Palantir Foundry — Data Intelligence](https://palantir.com)
- [Linear App — Premium Dark Dashboard](https://linear.app)
- [Three.js Globe Visualization](https://github.com/vasturiano/three-globe)
- [Cyberpunk 3D Earth](https://dev.to/dragonir/use-threejs-to-achieve-a-cool-cyberpunk-style-3d-digital-earth-screen-1fep)
- [Aufait UX — Cybersecurity Dashboard Guide](https://www.aufaitux.com/blog/cybersecurity-dashboard-ui-ux-design/)
