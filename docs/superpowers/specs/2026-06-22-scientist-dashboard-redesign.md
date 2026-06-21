# Scientist Dashboard Redesign

## Goal

Replace the decorative Mission Control layout with a scientist-first, three-mode dashboard showing real physics data: SoLEXS and HEL1OS light curves, MAG in-situ field, spectrograms, T/EM diagnostics, DNI, TNPSV phase space, and 5-class probabilistic forecast — all switchable between three workflow-oriented view modes.

## What Gets Removed

| Component | Reason |
|-----------|--------|
| `ArunaIntelligencePipeline` | Animated diagram, zero scientific information |
| `LLMBriefing` | AI narrative; scientists want raw data, not text summaries |
| `ThreatIntelligence` (from Mission Control) | Ops tool, stays on its own dedicated page |
| `LivePhysicsStream` | Five tiny unreadable sparklines; replaced by proper charts |
| `QuickActions` | Buttons collapse into header action bar |
| AI Agent diagram in sidebar | Decorative; removed from sidebar footer |

## Architecture

### View Mode System

A `useViewStore` Zustand store holds `mode: 'overview' | 'spectroscopy' | 'forecast'`. A `ViewModeBar` component (replaces the Pipeline strip row) renders three toggle pills. Keyboard shortcuts `1` / `2` / `3` switch modes globally via a `useEffect` listener in `MissionControl.tsx`.

Each mode renders a different set of panels inside the main content area. No routing changes — single page, conditional rendering on mode.

### Data Additions (`useMissionData.ts`)

New fields added to `MissionSnapshot`:

```typescript
// MAG instrument — interplanetary magnetic field (GSE, nT)
magBx: number[];      // length 360 (6h @ 1min cadence)
magBy: number[];
magBz: number[];
magBtotal: number[];

// DNI time series
dniHistory: number[];         // length 60 (60 min @ 1min)
dniThreshold: number;         // 1.5 — HOPE detection threshold

// 5-class forecast probabilities with MAPIE conformal intervals
forecastProbs: {
  horizon: '1h' | '3h' | '6h';
  B: number; C: number; M: number; X: number; Xplus: number;
  lower: number[]; upper: number[];  // [B_lo, C_lo, M_lo, X_lo, Xp_lo]
}[];

// Current flare phase label from L3 nowcasting
flarePhase: 'PRE-FLARE' | 'ONSET' | 'IMPULSIVE' | 'PEAK' | 'DECAY' | 'QUIET';
```

### New Components

All live in `web/src/components/scientist/`:

| File | What it renders |
|------|----------------|
| `ViewModeBar.tsx` | Three-pill switcher (OVERVIEW / SPECTROSCOPY / FORECAST) + keyboard hint |
| `SoLEXSLightCurve.tsx` | Full Plotly scatter, 6h window, log Y, per-channel toggle pills (SDD1/SDD2), GOES threshold lines, HOPE onset marker |
| `HEL1OSLightCurve.tsx` | HEL1OS stacked energy bands (10–25 keV / 25–50 keV / 50–150 keV), log Y, individual band toggles |
| `MAGPlot.tsx` | Three sub-panels stacked: Bz (with southward fill), By, |B|; 6h window; Bz=0 reference line |
| `DNIPlot.tsx` | DNI(t) line chart, dashed threshold at 1.5, HOPE zone shaded above threshold |
| `ForecastPanel.tsx` | Horizontal bar chart — 5 classes (B/C/M/X/X+), MAPIE error bars, horizon selector (1h/3h/6h) |
| `FlarePhaseStrip.tsx` | 5-node timeline (PRE-FLARE→ONSET→IMPULSIVE→PEAK→DECAY), current phase highlighted, replaces `NowcastingTimeline` |

### Updated Components

| File | Change |
|------|--------|
| `PhysicsEngine.tsx` | Rename to `SpectroscopyPanel.tsx`, add toggle pills for colorscale (linear/sqrt/log) and time window (30m/60m) |
| `MiniLightCurves.tsx` | Removed — replaced by full-size chart components |
| `AppSidebar.tsx` | Remove AI agent diagram from footer; keep nav links only |
| `HeroMetrics.tsx` | Add `flarePhase` badge; add Bz southward warning when `magBz` last value < −5 nT |
| `useMissionData.ts` | Add all new fields above |

---

## Three View Modes

### OVERVIEW (key: `1`)

```
┌─────────────────────────────────────────────────────────────┐
│ HeroMetrics (78px) — GOES class, T, EM, DNI, HOPE, Bz flag  │
├──────────────────────────────────┬──────────────────────────┤
│ SoLEXS Light Curve               │ MAG Plot                 │
│ 6h window, log Y                 │ Bz / By / |B|            │
│ Toggle: SDD1 SDD2 | GOES lines   │ 6h window, nT            │
│                                  │                          │
├──────────────────────────────────┼──────────────────────────┤
│ HEL1OS Light Curve               │ FlarePhaseStrip          │
│ 3 energy bands, log Y            │ 5-node phase timeline    │
│ Toggle: 10–25 / 25–50 / 50–150  │ + Alert history 24h      │
└──────────────────────────────────┴──────────────────────────┘
```

### SPECTROSCOPY (key: `2`)

```
┌──────────────────────────┬──────────────────────────────────┐
│ SoLEXS Spectrogram       │ HEL1OS Spectrogram               │
│ 60-min window            │ 60-min window                    │
│ Toggle: colorscale       │ Toggle: colorscale               │
│ Energy: 1–7 keV          │ Energy: 10–150 keV               │
├──────────────┬───────────┴──────┬───────────────────────────┤
│ T(t) + EM(t) │ TNPSV Phase Space│ DNI Time Series           │
│ 1σ bands     │ log-log, 200 pts │ threshold line at 1.5     │
│ dual Y-axis  │ HOPE zone        │ HOPE zone shaded           │
└──────────────┴──────────────────┴───────────────────────────┘
```

### FORECAST (key: `3`)

```
┌───────────────────────────────┬─────────────────────────────┐
│ 5-Class Forecast              │ TNPSV Phase Space           │
│ B / C / M / X / X+           │ (same as SPECTROSCOPY)       │
│ Horizon: [1h] [3h] [6h]      │                              │
│ MAPIE conformal error bars    │                              │
├───────────────────────────────┼─────────────────────────────┤
│ MAG Bz + Bz southward flag   │ Historical Analog           │
│ 6h window                    │ Top match: X6.3 Feb 2024     │
│                               │ Similarity: 92%             │
│                               │ Analog light curve overlay  │
└───────────────────────────────┴─────────────────────────────┘
```

---

## Chart Toggle Pills

Each chart panel has a row of small pill buttons in its header:

```tsx
<TogglePill active={showSDD1} onClick={() => setShowSDD1(v => !v)}>SDD1</TogglePill>
<TogglePill active={showSDD2} onClick={() => setShowSDD2(v => !v)}>SDD2</TogglePill>
```

Pill style: `height: 18px, fontSize: 9px, border-radius: 3px`, active = `background: rgba(244,168,37,0.15), border: rgba(244,168,37,0.5)`, inactive = `background: transparent, border: rgba(255,244,232,0.10)`.

Toggle state lives in local `useState` inside each chart component — no global store needed.

---

## Mock Data (useMissionData.ts)

**MAG**: Bz sinusoidal with southward excursion, mimicking a typical IMF rotation:
```typescript
magBz: Array.from({ length: 360 }, (_, i) =>
  -8 * Math.sin((i / 360) * 2 * Math.PI + Math.PI * 0.3) + gaussianNoise(1.5)
),
```

**DNI**: Ramps from 0 → 1.8 over the 60-min window, crossing threshold at ~40 min:
```typescript
dniHistory: Array.from({ length: 60 }, (_, i) => {
  const base = 0.1 + (i / 60) * 1.9;
  return Math.max(0, base + gaussianNoise(0.06));
}),
```

**Forecast probs**: Reference values from X6.3 precursor phase — P(M) dominant:
```typescript
forecastProbs: ['1h', '3h', '6h'].map(horizon => ({
  horizon, B: 0.02, C: 0.08, M: 0.54, X: 0.28, Xplus: 0.08,
  lower: [0.01, 0.05, 0.44, 0.20, 0.04],
  upper: [0.04, 0.13, 0.64, 0.36, 0.14],
})),
```

---

## File Changes Summary

**Create:**
- `web/src/stores/useViewStore.ts`
- `web/src/components/scientist/ViewModeBar.tsx`
- `web/src/components/scientist/SoLEXSLightCurve.tsx`
- `web/src/components/scientist/HEL1OSLightCurve.tsx`
- `web/src/components/scientist/MAGPlot.tsx`
- `web/src/components/scientist/DNIPlot.tsx`
- `web/src/components/scientist/ForecastPanel.tsx`
- `web/src/components/scientist/FlarePhaseStrip.tsx`
- `web/src/components/scientist/HistoricalAnalogPanel.tsx`
- `web/src/components/scientist/SpectroscopyPanel.tsx` (from PhysicsEngine.tsx)

**Modify:**
- `web/src/hooks/useMissionData.ts` — add new fields
- `web/src/pages/MissionControl.tsx` — three-mode conditional render
- `web/src/components/mission-control/AppSidebar.tsx` — remove AI agent diagram
- `web/src/components/mission-control/HeroMetrics.tsx` — add flarePhase + Bz flag
- `web/src/components/mission-control/PhysicsEngine.tsx` — keep as reference, replaced by SpectroscopyPanel

**Delete (rendered unused):**
- `ArunaIntelligencePipeline.tsx`
- `LLMBriefing.tsx`
- `LivePhysicsStream.tsx`
- `QuickActions.tsx`
- `ThreatIntelligence.tsx` (from Mission Control imports only — file stays for the dedicated Threat page)

---

## Constraints

- All mock data must match X6.3 (2024-02-22) reference values from `CLAUDE.md`
- `--legacy-peer-deps` required for all npm installs
- No new npm packages — use existing Plotly.js + Zustand + Tailwind + Framer Motion
- Plotly charts: `displayModeBar: false`, `responsive: true`, `useResizeHandler`
- No Streamlit, no Next.js — React + Vite only
- MAG data is mock (in-situ simulation) — real PRADAN wire-up is future work
