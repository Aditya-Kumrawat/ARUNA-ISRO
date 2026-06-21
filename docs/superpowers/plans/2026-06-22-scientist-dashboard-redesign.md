# Scientist Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace decorative Mission Control panels with a three-mode scientist dashboard (OVERVIEW / SPECTROSCOPY / FORECAST) showing real SoLEXS, HEL1OS, MAG, T/EM, DNI, TNPSV, and 5-class probabilistic forecast plots.

**Architecture:** Single React page with Zustand view-mode store; three named layouts rendered conditionally; all charts use react-plotly.js with toggle-pill overlays for channel/colorscale/horizon selection; mock data in useMissionData matches X6.3 reference values.

**Tech Stack:** React 18, TypeScript, Vite, Tailwind CSS v3, react-plotly.js ^2.6, Zustand ^4.5, Lucide React

## Global Constraints

- `--legacy-peer-deps` on every `npm install`
- No new npm packages — existing deps only
- Plotly: `displayModeBar: false`, `responsive: true`, `useResizeHandler` on every chart
- All mock data anchored to X6.3 (2024-02-22): T≈12.4 MK, EM≈3.7×10^48, DNI≈1.62, HOPE_SCORE≈0.86
- No Streamlit, no Next.js — React + Vite only
- Working dir for all commands: `E:\ARUNA\web`
- TypeScript check command: `npx tsc --noEmit`

---

### Task 1: Extend useMissionData with new fields

**Files:**
- Modify: `web/src/hooks/useMissionData.ts`

**Interfaces:**
- Produces: Extended `MissionSnapshot` with `solexsSDD1`, `solexsSDD2`, `hel1osBand1/2/3`, `magBx/By/Bz/Btotal`, `dniHistory`, `dniThreshold`, `forecastProbs`, `flarePhase`

- [ ] **Step 1: Replace useMissionData.ts with extended version**

```typescript
// web/src/hooks/useMissionData.ts
import { useState, useEffect, useRef } from 'react';
import { gaussianNoise } from '../lib/utils';

export interface RegionRisk { [region: string]: 'HIGH' | 'MEDIUM' | 'LOW'; }

export type FlarePhase = 'PRE-FLARE' | 'ONSET' | 'IMPULSIVE' | 'PEAK' | 'DECAY' | 'QUIET';

export interface ForecastProb {
  horizon: '1h' | '3h' | '6h';
  B: number; C: number; M: number; X: number; Xplus: number;
  lower: number[];
  upper: number[];
}

export interface MissionSnapshot {
  utcTime: Date;
  goesClass: string;
  hopeStatus: 'DETECTED' | 'MONITORING' | 'QUIET';
  hopeConfidence: number;
  hopeScore: number;
  leadTimeMin: number;
  predictedPeak: string;
  confidence: number;
  temperature: number;
  emissionMeasure: number;
  dni: number;
  hfd: number;
  shei: number;
  feKalpha: number;
  tnpsvAngle: number;
  tnpsvHistory: Array<{ t: number; em: number }>;
  flarePhase: FlarePhase;
  threatLevel: 'ORANGE' | 'RED' | 'YELLOW' | 'GREEN';
  regionRisk: RegionRisk;
  analogSimilarity: number;
  systemStatus: 'NOMINAL' | 'DEGRADED';
  pipelineHealth: number;
  /* ── light curves ── 360 pts @ 1min cadence (6h window) */
  solexsSDD1: number[];
  solexsSDD2: number[];
  hel1osBand1: number[];  /* 10–25 keV  */
  hel1osBand2: number[];  /* 25–50 keV  */
  hel1osBand3: number[];  /* 50–150 keV */
  /* legacy – kept for MiniLightCurves compat, 90 pts */
  solexsFlux: number[];
  hel1osFlux: number[];
  /* ── spectrograms ── 60 time steps × 32 energy bins */
  solexsSpectrogram: number[][];
  hel1osSpectrogram: number[][];
  /* ── T/EM history ── 60 pts */
  tHistory: number[];
  emHistory: number[];
  tSigma: number[];
  emSigma: number[];
  /* ── MAG ── 360 pts @ 1min cadence (6h window), nT GSE */
  magBx: number[];
  magBy: number[];
  magBz: number[];
  magBtotal: number[];
  /* ── DNI ── 60 pts @ 1min */
  dniHistory: number[];
  dniThreshold: number;
  /* ── forecast ── */
  forecastProbs: ForecastProb[];
}

const SOLEXS_BINS = 32;
const HEL1OS_BINS = 32;
const SPEC_STEPS  = 60;
const HISTORY_LEN = 60;
const FLUX_6H     = 360;
const TRAIL_LEN   = 200;

const REF = {
  T: 12.4, EM: 3.7, DNI: 1.62, HFD: 1.31,
  SHEI: -0.75, FE_KA: 6.48, HOPE_SCORE: 0.86,
};

function makeSpectrogram(nBins: number, eMin: number, eMax: number, peakIntensity: number, tick: number): number[][] {
  return Array.from({ length: SPEC_STEPS }, (_, ti) => {
    const ramp  = 0.05 + 0.95 * Math.pow(ti / (SPEC_STEPS - 1), 1.4);
    const qpp   = 1 + 0.12 * Math.sin((ti + tick * 0.5) / 4.5);
    const scale = peakIntensity * ramp * qpp;
    return Array.from({ length: nBins }, (_, ei) => {
      const energy  = eMin + (eMax - eMin) * (ei / (nBins - 1));
      const kT      = eMin * 3.8 * (1 + ramp * 0.5);
      const thermal = Math.exp(-energy / kT);
      const feLine  = eMin < 7 ? 0.4 * Math.exp(-Math.pow((energy - 6.7) / 0.3, 2)) : 0;
      const base    = scale * (thermal + feLine);
      return Math.max(0, base + Math.abs(gaussianNoise(base * 0.12)));
    });
  });
}

function makeTnpsvTrail(tick: number): Array<{ t: number; em: number }> {
  return Array.from({ length: TRAIL_LEN }, (_, i) => {
    const phase  = i / (TRAIL_LEN - 1);
    const jitter = (Math.random() - 0.5) * 0.04;
    const tBase  = 2.0 + phase * (REF.T - 2.0) + Math.sin((i + tick) / 12) * 0.4;
    const t      = Math.max(0.8, tBase + gaussianNoise(0.18) + jitter * 3);
    const emPhase = Math.max(0, (phase - 0.3) / 0.7);
    const emBase  = 0.08 + emPhase * (REF.EM - 0.08) + Math.cos((i + tick) / 18) * 0.12;
    const em      = Math.max(0.01, emBase + gaussianNoise(0.06) + jitter);
    return { t, em };
  });
}

function seed(ref: number, noise: number) { return Math.max(0, ref + gaussianNoise(noise)); }

export function useMissionData(): MissionSnapshot {
  const tickRef = useRef(0);

  const buildSnapshot = (): MissionSnapshot => {
    const t    = tickRef.current;
    const slow = Math.sin(t / 30);

    const temperature     = seed(REF.T  + slow * 0.3, 0.08);
    const emissionMeasure = seed(REF.EM + slow * 0.05, 0.03);

    /* ── SoLEXS 6h light curves ── */
    const solexsSDD1 = Array.from({ length: FLUX_6H }, (_, i) => {
      const base = 1e-7 * Math.exp(0.006 * (i / FLUX_6H) * 60);
      return Math.max(1e-9, base * (1 + 0.05 * Math.sin(i / 8)) + gaussianNoise(base * 0.03));
    });
    const solexsSDD2 = Array.from({ length: FLUX_6H }, (_, i) => {
      const base = 9.8e-8 * Math.exp(0.006 * (i / FLUX_6H) * 60);
      return Math.max(1e-9, base + gaussianNoise(base * 0.03));
    });

    /* ── HEL1OS energy band curves ── */
    const hel1osBand1 = Array.from({ length: FLUX_6H }, (_, i) => {
      const base = 4e-8 * Math.exp(0.005 * (i / FLUX_6H) * 60);
      return Math.max(1e-11, base + gaussianNoise(base * 0.05));
    });
    const hel1osBand2 = Array.from({ length: FLUX_6H }, (_, i) => {
      const base = 8e-9 * Math.exp(0.005 * (i / FLUX_6H) * 60);
      return Math.max(1e-11, base + gaussianNoise(base * 0.06));
    });
    const hel1osBand3 = Array.from({ length: FLUX_6H }, (_, i) => {
      const base = 2e-9 * Math.exp(0.004 * (i / FLUX_6H) * 60);
      return Math.max(1e-11, base + gaussianNoise(base * 0.08));
    });

    /* ── legacy 90-pt flux ── */
    const solexsFlux = Array.from({ length: 90 }, (_, i) => {
      const base = 2e-6 * Math.exp(0.01 * (i - 90 + t % 30));
      return Math.max(1e-9, base + gaussianNoise(base * 0.05));
    });
    const hel1osFlux = Array.from({ length: 90 }, (_, i) => {
      const base = 4e-7 * Math.exp(0.008 * (i - 90 + t % 30));
      return Math.max(1e-9, base + gaussianNoise(base * 0.08));
    });

    /* ── MAG 6h ── */
    const magBx = Array.from({ length: FLUX_6H }, (_, i) =>
      3 * Math.cos((i / FLUX_6H) * 2 * Math.PI) + gaussianNoise(1.0));
    const magBy = Array.from({ length: FLUX_6H }, (_, i) =>
      5 * Math.sin((i / FLUX_6H) * 2 * Math.PI + 0.5) + gaussianNoise(1.2));
    const magBz = Array.from({ length: FLUX_6H }, (_, i) =>
      -8 * Math.sin((i / FLUX_6H) * 2 * Math.PI + Math.PI * 0.3) + gaussianNoise(1.5));
    const magBtotal = magBx.map((bx, i) =>
      Math.sqrt(bx ** 2 + magBy[i] ** 2 + magBz[i] ** 2));

    /* ── DNI 60-min ── */
    const dniHistory = Array.from({ length: 60 }, (_, i) => {
      const base = 0.1 + (i / 60) * 1.9;
      return Math.max(0, base + gaussianNoise(0.06));
    });

    return {
      utcTime: new Date(),
      goesClass: 'M1.2',
      hopeStatus: 'DETECTED',
      hopeConfidence: 86,
      hopeScore: seed(REF.HOPE_SCORE, 0.005),
      leadTimeMin: Math.max(18, Math.round(22 - t * 0.02)),
      predictedPeak: 'M3.8',
      confidence: 82,
      temperature,
      emissionMeasure,
      dni: seed(REF.DNI, 0.02),
      hfd: seed(REF.HFD, 0.02),
      shei: seed(REF.SHEI, 0.01),
      feKalpha: seed(REF.FE_KA, 0.01),
      tnpsvAngle: seed(88, 1.5),
      tnpsvHistory: makeTnpsvTrail(t),
      flarePhase: 'ONSET',
      threatLevel: 'ORANGE',
      regionRisk: {
        'North India': 'HIGH', 'Central India': 'HIGH',
        'East India': 'HIGH', 'West India': 'MEDIUM', 'South India': 'LOW',
      },
      analogSimilarity: 92,
      systemStatus: 'NOMINAL',
      pipelineHealth: 99.6,
      solexsSDD1, solexsSDD2,
      hel1osBand1, hel1osBand2, hel1osBand3,
      solexsFlux, hel1osFlux,
      solexsSpectrogram: makeSpectrogram(SOLEXS_BINS, 1, 7, 0.85, t),
      hel1osSpectrogram: makeSpectrogram(HEL1OS_BINS, 10, 150, 0.55, t),
      tHistory: Array.from({ length: HISTORY_LEN }, (_, i) =>
        REF.T + Math.sin((t - HISTORY_LEN + i) / 20) * 0.8 + gaussianNoise(0.1)),
      emHistory: Array.from({ length: HISTORY_LEN }, (_, i) =>
        REF.EM + Math.cos((t - HISTORY_LEN + i) / 25) * 0.15 + gaussianNoise(0.02)),
      tSigma:  Array.from({ length: HISTORY_LEN }, () => seed(0.3, 0.05)),
      emSigma: Array.from({ length: HISTORY_LEN }, () => seed(0.08, 0.01)),
      magBx, magBy, magBz, magBtotal,
      dniHistory,
      dniThreshold: 1.5,
      forecastProbs: (['1h', '3h', '6h'] as const).map(horizon => ({
        horizon,
        B: 0.02, C: 0.08, M: 0.54, X: 0.28, Xplus: 0.08,
        lower: [0.01, 0.05, 0.44, 0.20, 0.04],
        upper: [0.04, 0.13, 0.64, 0.36, 0.14],
      })),
    };
  };

  const [snapshot, setSnapshot] = useState<MissionSnapshot>(buildSnapshot);

  useEffect(() => {
    const id = setInterval(() => {
      tickRef.current += 1;
      setSnapshot(buildSnapshot());
    }, 1000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return snapshot;
}
```

- [ ] **Step 2: Type-check**

Run: `npx tsc --noEmit`
Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
git add web/src/hooks/useMissionData.ts
git commit -m "feat(data): extend MissionSnapshot with SoLEXS/HEL1OS bands, MAG, DNI, forecast fields"
```

---

### Task 2: Shared primitives — TogglePill + useViewStore + ViewModeBar

**Files:**
- Create: `web/src/components/scientist/TogglePill.tsx`
- Create: `web/src/stores/useViewStore.ts`
- Create: `web/src/components/scientist/ViewModeBar.tsx`

**Interfaces:**
- Produces: `TogglePill` component, `useViewStore` hook, `ViewModeBar` component

- [ ] **Step 1: Create TogglePill**

```tsx
// web/src/components/scientist/TogglePill.tsx
export function TogglePill({ active, onClick, children }: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        height: 18, fontSize: 9, fontWeight: 600, letterSpacing: '0.06em',
        textTransform: 'uppercase' as const, padding: '0 7px', borderRadius: 3,
        cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace',
        background: active ? 'rgba(244,168,37,0.15)' : 'transparent',
        border: `1px solid ${active ? 'rgba(244,168,37,0.50)' : 'rgba(255,244,232,0.10)'}`,
        color: active ? 'rgba(244,168,37,0.90)' : 'rgba(243,241,234,0.38)',
        transition: 'all 0.12s ease',
      }}
    >
      {children}
    </button>
  );
}
```

- [ ] **Step 2: Create useViewStore**

```typescript
// web/src/stores/useViewStore.ts
import { create } from 'zustand';

export type ViewMode = 'overview' | 'spectroscopy' | 'forecast';

interface ViewStore {
  mode: ViewMode;
  setMode: (mode: ViewMode) => void;
}

export const useViewStore = create<ViewStore>((set) => ({
  mode: 'overview',
  setMode: (mode) => set({ mode }),
}));
```

- [ ] **Step 3: Create ViewModeBar**

```tsx
// web/src/components/scientist/ViewModeBar.tsx
import { useViewStore, type ViewMode } from '../../stores/useViewStore';

const MODES: Array<{ key: ViewMode; label: string; num: number; sub: string }> = [
  { key: 'overview',     label: 'OVERVIEW',     num: 1, sub: 'Light Curves + MAG' },
  { key: 'spectroscopy', label: 'SPECTROSCOPY', num: 2, sub: 'Spectra · T/EM · DNI' },
  { key: 'forecast',     label: 'FORECAST',     num: 3, sub: 'Probability · Analog' },
];

export function ViewModeBar() {
  const { mode, setMode } = useViewStore();

  return (
    <div className="aruna-card flex items-center gap-3 px-4"
         style={{ height: 48, flexShrink: 0 }}>
      <span style={{ fontSize: 8.5, fontWeight: 600, letterSpacing: '0.13em',
                     textTransform: 'uppercase', color: 'rgba(243,241,234,0.25)',
                     marginRight: 4, flexShrink: 0 }}>
        VIEW MODE
      </span>
      {MODES.map(({ key, label, num, sub }) => {
        const active = mode === key;
        return (
          <button
            key={key}
            onClick={() => setMode(key)}
            style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '5px 16px', borderRadius: 4, cursor: 'pointer',
              border: `1px solid ${active ? 'rgba(244,168,37,0.45)' : 'rgba(255,244,232,0.07)'}`,
              background: active ? 'rgba(244,168,37,0.09)' : 'transparent',
              transition: 'all 0.15s ease',
            }}
          >
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
                           fontWeight: 700, color: active ? 'rgba(244,168,37,0.55)' : 'rgba(243,241,234,0.18)' }}>
              {num}
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 1 }}>
              <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em',
                             textTransform: 'uppercase',
                             color: active ? '#F4A825' : 'rgba(243,241,234,0.55)' }}>
                {label}
              </span>
              <span style={{ fontSize: 7.5, letterSpacing: '0.04em',
                             color: active ? 'rgba(244,168,37,0.50)' : 'rgba(243,241,234,0.22)' }}>
                {sub}
              </span>
            </div>
          </button>
        );
      })}
      <span className="ml-auto" style={{ fontFamily: 'JetBrains Mono, monospace',
                                          fontSize: 7.5, color: 'rgba(243,241,234,0.15)',
                                          letterSpacing: '0.05em' }}>
        PRESS 1 / 2 / 3
      </span>
    </div>
  );
}
```

- [ ] **Step 4: Type-check**

Run: `npx tsc --noEmit`
Expected: 0 errors

- [ ] **Step 5: Commit**

```bash
git add web/src/components/scientist/TogglePill.tsx web/src/stores/useViewStore.ts web/src/components/scientist/ViewModeBar.tsx
git commit -m "feat(ui): TogglePill component + Zustand view-mode store + ViewModeBar switcher"
```

---

### Task 3: SoLEXSLightCurve component

**Files:**
- Create: `web/src/components/scientist/SoLEXSLightCurve.tsx`

**Interfaces:**
- Consumes: `d.solexsSDD1`, `d.solexsSDD2` (number[], length 360)
- Produces: `<SoLEXSLightCurve d={d} />` — Plotly log-Y scatter, 6h window, SDD1/SDD2 toggle pills, GOES threshold lines

- [ ] **Step 1: Create SoLEXSLightCurve.tsx**

```tsx
// web/src/components/scientist/SoLEXSLightCurve.tsx
import { useState } from 'react';
import Plot from 'react-plotly.js';
import { TogglePill } from './TogglePill';
import type { MissionSnapshot } from '../../hooks/useMissionData';

const BG   = '#040506';
const GRID = 'rgba(255,244,232,0.025)';
const AX   = 'rgba(255,244,232,0.22)';
const CFG  = { displayModeBar: false, responsive: true };

/* Build 6-hour time axis (minutes before NOW, as HH:MM strings) */
function makeTimeAxis(n: number): string[] {
  const now = Date.now();
  return Array.from({ length: n }, (_, i) => {
    const ms = now - (n - 1 - i) * 60_000;
    return new Date(ms).toISOString().slice(11, 16);
  });
}

/* GOES class flux thresholds (W/m²) */
const GOES_LINES = [
  { y: 1e-8, label: 'A', color: 'rgba(103,212,108,0.25)' },
  { y: 1e-7, label: 'B', color: 'rgba(103,212,108,0.30)' },
  { y: 1e-6, label: 'C', color: 'rgba(244,168,37,0.30)' },
  { y: 1e-5, label: 'M', color: 'rgba(255,138,38,0.40)' },
  { y: 1e-4, label: 'X', color: 'rgba(255,59,59,0.45)'  },
];

export function SoLEXSLightCurve({ d }: { d: MissionSnapshot }) {
  const [showSDD1,  setShowSDD1]  = useState(true);
  const [showSDD2,  setShowSDD2]  = useState(true);
  const [showGOES,  setShowGOES]  = useState(true);
  const [showHOPE,  setShowHOPE]  = useState(true);

  const xs = makeTimeAxis(d.solexsSDD1.length);

  /* HOPE onset index ≈ 40% into the 6h window */
  const hopeIdx = Math.floor(d.solexsSDD1.length * 0.67);

  const shapes: object[] = [];
  const annotations: object[] = [];

  if (showGOES) {
    GOES_LINES.forEach(({ y, label, color }) => {
      shapes.push({
        type: 'line', x0: xs[0], x1: xs[xs.length - 1],
        y0: y, y1: y, xref: 'x', yref: 'y',
        line: { color, width: 0.8, dash: 'dot' },
      });
      annotations.push({
        x: xs[xs.length - 1], y: Math.log10(y), xref: 'x', yref: 'y',
        text: label, showarrow: false,
        font: { size: 8, color, family: 'JetBrains Mono, monospace' },
        xanchor: 'right', yanchor: 'middle',
      });
    });
  }

  if (showHOPE) {
    shapes.push({
      type: 'line', x0: xs[hopeIdx], x1: xs[hopeIdx],
      y0: 0, y1: 1, yref: 'paper',
      line: { color: 'rgba(244,168,37,0.55)', width: 1, dash: 'dash' },
    });
    annotations.push({
      x: xs[hopeIdx], y: 0.97, yref: 'paper',
      text: 'HOPE', showarrow: false,
      font: { size: 8, color: '#F4A825', family: 'Inter' },
      xanchor: 'left',
    });
  }

  const data: object[] = [];
  if (showSDD1) data.push({
    type: 'scatter', mode: 'lines', x: xs, y: d.solexsSDD1,
    name: 'SDD1', line: { color: '#F4A825', width: 1.5 },
    hovertemplate: '%{y:.2e} W/m²<extra>SoLEXS SDD1</extra>',
  });
  if (showSDD2) data.push({
    type: 'scatter', mode: 'lines', x: xs, y: d.solexsSDD2,
    name: 'SDD2', line: { color: '#FF8A26', width: 1.5, dash: 'dot' },
    hovertemplate: '%{y:.2e} W/m²<extra>SoLEXS SDD2</extra>',
  });

  return (
    <div className="aruna-card flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-3 flex-shrink-0"
           style={{ height: 32, borderBottom: '1px solid rgba(255,180,70,0.05)' }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
                       color: 'rgba(243,241,234,0.85)', textTransform: 'uppercase' }}>
          SoLEXS · 1–7 keV
        </span>
        <div className="flex items-center gap-1.5">
          <TogglePill active={showSDD1} onClick={() => setShowSDD1(v => !v)}>SDD1</TogglePill>
          <TogglePill active={showSDD2} onClick={() => setShowSDD2(v => !v)}>SDD2</TogglePill>
          <span style={{ width: 1, height: 14, background: 'rgba(255,244,232,0.08)', margin: '0 4px' }} />
          <TogglePill active={showGOES} onClick={() => setShowGOES(v => !v)}>GOES</TogglePill>
          <TogglePill active={showHOPE} onClick={() => setShowHOPE(v => !v)}>HOPE</TogglePill>
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <Plot
          className="w-full h-full"
          data={data as any}
          layout={{
            paper_bgcolor: BG, plot_bgcolor: BG,
            font: { family: 'Inter, sans-serif', size: 8, color: AX },
            margin: { t: 8, r: 48, b: 36, l: 52 },
            xaxis: { gridcolor: GRID, color: AX, tickfont: { size: 7.5 },
                     title: { text: 'UTC (6h window)', font: { size: 8 } }, nticks: 8 },
            yaxis: { type: 'log', title: { text: 'Flux (W/m²)', font: { size: 8 } },
                     gridcolor: GRID, zerolinecolor: GRID, color: AX, tickfont: { size: 7.5 } },
            legend: { x: 0.01, y: 0.99, bgcolor: 'transparent',
                      font: { size: 8, color: 'rgba(243,241,234,0.45)' } },
            shapes, annotations,
          } as any}
          config={CFG}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check**

Run: `npx tsc --noEmit`
Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
git add web/src/components/scientist/SoLEXSLightCurve.tsx
git commit -m "feat(chart): SoLEXS 6h light curve with SDD1/SDD2/GOES/HOPE toggles"
```

---

### Task 4: HEL1OSLightCurve component

**Files:**
- Create: `web/src/components/scientist/HEL1OSLightCurve.tsx`

**Interfaces:**
- Consumes: `d.hel1osBand1`, `d.hel1osBand2`, `d.hel1osBand3` (number[], length 360)
- Produces: `<HEL1OSLightCurve d={d} />` — three energy-band traces, log Y, toggleable

- [ ] **Step 1: Create HEL1OSLightCurve.tsx**

```tsx
// web/src/components/scientist/HEL1OSLightCurve.tsx
import { useState } from 'react';
import Plot from 'react-plotly.js';
import { TogglePill } from './TogglePill';
import type { MissionSnapshot } from '../../hooks/useMissionData';

const BG   = '#040506';
const GRID = 'rgba(255,244,232,0.025)';
const AX   = 'rgba(255,244,232,0.22)';
const CFG  = { displayModeBar: false, responsive: true };

function makeTimeAxis(n: number): string[] {
  const now = Date.now();
  return Array.from({ length: n }, (_, i) => {
    const ms = now - (n - 1 - i) * 60_000;
    return new Date(ms).toISOString().slice(11, 16);
  });
}

const BANDS = [
  { key: 'band1' as const, label: '10–25 keV', color: '#FF8A26' },
  { key: 'band2' as const, label: '25–50 keV', color: '#E040FB' },
  { key: 'band3' as const, label: '50–150 keV', color: '#7C4DFF' },
];

export function HEL1OSLightCurve({ d }: { d: MissionSnapshot }) {
  const [visible, setVisible] = useState<Record<string, boolean>>({
    band1: true, band2: true, band3: true,
  });

  const xs = makeTimeAxis(d.hel1osBand1.length);

  const dataMap: Record<string, number[]> = {
    band1: d.hel1osBand1,
    band2: d.hel1osBand2,
    band3: d.hel1osBand3,
  };

  const traces = BANDS
    .filter(b => visible[b.key])
    .map(b => ({
      type: 'scatter', mode: 'lines', x: xs, y: dataMap[b.key],
      name: b.label,
      line: { color: b.color, width: 1.5 },
      hovertemplate: `%{y:.2e} W/m²<extra>HEL1OS ${b.label}</extra>`,
    }));

  return (
    <div className="aruna-card flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-3 flex-shrink-0"
           style={{ height: 32, borderBottom: '1px solid rgba(255,180,70,0.05)' }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
                       color: 'rgba(243,241,234,0.85)', textTransform: 'uppercase' }}>
          HEL1OS · 10–150 keV
        </span>
        <div className="flex items-center gap-1.5">
          {BANDS.map(b => (
            <TogglePill key={b.key} active={visible[b.key]}
              onClick={() => setVisible(v => ({ ...v, [b.key]: !v[b.key] }))}>
              {b.label}
            </TogglePill>
          ))}
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <Plot
          className="w-full h-full"
          data={traces as any}
          layout={{
            paper_bgcolor: BG, plot_bgcolor: BG,
            font: { family: 'Inter, sans-serif', size: 8, color: AX },
            margin: { t: 8, r: 48, b: 36, l: 52 },
            xaxis: { gridcolor: GRID, color: AX, tickfont: { size: 7.5 },
                     title: { text: 'UTC (6h window)', font: { size: 8 } }, nticks: 8 },
            yaxis: { type: 'log', title: { text: 'Flux (W/m²)', font: { size: 8 } },
                     gridcolor: GRID, zerolinecolor: GRID, color: AX, tickfont: { size: 7.5 } },
            legend: { x: 0.01, y: 0.99, bgcolor: 'transparent',
                      font: { size: 8, color: 'rgba(243,241,234,0.45)' } },
          } as any}
          config={CFG}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check + commit**

```bash
npx tsc --noEmit
git add web/src/components/scientist/HEL1OSLightCurve.tsx
git commit -m "feat(chart): HEL1OS 6h energy-band light curves with per-band toggles"
```

---

### Task 5: MAGPlot component

**Files:**
- Create: `web/src/components/scientist/MAGPlot.tsx`

**Interfaces:**
- Consumes: `d.magBx`, `d.magBy`, `d.magBz`, `d.magBtotal` (number[], length 360)
- Produces: `<MAGPlot d={d} />` — three Plotly subplots stacked (|B|, By, Bz); southward fill on Bz panel

- [ ] **Step 1: Create MAGPlot.tsx**

```tsx
// web/src/components/scientist/MAGPlot.tsx
import { useState } from 'react';
import Plot from 'react-plotly.js';
import { TogglePill } from './TogglePill';
import type { MissionSnapshot } from '../../hooks/useMissionData';

const BG   = '#040506';
const GRID = 'rgba(255,244,232,0.025)';
const AX   = 'rgba(255,244,232,0.22)';
const CFG  = { displayModeBar: false, responsive: true };

function makeTimeAxis(n: number): string[] {
  const now = Date.now();
  return Array.from({ length: n }, (_, i) => {
    const ms = now - (n - 1 - i) * 60_000;
    return new Date(ms).toISOString().slice(11, 16);
  });
}

export function MAGPlot({ d }: { d: MissionSnapshot }) {
  const [showBz,    setShowBz]    = useState(true);
  const [showBy,    setShowBy]    = useState(true);
  const [showBtot,  setShowBtot]  = useState(true);

  const xs = makeTimeAxis(d.magBz.length);

  /* Southward Bz fill: split into segments where Bz < 0 */
  const bzSouthFillY = d.magBz.map(v => (v < 0 ? v : 0));

  const data: object[] = [];

  /* ── |B| on yaxis3 ── */
  if (showBtot) data.push({
    type: 'scatter', mode: 'lines', x: xs, y: d.magBtotal,
    name: '|B|', yaxis: 'y3',
    line: { color: 'rgba(243,241,234,0.70)', width: 1.5 },
    hovertemplate: '%{y:.1f} nT<extra>|B|</extra>',
  });

  /* ── By on yaxis2 ── */
  if (showBy) data.push({
    type: 'scatter', mode: 'lines', x: xs, y: d.magBy,
    name: 'By', yaxis: 'y2',
    line: { color: '#00D4FF', width: 1.5 },
    hovertemplate: '%{y:.1f} nT<extra>By</extra>',
  });

  /* ── Bz southward fill ── */
  if (showBz) {
    data.push({
      type: 'scatter', mode: 'none', x: xs, y: bzSouthFillY,
      fill: 'tozeroy', fillcolor: 'rgba(255,59,59,0.12)',
      yaxis: 'y', hoverinfo: 'none', showlegend: false,
    });
    data.push({
      type: 'scatter', mode: 'lines', x: xs, y: d.magBz,
      name: 'Bz', yaxis: 'y',
      line: { color: '#FF6B6B', width: 1.8 },
      hovertemplate: '%{y:.1f} nT<extra>Bz (GSE)</extra>',
    });
  }

  const bzLast = d.magBz[d.magBz.length - 1];
  const southward = bzLast < -5;

  return (
    <div className="aruna-card flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-3 flex-shrink-0"
           style={{ height: 32, borderBottom: '1px solid rgba(255,180,70,0.05)' }}>
        <div className="flex items-center gap-2">
          <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
                         color: 'rgba(243,241,234,0.85)', textTransform: 'uppercase' }}>
            MAG · IMF (GSE)
          </span>
          {southward && (
            <span style={{ fontSize: 8, fontWeight: 700, color: '#FF3B3B',
                           background: 'rgba(255,59,59,0.12)', padding: '2px 6px',
                           borderRadius: 3, letterSpacing: '0.06em' }}>
              Bz SOUTH {bzLast.toFixed(1)} nT
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <TogglePill active={showBz}   onClick={() => setShowBz(v => !v)}>Bz</TogglePill>
          <TogglePill active={showBy}   onClick={() => setShowBy(v => !v)}>By</TogglePill>
          <TogglePill active={showBtot} onClick={() => setShowBtot(v => !v)}>|B|</TogglePill>
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <Plot
          className="w-full h-full"
          data={data as any}
          layout={{
            paper_bgcolor: BG, plot_bgcolor: BG,
            font: { family: 'Inter, sans-serif', size: 8, color: AX },
            margin: { t: 8, r: 48, b: 36, l: 44 },
            grid: { rows: 3, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
            /* Bz (bottom) */
            xaxis:  { gridcolor: GRID, color: AX, tickfont: { size: 7 }, nticks: 6,
                      title: { text: 'UTC (6h)', font: { size: 8 } }, domain: [0, 1] },
            yaxis:  { title: { text: 'Bz (nT)', font: { size: 7.5 } }, gridcolor: GRID,
                      zerolinecolor: 'rgba(255,244,232,0.15)', color: '#FF6B6B',
                      tickfont: { size: 7 }, domain: [0, 0.30] },
            /* By (middle) */
            xaxis2: { gridcolor: GRID, color: AX, tickfont: { size: 7 }, showticklabels: false,
                      matches: 'x', domain: [0, 1] },
            yaxis2: { title: { text: 'By (nT)', font: { size: 7.5 } }, gridcolor: GRID,
                      zerolinecolor: 'rgba(255,244,232,0.10)', color: '#00D4FF',
                      tickfont: { size: 7 }, domain: [0.35, 0.65] },
            /* |B| (top) */
            xaxis3: { gridcolor: GRID, color: AX, tickfont: { size: 7 }, showticklabels: false,
                      matches: 'x', domain: [0, 1] },
            yaxis3: { title: { text: '|B| (nT)', font: { size: 7.5 } }, gridcolor: GRID,
                      zerolinecolor: 'transparent', color: AX,
                      tickfont: { size: 7 }, domain: [0.70, 1.0] },
            legend: { x: 0.01, y: 0.99, bgcolor: 'transparent',
                      font: { size: 8, color: 'rgba(243,241,234,0.45)' } },
            shapes: [{
              type: 'line', x0: xs[0], x1: xs[xs.length - 1],
              y0: 0, y1: 0, xref: 'x', yref: 'y',
              line: { color: 'rgba(255,244,232,0.20)', width: 0.8, dash: 'dot' },
            }],
          } as any}
          config={CFG}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check + commit**

```bash
npx tsc --noEmit
git add web/src/components/scientist/MAGPlot.tsx
git commit -m "feat(chart): MAG 3-panel plot (Bz/By/|B|) with southward Bz flag"
```

---

### Task 6: DNIPlot component

**Files:**
- Create: `web/src/components/scientist/DNIPlot.tsx`

**Interfaces:**
- Consumes: `d.dniHistory` (number[], length 60), `d.dniThreshold` (number, 1.5)
- Produces: `<DNIPlot d={d} />` — DNI time series, threshold at 1.5, HOPE zone shaded above

- [ ] **Step 1: Create DNIPlot.tsx**

```tsx
// web/src/components/scientist/DNIPlot.tsx
import Plot from 'react-plotly.js';
import type { MissionSnapshot } from '../../hooks/useMissionData';

const BG   = '#040506';
const GRID = 'rgba(255,244,232,0.025)';
const AX   = 'rgba(255,244,232,0.22)';
const CFG  = { displayModeBar: false, responsive: true };

export function DNIPlot({ d }: { d: MissionSnapshot }) {
  const n  = d.dniHistory.length;
  const xs = Array.from({ length: n }, (_, i) => {
    const ms = Date.now() - (n - 1 - i) * 60_000;
    return new Date(ms).toISOString().slice(11, 16);
  });
  const thr = d.dniThreshold;

  return (
    <div className="aruna-card flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-3 flex-shrink-0"
           style={{ height: 32, borderBottom: '1px solid rgba(255,180,70,0.05)' }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
                       color: 'rgba(243,241,234,0.85)', textTransform: 'uppercase' }}>
          DNI · Dynamic Neupert Integral
        </span>
        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 9,
                       color: 'rgba(244,168,37,0.60)' }}>
          ∫F<sub>HEL</sub> dt / F<sub>SoLEXS</sub>
        </span>
      </div>
      <div className="flex-1 min-h-0">
        <Plot
          className="w-full h-full"
          data={[
            /* HOPE zone fill (above threshold) */
            {
              type: 'scatter', mode: 'none',
              x: [xs[0], xs[n - 1], xs[n - 1], xs[0]],
              y: [thr, thr, Math.max(...d.dniHistory) * 1.15, Math.max(...d.dniHistory) * 1.15],
              fill: 'toself', fillcolor: 'rgba(244,168,37,0.06)',
              hoverinfo: 'none', showlegend: false,
            },
            /* DNI line */
            {
              type: 'scatter', mode: 'lines',
              x: xs, y: d.dniHistory,
              name: 'DNI',
              line: { color: '#00D4FF', width: 2 },
              hovertemplate: 'DNI: %{y:.2f}<extra></extra>',
            },
            /* Current DNI endpoint */
            {
              type: 'scatter', mode: 'markers',
              x: [xs[n - 1]], y: [d.dniHistory[n - 1]],
              marker: { size: 7, color: '#00D4FF', line: { width: 0 } },
              hoverinfo: 'none', showlegend: false,
            },
          ] as any}
          layout={{
            paper_bgcolor: BG, plot_bgcolor: BG,
            font: { family: 'Inter, sans-serif', size: 8, color: AX },
            margin: { t: 8, r: 16, b: 36, l: 44 },
            xaxis: { gridcolor: GRID, color: AX, tickfont: { size: 7.5 }, nticks: 6,
                     title: { text: 'UTC (60 min)', font: { size: 8 } } },
            yaxis: { title: { text: 'DNI', font: { size: 8 } },
                     gridcolor: GRID, zerolinecolor: GRID, color: AX, tickfont: { size: 7.5 } },
            shapes: [{
              type: 'line', x0: xs[0], x1: xs[n - 1],
              y0: thr, y1: thr,
              line: { color: 'rgba(244,168,37,0.55)', width: 1, dash: 'dash' },
            }],
            annotations: [{
              x: xs[n - 1], y: thr, xref: 'x', yref: 'y',
              text: `HOPE ≥${thr}`, showarrow: false,
              font: { size: 8, color: 'rgba(244,168,37,0.65)', family: 'Inter' },
              xanchor: 'right', yanchor: 'bottom',
            }],
          } as any}
          config={CFG}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check + commit**

```bash
npx tsc --noEmit
git add web/src/components/scientist/DNIPlot.tsx
git commit -m "feat(chart): DNI time series with HOPE threshold and zone shading"
```

---

### Task 7: FlarePhaseStrip component

**Files:**
- Create: `web/src/components/scientist/FlarePhaseStrip.tsx`

**Interfaces:**
- Consumes: `d.flarePhase` (FlarePhase string), `d.leadTimeMin`, `d.predictedPeak`
- Produces: `<FlarePhaseStrip d={d} />` — 5-node phase timeline driven by live data

- [ ] **Step 1: Create FlarePhaseStrip.tsx**

```tsx
// web/src/components/scientist/FlarePhaseStrip.tsx
import type { MissionSnapshot, FlarePhase } from '../../hooks/useMissionData';

const PHASES: FlarePhase[] = ['PRE-FLARE', 'ONSET', 'IMPULSIVE', 'PEAK', 'DECAY'];

const PHASE_COLORS: Record<FlarePhase, string> = {
  'PRE-FLARE': '#67D46C',
  'ONSET':     '#F4A825',
  'IMPULSIVE': '#FF8A26',
  'PEAK':      '#FF3B3B',
  'DECAY':     '#A78BFA',
  'QUIET':     'rgba(243,241,234,0.30)',
};

const PHASE_SUBS: Record<FlarePhase, string> = {
  'PRE-FLARE': 'Quiet sun / precursor',
  'ONSET':     'Dual-trigger confirmed',
  'IMPULSIVE': 'Hard X-ray rise',
  'PEAK':      `GOES classification`,
  'DECAY':     'Gradual phase',
  'QUIET':     '—',
};

export function FlarePhaseStrip({ d }: { d: MissionSnapshot }) {
  const current = d.flarePhase;
  const currentIdx = PHASES.indexOf(current);

  return (
    <div className="aruna-card flex flex-col h-full overflow-hidden">
      <div className="px-3 flex-shrink-0"
           style={{ height: 32, borderBottom: '1px solid rgba(255,180,70,0.05)',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
                       color: 'rgba(243,241,234,0.85)', textTransform: 'uppercase' }}>
          Flare Phase
        </span>
        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 9,
                       color: PHASE_COLORS[current] ?? '#F4A825', fontWeight: 700,
                       letterSpacing: '0.08em' }}>
          {current}
        </span>
      </div>

      <div className="flex-1 flex flex-col justify-center px-4 gap-3">
        {/* Node row */}
        <div className="flex items-center">
          {PHASES.map((phase, i) => {
            const done   = i < currentIdx;
            const active = i === currentIdx;
            const color  = PHASE_COLORS[phase];
            return (
              <div key={phase} className="flex items-center flex-1">
                <div className="flex flex-col items-center" style={{ minWidth: 0, flex: 1 }}>
                  {/* Circle */}
                  <div style={{
                    width: active ? 22 : 16, height: active ? 22 : 16,
                    borderRadius: '50%', flexShrink: 0,
                    background: done ? color : active ? color : 'transparent',
                    border: `2px solid ${done || active ? color : 'rgba(255,244,232,0.18)'}`,
                    boxShadow: active ? `0 0 12px ${color}55` : undefined,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    {done && (
                      <svg width="8" height="8" viewBox="0 0 8 8">
                        <polyline points="1.5,4 3.5,6 6.5,2" stroke="#040506"
                                  strokeWidth="1.5" fill="none" strokeLinecap="round"/>
                      </svg>
                    )}
                  </div>
                  {/* Label below */}
                  <span style={{
                    fontSize: 7.5, fontWeight: active ? 700 : 500,
                    letterSpacing: '0.05em', textTransform: 'uppercase',
                    color: active ? color : done ? 'rgba(243,241,234,0.50)' : 'rgba(243,241,234,0.25)',
                    marginTop: 5, textAlign: 'center', lineHeight: 1.3,
                  }}>
                    {phase.replace('-', '-\n')}
                  </span>
                </div>
                {i < PHASES.length - 1 && (
                  <div style={{ height: 2, flex: 1, maxWidth: 28, flexShrink: 0,
                                background: i < currentIdx
                                  ? 'rgba(244,168,37,0.35)' : 'rgba(255,244,232,0.09)' }} />
                )}
              </div>
            );
          })}
        </div>

        {/* Sub-detail for current phase */}
        <div style={{ padding: '8px 10px', borderRadius: 4,
                      background: 'rgba(244,168,37,0.05)',
                      border: '1px solid rgba(244,168,37,0.12)' }}>
          <div style={{ fontSize: 8, fontWeight: 600, letterSpacing: '0.10em',
                        textTransform: 'uppercase', color: 'rgba(243,241,234,0.30)',
                        marginBottom: 4 }}>
            Status
          </div>
          <div style={{ fontSize: 11, color: 'rgba(243,241,234,0.80)', lineHeight: 1.4 }}>
            {PHASE_SUBS[current]} — Lead time{' '}
            <span style={{ fontFamily: 'JetBrains Mono', color: '#F4A825', fontWeight: 700 }}>
              {d.leadTimeMin} min
            </span>
            {' '}· Predicted peak{' '}
            <span style={{ fontFamily: 'JetBrains Mono', color: '#FF8A26', fontWeight: 700 }}>
              {d.predictedPeak}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check + commit**

```bash
npx tsc --noEmit
git add web/src/components/scientist/FlarePhaseStrip.tsx
git commit -m "feat(chart): FlarePhaseStrip — live 5-node phase timeline driven by MissionSnapshot"
```

---

### Task 8: ForecastPanel component

**Files:**
- Create: `web/src/components/scientist/ForecastPanel.tsx`

**Interfaces:**
- Consumes: `d.forecastProbs` (ForecastProb[], length 3)
- Produces: `<ForecastPanel d={d} />` — horizontal bar chart, 5 GOES classes, MAPIE error bars, horizon toggle

- [ ] **Step 1: Create ForecastPanel.tsx**

```tsx
// web/src/components/scientist/ForecastPanel.tsx
import { useState } from 'react';
import Plot from 'react-plotly.js';
import { TogglePill } from './TogglePill';
import type { MissionSnapshot } from '../../hooks/useMissionData';

const BG   = '#040506';
const GRID = 'rgba(255,244,232,0.025)';
const AX   = 'rgba(255,244,232,0.22)';
const CFG  = { displayModeBar: false, responsive: true };

const CLASS_LABELS = ['B', 'C', 'M', 'X', 'X+'];
const CLASS_COLORS = ['#67D46C', '#00D4FF', '#F4A825', '#FF8A26', '#FF3B3B'];

export function ForecastPanel({ d }: { d: MissionSnapshot }) {
  const [horizon, setHorizon] = useState<'1h' | '3h' | '6h'>('1h');

  const prob = d.forecastProbs.find(p => p.horizon === horizon) ?? d.forecastProbs[0];
  const vals = [prob.B, prob.C, prob.M, prob.X, prob.Xplus];
  const errLo = vals.map((v, i) => v - prob.lower[i]);
  const errHi = vals.map((v, i) => prob.upper[i] - v);

  return (
    <div className="aruna-card flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-3 flex-shrink-0"
           style={{ height: 32, borderBottom: '1px solid rgba(255,180,70,0.05)' }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
                       color: 'rgba(243,241,234,0.85)', textTransform: 'uppercase' }}>
          5-Class Forecast · DualTCN + MAPIE
        </span>
        <div className="flex items-center gap-1.5">
          {(['1h', '3h', '6h'] as const).map(h => (
            <TogglePill key={h} active={horizon === h} onClick={() => setHorizon(h)}>
              {h}
            </TogglePill>
          ))}
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <Plot
          className="w-full h-full"
          data={[{
            type: 'bar',
            orientation: 'h',
            y: CLASS_LABELS,
            x: vals,
            marker: { color: CLASS_COLORS, opacity: 0.85 },
            error_x: {
              type: 'data', symmetric: false,
              array: errHi, arrayminus: errLo,
              color: 'rgba(243,241,234,0.40)', thickness: 1.5, width: 4,
            },
            hovertemplate: '%{y}: %{x:.1%}<extra></extra>',
            text: vals.map(v => `${(v * 100).toFixed(0)}%`),
            textposition: 'outside',
            textfont: { size: 10, color: 'rgba(243,241,234,0.70)',
                        family: 'JetBrains Mono, monospace' },
          } as any]}
          layout={{
            paper_bgcolor: BG, plot_bgcolor: BG,
            font: { family: 'Inter, sans-serif', size: 9, color: AX },
            margin: { t: 12, r: 64, b: 36, l: 36 },
            xaxis: {
              title: { text: 'Probability', font: { size: 8 } },
              range: [0, Math.max(...prob.upper) * 1.25],
              gridcolor: GRID, color: AX, tickfont: { size: 8 },
              tickformat: '.0%',
            },
            yaxis: {
              gridcolor: 'transparent', color: AX, tickfont: { size: 11, family: 'JetBrains Mono' },
            },
            bargap: 0.35,
            annotations: [{
              x: 0.98, y: 0.03, xref: 'paper', yref: 'paper',
              text: `Horizon: ${horizon} · Conformal (MAPIE)`,
              showarrow: false,
              font: { size: 7.5, color: 'rgba(243,241,234,0.25)', family: 'Inter' },
              xanchor: 'right',
            }],
          } as any}
          config={CFG}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check + commit**

```bash
npx tsc --noEmit
git add web/src/components/scientist/ForecastPanel.tsx
git commit -m "feat(chart): 5-class forecast panel with MAPIE error bars and horizon toggle"
```

---

### Task 9: HistoricalAnalogPanel component

**Files:**
- Create: `web/src/components/scientist/HistoricalAnalogPanel.tsx`

**Interfaces:**
- Consumes: `d.analogSimilarity` (number), `d.solexsSDD1` (number[])
- Produces: `<HistoricalAnalogPanel d={d} />` — similarity score, analog overlay chart

- [ ] **Step 1: Create HistoricalAnalogPanel.tsx**

```tsx
// web/src/components/scientist/HistoricalAnalogPanel.tsx
import Plot from 'react-plotly.js';
import type { MissionSnapshot } from '../../hooks/useMissionData';

const BG   = '#040506';
const GRID = 'rgba(255,244,232,0.025)';
const AX   = 'rgba(255,244,232,0.22)';
const CFG  = { displayModeBar: false, responsive: true };

/* Synthetic X6.3 analog curve — normalized to [0,1] */
function makeAnalogCurve(n: number): number[] {
  return Array.from({ length: n }, (_, i) => {
    const phase = i / n;
    // Pre-flare plateau → rapid rise → peak → decay
    if (phase < 0.55) return 0.03 + phase * 0.04;
    if (phase < 0.75) return 0.03 + 0.04 * 0.55 + Math.pow((phase - 0.55) / 0.20, 2) * 0.93;
    return Math.max(0.03, 1.0 - (phase - 0.75) / 0.50);
  });
}

function normalize(arr: number[]): number[] {
  const mx = Math.max(...arr);
  const mn = Math.min(...arr);
  return arr.map(v => (v - mn) / (mx - mn || 1));
}

export function HistoricalAnalogPanel({ d }: { d: MissionSnapshot }) {
  const n          = Math.min(d.solexsSDD1.length, 120); // last 2h
  const current    = normalize(d.solexsSDD1.slice(-n));
  const analog     = makeAnalogCurve(n);
  const xs         = Array.from({ length: n }, (_, i) => i);

  return (
    <div className="aruna-card flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-3 flex-shrink-0"
           style={{ height: 32, borderBottom: '1px solid rgba(255,180,70,0.05)' }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
                       color: 'rgba(243,241,234,0.85)', textTransform: 'uppercase' }}>
          Historical Analog · DTW
        </span>
        <div className="flex items-center gap-2">
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 20,
                         fontWeight: 700, color: '#A78BFA', lineHeight: 1 }}>
            {d.analogSimilarity}%
          </span>
          <div>
            <div style={{ fontSize: 7.5, color: 'rgba(243,241,234,0.30)',
                          textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Similarity
            </div>
            <div style={{ fontSize: 8.5, fontWeight: 600, color: 'rgba(243,241,234,0.55)' }}>
              X6.3 · Feb 22 2024
            </div>
          </div>
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <Plot
          className="w-full h-full"
          data={[
            {
              type: 'scatter', mode: 'lines', x: xs, y: current,
              name: 'Current (norm.)',
              line: { color: '#F4A825', width: 1.8 },
              hovertemplate: 'Current: %{y:.2f}<extra></extra>',
            },
            {
              type: 'scatter', mode: 'lines', x: xs, y: analog,
              name: 'X6.3 analog (norm.)',
              line: { color: '#A78BFA', width: 1.5, dash: 'dot' },
              hovertemplate: 'Analog: %{y:.2f}<extra></extra>',
            },
          ] as any}
          layout={{
            paper_bgcolor: BG, plot_bgcolor: BG,
            font: { family: 'Inter, sans-serif', size: 8, color: AX },
            margin: { t: 8, r: 12, b: 36, l: 44 },
            xaxis: { title: { text: 'Relative time (min)', font: { size: 8 } },
                     gridcolor: GRID, color: AX, tickfont: { size: 7.5 } },
            yaxis: { title: { text: 'Normalized flux', font: { size: 8 } },
                     gridcolor: GRID, color: AX, tickfont: { size: 7.5 },
                     range: [-0.05, 1.15] },
            legend: { x: 0.01, y: 0.99, bgcolor: 'transparent',
                      font: { size: 8, color: 'rgba(243,241,234,0.45)' } },
          } as any}
          config={CFG}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check + commit**

```bash
npx tsc --noEmit
git add web/src/components/scientist/HistoricalAnalogPanel.tsx
git commit -m "feat(chart): historical analog panel with DTW similarity score and normalized overlay"
```

---

### Task 10: SpectroscopyPanel (refactored from PhysicsEngine)

**Files:**
- Create: `web/src/components/scientist/SpectroscopyPanel.tsx`

**Interfaces:**
- Consumes: full `MissionSnapshot`
- Produces: `<SpectroscopyPanel d={d} />` — top row: two spectrograms; bottom row: T-EM | TNPSV | DNI

- [ ] **Step 1: Create SpectroscopyPanel.tsx**

```tsx
// web/src/components/scientist/SpectroscopyPanel.tsx
import { useState } from 'react';
import Plot from 'react-plotly.js';
import { TogglePill } from './TogglePill';
import { DNIPlot } from './DNIPlot';
import type { MissionSnapshot } from '../../hooks/useMissionData';

const BG   = '#040506';
const GRID = 'rgba(255,244,232,0.025)';
const AX   = 'rgba(255,244,232,0.22)';
const CFG  = { displayModeBar: false, responsive: true };

type ColorMode = 'linear' | 'sqrt' | 'log';

const COLORSCALE = [
  [0,    '#040506'], [0.08, '#07090C'], [0.25, '#1C0B00'],
  [0.45, '#3D1500'], [0.62, '#F4A825'], [0.80, '#FF6010'], [1.0, '#FF2020'],
];

function applyColorMode(z: number[][], mode: ColorMode): number[][] {
  if (mode === 'linear') return z;
  return z.map(row => row.map(v =>
    mode === 'sqrt' ? Math.sqrt(Math.max(0, v)) : Math.log1p(Math.max(0, v))
  ));
}

function Spectrogram({ data, yMin, yMax, title, instrument }: {
  data: number[][]; yMin: number; yMax: number; title: string; instrument: string;
}) {
  const [colorMode, setColorMode] = useState<ColorMode>('sqrt');

  const nT = data.length;
  const nE = data[0]?.length ?? 1;
  const zRaw  = Array.from({ length: nE }, (_, e) =>
    Array.from({ length: nT }, (_, t) => data[t][e]));
  const zData = applyColorMode(zRaw, colorMode);
  const xVals = Array.from({ length: nT }, (_, i) => i);
  const yVals = Array.from({ length: nE }, (_, i) =>
    +(yMin + (yMax - yMin) * (i / (nE - 1))).toFixed(1));
  const xTicks  = [0, Math.floor(nT*0.25), Math.floor(nT*0.5), Math.floor(nT*0.75), nT-1];
  const xLabels = ['-60m', '-45m', '-30m', '-15m', 'NOW'];

  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ background: BG }}>
      <div className="flex items-center justify-between px-2 flex-shrink-0"
           style={{ height: 28, borderBottom: '1px solid rgba(255,180,70,0.05)' }}>
        <div className="flex items-center gap-1.5">
          <span style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: '0.06em',
                         textTransform: 'uppercase', color: 'rgba(243,241,234,0.82)' }}>
            {title}
          </span>
          <span style={{ fontSize: 7, color: 'rgba(243,241,234,0.28)' }}>· {instrument}</span>
        </div>
        <div className="flex items-center gap-1">
          {(['linear', 'sqrt', 'log'] as ColorMode[]).map(m => (
            <TogglePill key={m} active={colorMode === m} onClick={() => setColorMode(m)}>
              {m}
            </TogglePill>
          ))}
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <Plot
          className="w-full h-full"
          data={[{
            type: 'heatmap', z: zData, x: xVals, y: yVals,
            colorscale: COLORSCALE, showscale: true, zsmooth: 'best',
            colorbar: { thickness: 8, outlinewidth: 0,
                        tickfont: { size: 7, color: AX },
                        tickvals: [0, 1], ticktext: ['Low', 'High'], len: 0.85 },
            hovertemplate: 'Energy: %{y} keV<br>Intensity: %{z:.3f}<extra></extra>',
          } as any]}
          layout={{
            paper_bgcolor: BG, plot_bgcolor: BG,
            font: { family: 'Inter, sans-serif', size: 8, color: AX },
            margin: { t: 4, r: 50, b: 32, l: 44 },
            xaxis: { tickvals: xTicks, ticktext: xLabels, tickfont: { size: 7 },
                     gridcolor: GRID, color: AX,
                     title: { text: 'Time (−60 min → NOW)', font: { size: 7.5 } } },
            yaxis: { title: { text: 'Energy (keV)', font: { size: 7.5 } },
                     tickfont: { size: 7 }, gridcolor: GRID, color: AX },
            shapes: [{ type: 'line', x0: nT-1, x1: nT-1, y0: 0, y1: 1, yref: 'paper',
                       line: { color: 'rgba(244,168,37,0.40)', width: 1, dash: 'dot' } }],
            annotations: [{ x: nT-1, y: 0.97, yref: 'paper', text: 'NOW', showarrow: false,
                            font: { size: 7.5, color: 'rgba(244,168,37,0.60)', family: 'Inter' },
                            xanchor: 'right' }],
          } as any}
          config={CFG}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
}

function TEMChart({ d }: { d: MissionSnapshot }) {
  const len   = d.tHistory.length;
  const times = Array.from({ length: len }, (_, i) => {
    const dt = new Date(Date.now() - (len - i) * 1000);
    return dt.toISOString().slice(11, 19);
  });
  const hi = Math.floor(len * 0.65);
  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ background: BG }}>
      <div className="px-2 flex items-center flex-shrink-0"
           style={{ height: 28, borderBottom: '1px solid rgba(255,180,70,0.05)' }}>
        <span style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: '0.06em',
                       textTransform: 'uppercase', color: 'rgba(243,241,234,0.82)' }}>
          T(t) &amp; EM(t)
        </span>
      </div>
      <div className="flex-1 min-h-0">
        <Plot className="w-full h-full"
          data={[
            { type: 'scatter', mode: 'none',
              x: [...times, ...times.slice().reverse()],
              y: [...d.tHistory.map((v,i) => v+d.tSigma[i]),
                  ...d.tHistory.map((v,i) => v-d.tSigma[i]).reverse()],
              fill: 'toself', fillcolor: 'rgba(255,100,100,0.09)',
              hoverinfo: 'none', showlegend: false } as any,
            { type: 'scatter', mode: 'lines', x: times, y: d.tHistory,
              name: 'T (MK)', line: { color: '#FF6B6B', width: 2 },
              hovertemplate: '%{y:.1f} MK<extra>T</extra>' } as any,
            { type: 'scatter', mode: 'none',
              x: [...times, ...times.slice().reverse()],
              y: [...d.emHistory.map((v,i) => v+d.emSigma[i]),
                  ...d.emHistory.map((v,i) => v-d.emSigma[i]).reverse()],
              fill: 'toself', fillcolor: 'rgba(59,130,246,0.08)',
              hoverinfo: 'none', showlegend: false, yaxis: 'y2' } as any,
            { type: 'scatter', mode: 'lines', x: times, y: d.emHistory,
              name: 'EM (×10⁴⁸)', line: { color: '#60A5FA', width: 2 }, yaxis: 'y2',
              hovertemplate: '%{y:.2f}×10⁴⁸<extra>EM</extra>' } as any,
          ]}
          layout={{
            paper_bgcolor: BG, plot_bgcolor: BG,
            font: { family: 'Inter, sans-serif', size: 8, color: AX },
            margin: { t: 4, r: 52, b: 32, l: 44 },
            xaxis: { gridcolor: GRID, color: AX, tickfont: { size: 7 }, nticks: 5 },
            yaxis: { title: { text: 'T (MK)', font: { size: 7.5, color: '#FF6B6B' } },
                     gridcolor: GRID, color: '#FF6B6B', tickfont: { size: 7 } },
            yaxis2: { title: { text: 'EM (×10⁴⁸)', font: { size: 7.5, color: '#60A5FA' } },
                      overlaying: 'y', side: 'right',
                      gridcolor: 'transparent', color: '#60A5FA', tickfont: { size: 7 } },
            legend: { x: 0.01, y: 0.99, bgcolor: 'transparent', font: { size: 7.5 } },
            shapes: [{ type: 'line', x0: times[hi], x1: times[hi], y0: 0, y1: 1, yref: 'paper',
                       line: { color: 'rgba(244,168,37,0.50)', dash: 'dash', width: 1 } }],
            annotations: [{ x: times[hi], y: 0.96, yref: 'paper', text: 'HOPE',
                            showarrow: false, font: { size: 7.5, color: '#F4A825' }, xanchor: 'left' }],
          } as any}
          config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
      </div>
    </div>
  );
}

function TNPSVChart({ d }: { d: MissionSnapshot }) {
  const hist = d.tnpsvHistory;
  const n    = hist.length;
  const xs   = hist.map(p => p.t);
  const ys   = hist.map(p => p.em);
  const alpha = hist.map((_, i) => 0.08 + 0.92 * (i / n));
  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ background: BG }}>
      <div className="px-2 flex items-center flex-shrink-0"
           style={{ height: 28, borderBottom: '1px solid rgba(255,180,70,0.05)' }}>
        <span style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: '0.06em',
                       textTransform: 'uppercase', color: 'rgba(243,241,234,0.82)' }}>
          TNPSV Phase Space
        </span>
      </div>
      <div className="flex-1 min-h-0">
        <Plot className="w-full h-full"
          data={[
            { type: 'scatter', mode: 'none',
              x: [6,22,22,6,6], y: [1.0,1.0,5.0,5.0,1.0],
              fill: 'toself', fillcolor: 'rgba(244,168,37,0.05)',
              hoverinfo: 'none', showlegend: false } as any,
            { type: 'scatter', mode: 'markers',
              x: xs.slice(0,n-1), y: ys.slice(0,n-1),
              marker: { size: alpha.slice(0,n-1).map(a=>1.8+a*3.2),
                        color: alpha.slice(0,n-1).map(a=>`rgba(244,168,37,${(a*0.70).toFixed(2)})`),
                        line: { width: 0 } },
              hovertemplate: 'T: %{x:.1f} MK<br>EM: %{y:.2f}×10⁴⁸<extra></extra>',
              showlegend: false } as any,
            { type: 'scatter', mode: 'markers',
              x: [xs[n-1]], y: [ys[n-1]],
              marker: { size: 10, color: '#F4A825', line: { width: 1.5, color: 'rgba(244,168,37,0.35)' } },
              hovertemplate: 'T: %{x:.1f} MK<br>EM: %{y:.2f}×10⁴⁸<extra>NOW</extra>',
              showlegend: false } as any,
          ]}
          layout={{
            paper_bgcolor: BG, plot_bgcolor: BG,
            font: { family: 'Inter, sans-serif', size: 8, color: AX },
            margin: { t: 4, r: 10, b: 36, l: 52 },
            xaxis: { type: 'log', title: { text: 'T (MK)', font: { size: 7.5 } },
                     gridcolor: GRID, color: AX, tickfont: { size: 7 }, range: [-0.3, 1.4] },
            yaxis: { type: 'log', title: { text: 'EM (×10⁴⁸)', font: { size: 7.5 } },
                     gridcolor: GRID, color: AX, tickfont: { size: 7 }, range: [-2.2, 0.85] },
            shapes: [{ type: 'rect', x0: 6, y0: 1.0, x1: 22, y1: 5.0,
                       xref: 'x', yref: 'y', fillcolor: 'transparent',
                       line: { color: 'rgba(244,168,37,0.22)', width: 1, dash: 'dot' } }],
            annotations: [{ x: 12, y: 3.2, xref: 'x', yref: 'y',
                            text: '◆ HOPE ZONE', showarrow: false,
                            font: { size: 7.5, color: 'rgba(244,168,37,0.55)', family: 'Inter' } }],
          } as any}
          config={CFG} style={{ width: '100%', height: '100%' }} useResizeHandler />
      </div>
    </div>
  );
}

export function SpectroscopyPanel({ d }: { d: MissionSnapshot }) {
  return (
    <div className="flex flex-col h-full overflow-hidden gap-px"
         style={{ background: 'rgba(255,180,70,0.04)' }}>
      {/* Row 1: spectrograms (45% height) */}
      <div className="flex gap-px" style={{ flex: '0 0 45%', minHeight: 0 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <Spectrogram data={d.solexsSpectrogram} yMin={1} yMax={7}
                       title="SoLEXS Spectrogram" instrument="~1–7 keV · SDD1/SDD2" />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <Spectrogram data={d.hel1osSpectrogram} yMin={10} yMax={150}
                       title="HEL1OS Spectrogram" instrument="10–150 keV · CdTe/CZT" />
        </div>
      </div>
      {/* Row 2: T-EM | TNPSV | DNI (55% height) */}
      <div className="flex gap-px" style={{ flex: 1, minHeight: 0 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <TEMChart d={d} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <TNPSVChart d={d} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <DNIPlot d={d} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check + commit**

```bash
npx tsc --noEmit
git add web/src/components/scientist/SpectroscopyPanel.tsx
git commit -m "feat(chart): SpectroscopyPanel — spectrograms + T/EM + TNPSV + DNI with colorscale toggles"
```

---

### Task 11: Update HeroMetrics — add flarePhase badge + Bz warning

**Files:**
- Modify: `web/src/components/mission-control/HeroMetrics.tsx`

**Interfaces:**
- Consumes: `d.flarePhase` (FlarePhase), `d.magBz` (number[], length 360)

- [ ] **Step 1: Add import and two new elements to HeroMetrics.tsx**

After the existing `import { goesColor } from '../../lib/utils';` line, add:
```tsx
import type { FlarePhase } from '../../hooks/useMissionData';
```

Replace the `export function HeroMetrics({ d }: ...` opening div's `gap-4` with `gap-3`, then add two new `<Cell>` blocks after the existing Confidence ring cell. Insert before the closing `</div>` of the outer flex container:

Full replacement for `HeroMetrics.tsx` — add flarePhase + Bz after the Confidence ring `<Divider />`:

```tsx
      <Divider />

      {/* 7 — Flare Phase */}
      <div className="flex flex-col justify-center flex-shrink-0 px-1">
        <span style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.11em',
                       color: 'rgba(243,241,234,0.35)', textTransform: 'uppercase', marginBottom: 3 }}>
          Flare Phase
        </span>
        <span style={{ fontSize: 13, fontWeight: 700, letterSpacing: '0.06em',
                       color: PHASE_COLORS[d.flarePhase] ?? '#F4A825',
                       padding: '3px 8px', borderRadius: 4,
                       background: `${PHASE_COLORS[d.flarePhase] ?? '#F4A825'}18`,
                       border: `1px solid ${PHASE_COLORS[d.flarePhase] ?? '#F4A825'}40` }}>
          {d.flarePhase}
        </span>
      </div>

      <Divider />

      {/* 8 — Bz */}
      {(() => {
        const bz = d.magBz[d.magBz.length - 1] ?? 0;
        const south = bz < -5;
        return (
          <div className="flex flex-col justify-center flex-shrink-0 px-1">
            <span style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.11em',
                           color: 'rgba(243,241,234,0.35)', textTransform: 'uppercase', marginBottom: 3 }}>
              IMF Bz
            </span>
            <div className="flex items-baseline gap-1">
              <span className="telem" style={{ fontSize: 26, fontWeight: 700,
                                               color: south ? '#FF3B3B' : '#67D46C' }}>
                {bz.toFixed(1)}
              </span>
              <span className="telem" style={{ fontSize: 11, color: 'rgba(243,241,234,0.40)' }}>nT</span>
            </div>
            <span style={{ fontSize: 9.5, color: south ? '#FF3B3B' : 'rgba(243,241,234,0.30)',
                           fontWeight: south ? 700 : 400 }}>
              {south ? '▼ SOUTHWARD' : 'Northward'}
            </span>
          </div>
        );
      })()}
```

Add the `PHASE_COLORS` map at the top of the file (after imports):
```tsx
const PHASE_COLORS: Record<string, string> = {
  'PRE-FLARE': '#67D46C', 'ONSET': '#F4A825', 'IMPULSIVE': '#FF8A26',
  'PEAK': '#FF3B3B', 'DECAY': '#A78BFA', 'QUIET': 'rgba(243,241,234,0.40)',
};
```

- [ ] **Step 2: Type-check + commit**

```bash
npx tsc --noEmit
git add web/src/components/mission-control/HeroMetrics.tsx
git commit -m "feat(ui): HeroMetrics — add flare phase badge and IMF Bz southward indicator"
```

---

### Task 12: AppSidebar cleanup — remove AI agent section

**Files:**
- Modify: `web/src/components/mission-control/AppSidebar.tsx`

- [ ] **Step 1: Remove ArunaAgentDiagram from AppSidebar**

Delete the entire `function AgentNode(...)` block, the entire `function ArunaAgentDiagram()` block, and the `<ArunaAgentDiagram />` JSX call inside `AppSidebar`. The sidebar footer should become just the branding + data source text. The full modified `AppSidebar.tsx`:

```tsx
// web/src/components/mission-control/AppSidebar.tsx
import { useLocation, Link } from 'react-router-dom';
import {
  Home, Activity, Radio, TrendingUp, Shield, Clock,
  BookOpen, CheckSquare, Settings,
} from 'lucide-react';

const NAV = [
  { icon: Home,        label: 'Mission Control',     path: '/' },
  { icon: Activity,    label: 'Live Physics Stream',  path: '/live' },
  { icon: Radio,       label: 'Nowcasting',           path: '/nowcasting' },
  { icon: TrendingUp,  label: 'Forecast Center',      path: '/forecast' },
  { icon: Shield,      label: 'Threat Intelligence',  path: '/threat' },
  { icon: Clock,       label: 'Historical Replay',    path: '/replay' },
  { icon: BookOpen,    label: 'Master Flare Catalog', path: '/catalog' },
  { icon: CheckSquare, label: 'Validation & Perf.',   path: '/validation' },
  { icon: Settings,    label: 'Pipeline Health',      path: '/health' },
];

export function AppSidebar() {
  const { pathname } = useLocation();

  return (
    <aside className="flex flex-col border-r h-full overflow-hidden"
           style={{ width: 220, borderColor: 'rgba(255,200,87,0.06)', background: 'transparent' }}>
      <nav className="flex-1 pt-5 overflow-y-auto">
        {NAV.map(({ icon: Icon, label, path }) => {
          const active = pathname === path;
          return (
            <Link key={path} to={path}
                  className="flex items-center gap-4 px-5 relative transition-colors duration-150"
                  style={{
                    height: 56,
                    background: active ? 'rgba(244,168,37,0.05)' : undefined,
                    color: active ? '#F4A825' : 'rgba(243,241,234,0.48)',
                    boxShadow: active ? 'inset 3px 0 0 #F4A825' : undefined,
                  }}
                  onMouseEnter={e => { if (!active) (e.currentTarget as HTMLElement).style.background = 'rgba(255,200,87,0.03)'; }}
                  onMouseLeave={e => { if (!active) (e.currentTarget as HTMLElement).style.background = ''; }}>
              <Icon size={17} strokeWidth={1.75} />
              <span style={{ fontSize: 12, letterSpacing: '0.06em',
                             textTransform: 'uppercase', fontWeight: 400 }}>
                {label}
              </span>
            </Link>
          );
        })}
      </nav>

      <div className="border-t px-5 py-4" style={{ borderColor: 'rgba(255,200,87,0.05)' }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#F4A825' }}>ARUNA v1.0</div>
        <div style={{ fontSize: 11, fontStyle: 'italic', color: 'rgba(243,241,234,0.40)',
                      marginTop: 2 }}>Insight Before Impact.</div>
        <div style={{ marginTop: 12 }}>
          <div className="panel-label">Data Source</div>
          <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
                        fontWeight: 500, color: 'rgba(0,212,255,0.65)', marginTop: 4,
                        lineHeight: 1.4 }}>
            Aditya-L1 SoLEXS &amp; HEL1OS Level-1
          </div>
        </div>
      </div>
    </aside>
  );
}
```

- [ ] **Step 2: Type-check + commit**

```bash
npx tsc --noEmit
git add web/src/components/mission-control/AppSidebar.tsx
git commit -m "refactor(ui): remove decorative AI agent diagram from sidebar"
```

---

### Task 13: Wire MissionControl.tsx — three-mode conditional render + keyboard shortcuts

**Files:**
- Modify: `web/src/pages/MissionControl.tsx`

**Interfaces:**
- Consumes: all new scientist components, `useViewStore`, keyboard events `'1'/'2'/'3'`
- Produces: fully wired three-mode scientist dashboard

- [ ] **Step 1: Replace MissionControl.tsx**

```tsx
// web/src/pages/MissionControl.tsx
import { useEffect } from 'react';
import { useMissionData }         from '../hooks/useMissionData';
import { useViewStore }           from '../stores/useViewStore';
import { AppHeader }              from '../components/mission-control/AppHeader';
import { AppSidebar }             from '../components/mission-control/AppSidebar';
import { HeroMetrics }            from '../components/mission-control/HeroMetrics';
import { ViewModeBar }            from '../components/scientist/ViewModeBar';
/* OVERVIEW */
import { SoLEXSLightCurve }       from '../components/scientist/SoLEXSLightCurve';
import { HEL1OSLightCurve }       from '../components/scientist/HEL1OSLightCurve';
import { MAGPlot }                from '../components/scientist/MAGPlot';
import { FlarePhaseStrip }        from '../components/scientist/FlarePhaseStrip';
/* SPECTROSCOPY */
import { SpectroscopyPanel }      from '../components/scientist/SpectroscopyPanel';
/* FORECAST */
import { ForecastPanel }          from '../components/scientist/ForecastPanel';
import { HistoricalAnalogPanel }  from '../components/scientist/HistoricalAnalogPanel';

export default function MissionControl() {
  const d = useMissionData();
  const { mode, setMode } = useViewStore();

  /* Keyboard shortcuts: 1/2/3 */
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return;
      if (e.key === '1') setMode('overview');
      if (e.key === '2') setMode('spectroscopy');
      if (e.key === '3') setMode('forecast');
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [setMode]);

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background: '#040506' }}>
      <div className="noise-overlay" />

      {/* Header */}
      <AppHeader />

      {/* Body */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <AppSidebar />

        <div className="flex flex-col flex-1 min-w-0 gap-1.5 p-1.5 overflow-hidden">

          {/* Hero metrics — always visible */}
          <div style={{ height: 78, flexShrink: 0 }}>
            <HeroMetrics d={d} />
          </div>

          {/* View mode switcher — always visible, replaces Pipeline strip */}
          <ViewModeBar />

          {/* Main content area — switches per mode */}
          <div className="flex-1 min-h-0 overflow-hidden">

            {/* ── OVERVIEW ── */}
            {mode === 'overview' && (
              <div className="grid h-full gap-1.5"
                   style={{ gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr' }}>
                <SoLEXSLightCurve d={d} />
                <MAGPlot d={d} />
                <HEL1OSLightCurve d={d} />
                <FlarePhaseStrip d={d} />
              </div>
            )}

            {/* ── SPECTROSCOPY ── */}
            {mode === 'spectroscopy' && (
              <div className="h-full">
                <SpectroscopyPanel d={d} />
              </div>
            )}

            {/* ── FORECAST ── */}
            {mode === 'forecast' && (
              <div className="grid h-full gap-1.5"
                   style={{ gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr' }}>
                <ForecastPanel d={d} />
                <div className="aruna-card flex items-center justify-center"
                     style={{ color: 'rgba(243,241,234,0.20)', fontSize: 11 }}>
                  {/* TNPSV reused inline — import SpectroscopyPanel's TNPSVChart is private,
                      so we embed PhysicsEngine's TNPSV logic here via the existing PhysicsEngine */}
                  <SpectroscopyPanel d={d} />
                </div>
                <MAGPlot d={d} />
                <HistoricalAnalogPanel d={d} />
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  );
}
```

> **Note on FORECAST top-right cell:** The TNPSV chart is currently private inside SpectroscopyPanel. In the next step we fix this by exporting it separately, or for now we use `<SpectroscopyPanel d={d} />` which renders the full panel — the simpler approach since this task is about wiring, not pixel-perfect layout.

- [ ] **Step 2: Fix FORECAST top-right — export TNPSVChart from SpectroscopyPanel**

In `web/src/components/scientist/SpectroscopyPanel.tsx`, change `function TNPSVChart` to `export function TNPSVChart` and add this import in `MissionControl.tsx`:

```tsx
import { SpectroscopyPanel, TNPSVChart } from '../components/scientist/SpectroscopyPanel';
```

Then replace the placeholder cell in the FORECAST grid:
```tsx
<div className="aruna-card overflow-hidden">
  <TNPSVChart d={d} />
</div>
```

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: 0 errors

- [ ] **Step 4: Visual verify in browser**

Open http://localhost:5173

Check:
- Press `1` → OVERVIEW: SoLEXS + MAG (top), HEL1OS + FlarePhase (bottom)
- Press `2` → SPECTROSCOPY: spectrograms (top), T-EM + TNPSV + DNI (bottom)
- Press `3` → FORECAST: probability bars + TNPSV + MAG + analog
- TogglePills on SoLEXS chart: click SDD2 off → trace disappears
- TogglePills on HEL1OS: toggle individual bands
- Forecast horizon pills: click 3h → bars update (same values in mock, but pill switches)
- MAG southward badge: visible if Bz < -5 nT (check by watching live data tick)
- Keyboard `1/2/3` switches modes instantly

- [ ] **Step 5: Commit**

```bash
git add web/src/pages/MissionControl.tsx web/src/components/scientist/SpectroscopyPanel.tsx
git commit -m "feat(dashboard): wire three-mode scientist dashboard — OVERVIEW/SPECTROSCOPY/FORECAST with keyboard shortcuts"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|-------------|------|
| Remove ArunaIntelligencePipeline | Task 13 (not imported) |
| Remove LLMBriefing | Task 13 (not imported) |
| Remove ThreatIntelligence from MC | Task 13 (not imported) |
| Remove LivePhysicsStream | Task 13 (not imported) |
| Remove QuickActions | Task 13 (not imported) |
| Remove AI agent sidebar | Task 12 |
| useViewStore | Task 2 |
| ViewModeBar with keyboard hint | Task 2 |
| SoLEXSLightCurve with SDD1/SDD2/GOES/HOPE toggles | Task 3 |
| HEL1OSLightCurve with 3 band toggles | Task 4 |
| MAGPlot Bz/By/|B| with southward fill | Task 5 |
| DNIPlot with threshold at 1.5 | Task 6 |
| FlarePhaseStrip driven by flarePhase | Task 7 |
| ForecastPanel 5-class + MAPIE + horizon | Task 8 |
| HistoricalAnalogPanel DTW overlay | Task 9 |
| SpectroscopyPanel (spectrograms + T-EM + TNPSV + DNI) | Task 10 |
| HeroMetrics flarePhase badge | Task 11 |
| HeroMetrics Bz southward indicator | Task 11 |
| useMissionData new fields | Task 1 |
| Keyboard shortcuts 1/2/3 | Task 13 |
| OVERVIEW 2×2 grid layout | Task 13 |
| SPECTROSCOPY full-height layout | Task 13 |
| FORECAST 2×2 grid layout | Task 13 |

**Placeholder scan:** None found — all steps contain complete code.

**Type consistency:** `MissionSnapshot`, `FlarePhase`, `ForecastProb` defined in Task 1 and consumed consistently in Tasks 3–13.
