import { useState } from 'react';
import { useConfigQuery, useConfigMutation } from '../hooks/useRestQuery';

const GOLD = '#F4A825';
const CARD = { background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,200,87,0.07)', borderRadius: 12, padding: 24 };

const PROFILES = [
  { id: 'competition', label: 'Competition Mode', description: 'Locked — BAH 2026 evaluation settings', locked: true },
  { id: 'nominal',     label: 'Nominal Operations', description: 'Standard thresholds for daily monitoring', locked: false },
  { id: 'sensitive',   label: 'High Sensitivity', description: 'Lower thresholds — more alerts, less misses', locked: false },
  { id: 'conservative',label: 'Conservative', description: 'Higher thresholds — fewer false positives', locked: false },
  { id: 'research',    label: 'Research Mode', description: 'All thresholds disabled — full data output', locked: false },
];

const SLIDERS = [
  { key: 'hope_threshold',        label: 'HOPE Detection Threshold',   min: 0.3, max: 0.9, step: 0.01, unit: '' },
  { key: 'confidence_min',        label: 'Min Decision Confidence',    min: 0.0, max: 1.0, step: 0.05, unit: '' },
  { key: 'p85_window_min',        label: 'P85 Rolling Window',         min: 30,  max: 360, step: 30,   unit: 'min' },
  { key: 'tcn_weight',            label: 'TCN Model Weight',           min: 0.0, max: 1.0, step: 0.05, unit: '' },
  { key: 'moirai_weight',         label: 'Moirai Model Weight',        min: 0.0, max: 1.0, step: 0.05, unit: '' },
  { key: 'dual_trigger_window',   label: 'Dual-Trigger Window',        min: 60,  max: 300, step: 30,   unit: 's' },
  { key: 'analog_similarity_min', label: 'Analog Similarity Minimum',  min: 0.5, max: 0.99,step: 0.01, unit: '' },
  { key: 'hfd_window_sec',        label: 'HFD Window',                 min: 300, max: 1200,step: 60,   unit: 's' },
];

const DEFAULTS: Record<string, number> = {
  hope_threshold: 0.65, confidence_min: 0.30, p85_window_min: 180,
  tcn_weight: 0.6, moirai_weight: 0.4, dual_trigger_window: 180,
  analog_similarity_min: 0.80, hfd_window_sec: 600,
};

export default function ConfigurationPage() {
  const { data: configData } = useConfigQuery();
  const mutation = useConfigMutation();
  const [activeProfile, setActiveProfile] = useState('nominal');
  const [values, setValues] = useState<Record<string, number>>(DEFAULTS);
  const [saved, setSaved] = useState(false);

  const locked = activeProfile === 'competition';

  const handleSave = () => {
    mutation.mutate({ profile: activeProfile, thresholds: values });
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  return (
    <div style={{ padding: 32, maxWidth: 1200 }}>
      <div style={{ marginBottom: 32 }}>
        <div style={{ fontSize: 11, letterSpacing: '0.15em', color: 'rgba(255,244,232,0.35)', marginBottom: 8, textTransform: 'uppercase' }}>Operations</div>
        <h1 style={{ fontSize: 28, fontWeight: 700, color: GOLD, margin: 0 }}>Configuration</h1>
        <p style={{ fontSize: 14, color: 'rgba(255,244,232,0.5)', marginTop: 8 }}>Pipeline threshold profiles and model weights</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 24 }}>
        {/* Profile selector */}
        <div style={CARD}>
          <div style={{ fontSize: 11, letterSpacing: '0.12em', color: 'rgba(255,244,232,0.35)', marginBottom: 14, textTransform: 'uppercase' }}>
            Profiles
          </div>
          {PROFILES.map(p => (
            <button key={p.id} onClick={() => !p.locked && setActiveProfile(p.id)}
                    style={{
                      display: 'block', width: '100%', textAlign: 'left', padding: '12px 14px',
                      marginBottom: 6, borderRadius: 8, cursor: p.locked ? 'not-allowed' : 'pointer',
                      background: activeProfile === p.id ? 'rgba(244,168,37,0.08)' : 'rgba(255,255,255,0.02)',
                      border: `1px solid ${activeProfile === p.id ? 'rgba(244,168,37,0.3)' : 'rgba(255,255,255,0.05)'}`,
                      opacity: p.locked ? 0.7 : 1,
                    }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: activeProfile === p.id ? GOLD : 'rgba(255,244,232,0.70)' }}>
                  {p.label}
                </span>
                {p.locked && <span style={{ fontSize: 9, color: '#EF4444', fontWeight: 700, padding: '1px 6px', borderRadius: 8, border: '1px solid #EF444444', background: '#EF444415' }}>LOCKED</span>}
              </div>
              <div style={{ fontSize: 10, color: 'rgba(255,244,232,0.35)', marginTop: 4 }}>{p.description}</div>
            </button>
          ))}
          {!!configData && (
            <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.05)', fontSize: 10, color: 'rgba(255,244,232,0.25)' }}>
              Config loaded from config.yaml
            </div>
          )}
        </div>

        {/* Slider panel */}
        <div style={CARD}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
            <div style={{ fontSize: 11, letterSpacing: '0.12em', color: 'rgba(255,244,232,0.35)', textTransform: 'uppercase' }}>
              Threshold Sliders {locked && '— LOCKED'}
            </div>
            <button onClick={handleSave} disabled={locked || mutation.isPending}
                    style={{
                      fontSize: 12, fontWeight: 700, padding: '8px 20px', borderRadius: 8, cursor: locked ? 'not-allowed' : 'pointer',
                      background: saved ? 'rgba(74,222,128,0.15)' : 'rgba(244,168,37,0.12)',
                      color: saved ? '#4ADE80' : GOLD,
                      border: `1px solid ${saved ? '#4ADE8044' : 'rgba(244,168,37,0.3)'}`,
                      opacity: locked ? 0.4 : 1,
                    }}>
              {saved ? '✓ Saved' : mutation.isPending ? 'Saving…' : 'Save Changes'}
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
            {SLIDERS.map(s => (
              <div key={s.key}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: 'rgba(255,244,232,0.55)' }}>{s.label}</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: GOLD, fontFamily: 'monospace' }}>
                    {values[s.key]?.toFixed(2)}{s.unit}
                  </span>
                </div>
                <input type="range" min={s.min} max={s.max} step={s.step}
                       value={values[s.key] ?? DEFAULTS[s.key]}
                       disabled={locked}
                       onChange={e => !locked && setValues(v => ({ ...v, [s.key]: parseFloat(e.target.value) }))}
                       style={{ width: '100%', accentColor: GOLD, cursor: locked ? 'not-allowed' : 'pointer' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'rgba(255,244,232,0.20)', marginTop: 2 }}>
                  <span>{s.min}{s.unit}</span><span>{s.max}{s.unit}</span>
                </div>
              </div>
            ))}
          </div>

          {locked && (
            <div style={{
              marginTop: 24, padding: '12px 16px', borderRadius: 8,
              background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)',
              fontSize: 12, color: '#EF9999',
            }}>
              ⚠ Competition Mode is locked. Thresholds are fixed for BAH 2026 Problem Statement 15 evaluation.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
