import { useState } from 'react';
import { useTimelineQuery } from '../hooks/useRestQuery';
import { AppHeader } from '../components/mission-control/AppHeader';
import { AppSidebar } from '../components/mission-control/AppSidebar';

type Severity = 'INFO' | 'WARNING' | 'CRITICAL';

const SEV_COLORS: Record<Severity, string> = {
  INFO:     '#00C48C',
  WARNING:  '#FFC857',
  CRITICAL: '#FF3B3B',
};

const SEV_ICONS: Record<Severity, string> = {
  INFO:     '◉',
  WARNING:  '▲',
  CRITICAL: '◉',
};

const EVENT_ICONS: Record<string, string> = {
  PHYSICS_FRAME:      '◦',
  HOPE_DETECTED:      '◉',
  FLARE_ONSET:        '◉',
  FORECAST_UPDATED:   '◎',
  CONFIDENCE_UPDATED: '◎',
  THREAT_UPDATED:     '▲',
  NAVIC_IMPACT_UPDATED: '▲',
  DECISION_UPDATED:   '⬡',
  PIPELINE_WARNING:   '⚠',
  PIPELINE_ANOMALY:   '⚠',
  MODEL_SWITCHED:     '↻',
  REPLAY_STARTED:     '▶',
  REPLAY_COMPLETE:    '■',
  MISSION_COMPLETE:   '★',
};

interface TimelineEvent {
  id: string;
  timestamp: string;
  event_type: string;
  label: string;
  detail: string;
  severity: Severity;
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toISOString().slice(11, 19) + ' UTC';
  } catch {
    return iso;
  }
}

function EventRow({ event, selected, onClick }: {
  event: TimelineEvent;
  selected: boolean;
  onClick: () => void;
}) {
  const color = SEV_COLORS[event.severity] ?? '#6B7280';
  const icon = EVENT_ICONS[event.event_type] ?? SEV_ICONS[event.severity] ?? '·';

  return (
    <button
      onClick={onClick}
      className="w-full text-left flex items-start gap-3 px-4 py-3 border-b transition-colors"
      style={{
        borderColor: 'rgba(255,200,87,0.05)',
        background: selected ? `${color}0d` : 'transparent',
      }}
    >
      <span className="text-sm flex-shrink-0 mt-0.5" style={{ color }}>{icon}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3">
          <span className="font-mono text-[10px] text-solar-glow/40">{formatTime(event.timestamp)}</span>
          <span className="text-[11px] font-bold uppercase tracking-wider" style={{ color }}>
            {event.label}
          </span>
        </div>
        <div className="text-[10px] text-solar-glow/50 mt-0.5 truncate">{event.detail}</div>
      </div>
      <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded flex-shrink-0"
            style={{ background: `${color}18`, color }}>
        {event.severity}
      </span>
    </button>
  );
}

function EventDetail({ event }: { event: TimelineEvent }) {
  const color = SEV_COLORS[event.severity] ?? '#6B7280';
  return (
    <div className="aruna-card p-5 flex flex-col gap-4">
      <div>
        <div className="panel-label mb-1">EVENT DETAIL</div>
        <div className="text-lg font-bold" style={{ color }}>{event.label}</div>
      </div>
      <div className="space-y-3 text-xs">
        <div className="flex gap-3">
          <span className="text-solar-glow/40 w-28 flex-shrink-0">Time</span>
          <span className="font-mono text-solar-glow/80">{formatTime(event.timestamp)}</span>
        </div>
        <div className="flex gap-3">
          <span className="text-solar-glow/40 w-28 flex-shrink-0">Event Type</span>
          <span className="font-mono text-solar-glow/80">{event.event_type}</span>
        </div>
        <div className="flex gap-3">
          <span className="text-solar-glow/40 w-28 flex-shrink-0">Severity</span>
          <span style={{ color }} className="font-bold">{event.severity}</span>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-solar-glow/40">Detail</span>
          <span className="text-solar-glow/70 leading-relaxed">{event.detail}</span>
        </div>
      </div>
    </div>
  );
}

export default function MissionChroniclePage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [severityFilter, setSeverityFilter] = useState<Severity | 'ALL'>('ALL');
  const { data, isLoading } = useTimelineQuery(200);

  const events: TimelineEvent[] = (data as { events?: TimelineEvent[] })?.events ?? [];
  const filtered = severityFilter === 'ALL'
    ? events
    : events.filter(e => e.severity === severityFilter);
  const selected = events.find(e => e.id === selectedId) ?? null;

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background: '#0B1020' }}>
      <div className="noise-overlay" />
      <AppHeader />
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <AppSidebar />
        <main className="flex flex-1 min-h-0 overflow-hidden gap-4 p-4">
          {/* Left — Chronicle list */}
          <div className="flex flex-col flex-1 min-w-0 aruna-card overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-sunlit-gold/10">
              <div>
                <div className="panel-label">MISSION CHRONICLE</div>
                <div className="text-[9px] text-solar-glow/30 uppercase tracking-widest mt-0.5">
                  {filtered.length} events
                </div>
              </div>
              <div className="flex gap-2">
                {(['ALL', 'INFO', 'WARNING', 'CRITICAL'] as const).map(sev => (
                  <button
                    key={sev}
                    onClick={() => setSeverityFilter(sev)}
                    className="text-[9px] uppercase tracking-wider px-2 py-1 rounded transition-colors"
                    style={{
                      background: severityFilter === sev ? 'rgba(255,200,87,0.12)' : 'transparent',
                      color: severityFilter === sev ? '#FFC857' : 'rgba(255,244,232,0.35)',
                      border: `1px solid ${severityFilter === sev ? 'rgba(255,200,87,0.25)' : 'transparent'}`,
                    }}
                  >
                    {sev}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex-1 overflow-y-auto">
              {isLoading ? (
                <div className="p-8 text-solar-glow/30 text-xs text-center">Loading chronicle…</div>
              ) : filtered.length === 0 ? (
                <div className="p-8 text-solar-glow/30 text-xs text-center">
                  No events yet — chronicle populates as the pipeline runs.
                </div>
              ) : (
                filtered.map(event => (
                  <EventRow
                    key={event.id}
                    event={event}
                    selected={event.id === selectedId}
                    onClick={() => setSelectedId(event.id === selectedId ? null : event.id)}
                  />
                ))
              )}
            </div>
          </div>

          {/* Right — Event detail */}
          <div className="w-80 flex-shrink-0">
            {selected ? (
              <EventDetail event={selected} />
            ) : (
              <div className="aruna-card p-5">
                <div className="panel-label mb-2">EVENT DETAIL</div>
                <div className="text-xs text-solar-glow/30">
                  Click any chronicle entry to see full event detail.
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
