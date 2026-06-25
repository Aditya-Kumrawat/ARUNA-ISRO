import { useHealthQuery } from '../hooks/useRestQuery';
import { useMissionStore } from '../stores/missionStore';
import { AppHeader } from '../components/mission-control/AppHeader';
import { AppSidebar } from '../components/mission-control/AppSidebar';

const DOMAIN_MAP = [
  {
    domain: 'DATA',
    color: '#00D4FF',
    stages: ['L0', 'L1'],
    metrics: ['PRADAN status', 'FITS freshness', 'Dead-time fraction', 'Gap count'],
  },
  {
    domain: 'PHYSICS',
    color: '#FFC857',
    stages: ['L2', 'XL2', 'XL3'],
    metrics: ['T/EM fit quality', 'MPCI score', 'Frame rate', 'χ²'],
  },
  {
    domain: 'FORECAST',
    color: '#A855F7',
    stages: ['L4', 'XL4'],
    metrics: ['TCN latency', 'MAPIE coverage', 'Confidence Engine'],
  },
  {
    domain: 'THREAT',
    color: '#FF8A00',
    stages: ['L6'],
    metrics: ['NavIC model latency', 'Bulletin generation'],
  },
  {
    domain: 'DECISION',
    color: '#FF3B3B',
    stages: ['XL5', 'DX'],
    metrics: ['Mission Impact latency', 'Decision Engine latency', 'Recommendation age'],
  },
  {
    domain: 'UI',
    color: '#00C48C',
    stages: ['Socket.IO', 'API'],
    metrics: ['Frame delivery latency', 'Client connections', 'REST response times'],
  },
];

export default function PipelineHealthPage() {
  const { data: healthData } = useHealthQuery();
  const mission = useMissionStore();

  const health = healthData as {
    stages?: Array<{ id: string; healthy: boolean; latency?: number }>;
    db_connected?: boolean;
    uptime_sec?: number;
    pipeline_version?: string;
  } | undefined;

  const getStageHealth = (stageId: string): boolean => {
    return health?.stages?.find(s => s.id === stageId)?.healthy ?? true;
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background: '#0B1020' }}>
      <div className="noise-overlay" />
      <AppHeader />
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <AppSidebar />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="panel-label mb-1">PIPELINE HEALTH</div>
          <div className="text-[10px] text-solar-glow/35 uppercase tracking-widest mb-6">
            Platform integrity — domain view
          </div>

          {/* System overview strip */}
          <div className="aruna-card p-4 mb-6 flex gap-8">
            {[
              ['Connection', mission.connectionMode, mission.connectionMode === 'LIVE' ? '#00C48C' : '#FFC857'],
              ['Uptime', health?.uptime_sec ? `${Math.floor(health.uptime_sec / 60)}m` : '—', '#00D4FF'],
              ['Database', health?.db_connected ? 'CONNECTED' : 'OFFLINE', health?.db_connected ? '#00C48C' : '#FF3B3B'],
              ['Version', health?.pipeline_version ?? '2.0.0', '#FFC857'],
            ].map(([label, value, color]) => (
              <div key={label} className="flex flex-col gap-0.5">
                <div className="text-[9px] text-solar-glow/40 uppercase tracking-widest">{label}</div>
                <div className="text-sm font-bold" style={{ color: color as string }}>{value}</div>
              </div>
            ))}
          </div>

          {/* Domain cards */}
          <div className="grid grid-cols-3 gap-4">
            {DOMAIN_MAP.map(({ domain, color, stages, metrics }) => {
              const allHealthy = stages.every(s => getStageHealth(s));
              return (
                <div key={domain} className="aruna-card p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="panel-label" style={{ color }}>{domain}</div>
                    <span className="text-[9px] font-bold px-2 py-0.5 rounded"
                          style={{
                            background: allHealthy ? 'rgba(0,196,140,0.12)' : 'rgba(255,59,59,0.12)',
                            color: allHealthy ? '#00C48C' : '#FF3B3B',
                          }}>
                      {allHealthy ? 'NOMINAL' : 'DEGRADED'}
                    </span>
                  </div>

                  {/* Stage chips */}
                  <div className="flex flex-wrap gap-1.5 mb-3">
                    {stages.map(s => {
                      const ok = getStageHealth(s);
                      return (
                        <span key={s} className="text-[9px] px-2 py-0.5 rounded font-mono"
                              style={{
                                background: ok ? `${color}15` : 'rgba(255,59,59,0.12)',
                                color: ok ? color : '#FF3B3B',
                                border: `1px solid ${ok ? `${color}30` : 'rgba(255,59,59,0.30)'}`,
                              }}>
                          {s}
                        </span>
                      );
                    })}
                  </div>

                  {/* Metric labels */}
                  <div className="space-y-1">
                    {metrics.map(m => (
                      <div key={m} className="text-[9px] text-solar-glow/40 flex items-center gap-1.5">
                        <span className="w-1 h-1 rounded-full flex-shrink-0" style={{ background: `${color}60` }} />
                        {m}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </main>
      </div>
    </div>
  );
}
