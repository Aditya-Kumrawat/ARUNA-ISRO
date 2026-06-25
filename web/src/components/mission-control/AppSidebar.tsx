import { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import {
  Home, Activity, Radio, TrendingUp, Shield, Clock,
  BookOpen, CheckSquare, Sun, ScrollText, Sliders, HelpCircle, Cpu,
  ChevronLeft, ChevronRight
} from 'lucide-react';

const NAV_GROUPS = [
  {
    group: 'OBSERVE',
    items: [
      { icon: Home,        label: 'Mission Control',    path: '/' },
    ],
  },
  {
    group: 'UNDERSTAND',
    items: [
      { icon: Activity,    label: 'Live Physics',       path: '/live' },
      { icon: Sun,         label: 'Solar View',         path: '/solar' },
    ],
  },
  {
    group: 'FORESEE',
    items: [
      { icon: Radio,       label: 'Nowcasting',         path: '/nowcasting' },
      { icon: TrendingUp,  label: 'Forecast Center',    path: '/forecast' },
      { icon: HelpCircle,  label: 'Explainability',     path: '/explain' },
    ],
  },
  {
    group: 'PROTECT',
    items: [
      { icon: Shield,      label: 'Threat Intel',       path: '/threat' },
      { icon: CheckSquare, label: 'Gaganyaan Advisor',  path: '/gaganyaan' },
    ],
  },
  {
    group: 'LEARN',
    items: [
      { icon: Clock,       label: 'Historical Replay',  path: '/replay' },
      { icon: BookOpen,    label: 'Master Catalog',     path: '/catalog' },
      { icon: CheckSquare, label: 'Validation',         path: '/validation' },
    ],
  },
  {
    group: 'OPERATE',
    items: [
      { icon: Cpu,         label: 'Pipeline Health',    path: '/health' },
      { icon: Sliders,     label: 'Configuration',      path: '/config' },
      { icon: ScrollText,  label: 'Mission Chronicle',  path: '/chronicle' },
    ],
  },
];

function AgentNode({ label, sublabel, color, cx, cy, r = 18 }:
  { label: string; sublabel: string; color: string; cx: number; cy: number; r?: number }) {
  return (
    <g>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth="1.5" opacity="0.6" />
      <circle cx={cx} cy={cy} r={r - 6} fill={color} opacity="0.15" />
      <circle cx={cx} cy={cy} r="4" fill={color} opacity="0.9" />
      <text x={cx} y={cy + r + 12} textAnchor="middle" fill="rgba(255,244,232,0.6)"
            fontSize="8" fontFamily="Inter, sans-serif" fontWeight="500">
        {label}
      </text>
      <text x={cx} y={cy + r + 22} textAnchor="middle" fill={color}
            fontSize="7" fontFamily="Inter, sans-serif">
        {sublabel}
      </text>
    </g>
  );
}

function ArunaAgentDiagram() {
  return (
    <div className="px-4 py-3">
      <div className="panel-label mb-2">ARUNA AI · MULTI-AGENT</div>
      <svg width="198" height="140" viewBox="0 0 198 140">
        <line x1="99" y1="32" x2="40" y2="105" stroke="rgba(255,200,87,0.25)"
              strokeWidth="1" strokeDasharray="4 3">
          <animate attributeName="stroke-dashoffset" from="14" to="0" dur="2s" repeatCount="indefinite"/>
        </line>
        <line x1="99" y1="32" x2="158" y2="105" stroke="rgba(0,212,255,0.25)"
              strokeWidth="1" strokeDasharray="4 3">
          <animate attributeName="stroke-dashoffset" from="14" to="0" dur="2s" repeatCount="indefinite"/>
        </line>
        <line x1="40" y1="105" x2="158" y2="105" stroke="rgba(0,196,140,0.20)"
              strokeWidth="1" strokeDasharray="4 3">
          <animate attributeName="stroke-dashoffset" from="14" to="0" dur="2.5s" repeatCount="indefinite"/>
        </line>
        <circle cx="99" cy="32" r="22" fill="none" stroke="#FFC857" strokeWidth="1.5" opacity="0.7">
          <animate attributeName="opacity" values="0.7;1;0.7" dur="2s" repeatCount="indefinite"/>
        </circle>
        <circle cx="99" cy="32" r="14" fill="rgba(255,200,87,0.12)" />
        <circle cx="99" cy="32" r="5" fill="#FFC857" opacity="0.9" />
        <text x="99" y="58" textAnchor="middle" fill="#FFC857"
              fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600">ARUNA AI</text>
        <AgentNode label="PHYSICS"  sublabel="ACTIVE" color="#00D4FF" cx={40}  cy={105} r={16} />
        <AgentNode label="FORECAST" sublabel="ACTIVE" color="#00D4FF" cx={158} cy={105} r={16} />
      </svg>
      <div className="flex items-center gap-1.5 mt-1">
        <span className="w-1.5 h-1.5 rounded-full bg-forecast-green animate-live-dot" />
        <span className="text-[9px] text-forecast-green font-semibold">THREAT AGENT · ACTIVE</span>
      </div>
    </div>
  );
}

export function AppSidebar() {
  const { pathname } = useLocation();
  const [collapsed, setCollapsed] = useState(() => {
    return localStorage.getItem('aruna-sidebar') === 'collapsed';
  });

  const toggle = () => {
    const next = !collapsed;
    setCollapsed(next);
    localStorage.setItem('aruna-sidebar', next ? 'collapsed' : 'expanded');
  };

  return (
    <div className="flex-shrink-0 transition-all duration-300 ease-in-out py-4 pl-4"
         style={{ width: collapsed ? 80 : 276, height: '100%' }}>
      <aside className="flex flex-col h-full overflow-hidden relative"
             style={{ 
               background: 'rgba(11, 16, 32, 0.65)',
               backdropFilter: 'blur(16px)',
               border: '1px solid rgba(255,200,87,0.15)', 
               borderRadius: 16,
               boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
             }}>
        
        {/* Toggle Button */}
        <button onClick={toggle}
                className="absolute top-4 right-4 z-10 p-1 rounded-full hover:bg-white/10 text-sunlit-gold transition-colors">
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>

        {/* Grouped navigation */}
        <nav className="flex-1 pt-12 overflow-y-auto overflow-x-hidden"
             style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
          {NAV_GROUPS.map(({ group, items }) => (
            <div key={group} className="mb-2">
              <div className="px-6 pb-2 text-[9px] font-semibold tracking-widest whitespace-nowrap"
                   style={{ 
                     color: 'rgba(255,200,87,0.40)', 
                     opacity: collapsed ? 0 : 1, 
                     transition: 'opacity 0.2s',
                     height: collapsed ? 0 : 'auto',
                     overflow: 'hidden'
                   }}>
                {group}
              </div>
              {items.map(({ icon: Icon, label, path }) => {
                const active = pathname === path;
                return (
                  <Link key={path} to={path}
                        className="flex items-center gap-4 px-[20px] relative transition-colors duration-150"
                        title={collapsed ? label : undefined}
                        style={{
                          height: 44,
                          background: active ? 'rgba(244,168,37,0.08)' : undefined,
                          color: active ? '#F4A825' : 'rgba(243,241,234,0.48)',
                          boxShadow: active ? 'inset 3px 0 0 #F4A825' : undefined,
                        }}
                        onMouseEnter={e => { if (!active) (e.currentTarget as HTMLElement).style.background = 'rgba(255,200,87,0.04)'; }}
                        onMouseLeave={e => { if (!active) (e.currentTarget as HTMLElement).style.background = ''; }}>
                    <Icon size={18} strokeWidth={1.75} className="flex-shrink-0" />
                    <span className="text-[12px] uppercase whitespace-nowrap transition-opacity duration-200"
                          style={{ 
                            letterSpacing: '0.07em', 
                            fontWeight: 500, 
                            opacity: collapsed ? 0 : 1,
                            pointerEvents: collapsed ? 'none' : 'auto'
                          }}>
                      {label}
                    </span>
                  </Link>
                );
              })}
            </div>
          ))}
        </nav>

        {!collapsed && (
          <div className="animate-in fade-in duration-300">
            {/* AI Agent status */}
            <div className="border-t border-sunlit-gold/10">
              <ArunaAgentDiagram />
            </div>

            {/* Footer branding */}
            <div className="border-t border-sunlit-gold/10 px-6 py-4">
              <div className="text-[12px] font-semibold text-sunlit-gold">ARUNA v2.0</div>
              <div className="text-[11px] italic text-solar-glow/45 mt-0.5">Insight Before Impact.</div>
              <div className="mt-3">
                <div className="panel-label">Data Source</div>
                <div className="telem text-[10px] font-medium text-telemetry-cyan/70 mt-1 leading-tight">
                  Aditya-L1 SoLEXS &amp; HEL1OS L1
                </div>
              </div>
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}
