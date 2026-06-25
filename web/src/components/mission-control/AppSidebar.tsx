import { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import {
  Home, Activity, Radio, TrendingUp, Shield, Clock,
  BookOpen, CheckSquare, Sun, ScrollText, Sliders, HelpCircle, Cpu,
  ChevronLeft, ChevronRight
} from 'lucide-react';

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

  const icons = [
    { Icon: Home, id: '/', label: 'Mission Control' },
    { Icon: Activity, id: '/live', label: 'Live Physics' },
    { Icon: Sun, id: '/solar', label: 'Solar View' },
    { Icon: Radio, id: '/nowcasting', label: 'Nowcasting' },
    { Icon: TrendingUp, id: '/forecast', label: 'Forecast Center' },
    { Icon: HelpCircle, id: '/explain', label: 'Explainability' },
    { Icon: Shield, id: '/threat', label: 'Threat Intel' },
    { Icon: CheckSquare, id: '/gaganyaan', label: 'Gaganyaan' },
    { Icon: Clock, id: '/replay', label: 'Historical Replay' },
    { Icon: BookOpen, id: '/catalog', label: 'Master Catalog' },
    { Icon: Cpu, id: '/health', label: 'Pipeline Health' },
    { Icon: Sliders, id: '/config', label: 'Configuration' },
    { Icon: ScrollText, id: '/chronicle', label: 'Chronicle' },
  ];

  return (
    <>
      {/* Spacer is fixed at 80px so the sidebar floats OVER the content when expanded */}
      <div className="flex-shrink-0" style={{ width: 80 }} />
      
      <div 
        className="fixed left-0 top-0 bottom-0 flex flex-col z-[60] transition-all duration-300"
        style={{ 
          width: collapsed ? 80 : 240, 
          background: '#000000', /* Vanta Black */
          borderRight: '1px solid #1A1A1A', /* Exact border */
          alignItems: collapsed ? 'center' : 'stretch',
          paddingTop: '24px',
          paddingBottom: '24px'
        }}
      >
        <div className={`flex w-full ${collapsed ? 'justify-center' : 'justify-between px-6'} items-center mb-[40px]`}>
          <div className="w-[38px] h-[38px] font-sans font-medium text-[16px] text-sunlit-gold flex items-center justify-center tracking-tighter opacity-90 mt-1 transition-opacity">
            {collapsed ? 'AR' : 'ARUNA'}
          </div>
          {!collapsed && (
            <button onClick={toggle} className="text-white/50 hover:text-white transition-colors">
              <ChevronLeft size={20} />
            </button>
          )}
        </div>

        {collapsed && (
          <button onClick={toggle} className="absolute top-8 right-[-12px] bg-[#000000] border border-[#1A1A1A] rounded-full p-0.5 text-sunlit-gold z-[70]">
            <ChevronRight size={14} />
          </button>
        )}

        <div className={`flex flex-col flex-1 overflow-y-auto w-full ${collapsed ? 'items-center space-y-[24px]' : 'px-4 space-y-2'}`}
             style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
          {icons.map((item, index) => {
            const isActive = pathname === item.id;
            return (
              <Link
                key={index}
                to={item.id}
                title={collapsed ? item.label : undefined}
                className={`relative group cursor-pointer flex items-center ${collapsed ? 'justify-center' : 'px-4 py-3 rounded-lg'} transition-all duration-300`}
                style={{
                  background: !collapsed && isActive ? 'rgba(244,168,37,0.1)' : 'transparent',
                }}
              >
                <div className="relative rounded-full transition-all duration-300">
                  <div className={`mx-auto flex items-center justify-center transition-colors ${isActive ? 'text-sunlit-gold' : 'text-white/50 group-hover:text-white/80'}`}>
                    <item.Icon size={20} strokeWidth={isActive ? 2 : 1.5} />
                  </div>
                </div>
                {!collapsed && (
                  <span className={`ml-4 text-[13px] font-medium tracking-wide transition-colors ${isActive ? 'text-sunlit-gold' : 'text-white/60 group-hover:text-white/90'}`}>
                    {item.label}
                  </span>
                )}
              </Link>
            );
          })}
        </div>
      </div>
    </>
  );
}
