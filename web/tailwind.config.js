/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        'deep-space':      '#040506',
        'cosmic-indigo':   '#07090B',
        'elevated':        '#0B0D10',
        'card-bg':         '#0C1014',
        'solar-flare':     '#FF8A26',
        'sunlit-gold':     '#F4A825',
        'solar-glow':      '#F3F1EA',
        'emergency-red':   '#FF3B3B',
        'forecast-green':  '#67D46C',
        'confidence-blue': '#3B82F6',
        'telemetry-cyan':  '#00D4FF',
        'warning-amber':   '#F59E0B',
        'graph-bg':        '#07090B',
      },
      fontFamily: {
        brand:     ['Inter', 'sans-serif'],
        ui:        ['Inter', 'sans-serif'],
        telemetry: ['"JetBrains Mono"', 'monospace'],
      },
      borderRadius: {
        'card': '8px',
        'btn':  '4px',
      },
      boxShadow: {
        'card':       '0 8px 24px rgba(0,0,0,0.35)',
        'card-alert': '0 8px 24px rgba(255,138,0,0.12)',
        'hope':       '0 0 24px rgba(255,200,87,0.10)',
        'glow-gold':  '0 0 16px rgba(255,200,87,0.15)',
        'glow-orange':'0 0 20px rgba(255,138,0,0.15)',
        'glow-cyan':  '0 0 12px rgba(0,212,255,0.10)',
      },
      animation: {
        'hope-pulse':    'hope-pulse 2s ease-in-out infinite',
        'live-dot':      'live-dot 1s ease-in-out infinite',
        'telemetry-upd': 'telemetry-update 1s linear',
        'orbit':         'orbit 60s linear infinite',
        'risk-ring':     'risk-ring 3s ease-out infinite',
        'dash-flow':     'dash-flow 2s linear infinite',
      },
      keyframes: {
        'hope-pulse': {
          '0%, 100%': { boxShadow: '0 0 12px rgba(255,200,87,0.20)' },
          '50%':       { boxShadow: '0 0 28px rgba(255,200,87,0.50)' },
        },
        'live-dot': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%':       { opacity: '0.4', transform: 'scale(0.75)' },
        },
        'telemetry-update': {
          '0%':  { opacity: '1' },
          '15%': { opacity: '0.4' },
          '30%': { opacity: '1' },
        },
        'orbit': {
          from: { transform: 'rotate(0deg)' },
          to:   { transform: 'rotate(360deg)' },
        },
        'risk-ring': {
          '0%':   { transform: 'scale(1)', opacity: '0.8' },
          '100%': { transform: 'scale(2.5)', opacity: '0' },
        },
        'dash-flow': {
          from: { strokeDashoffset: '20' },
          to:   { strokeDashoffset: '0' },
        },
      },
    },
  },
  plugins: [],
};
