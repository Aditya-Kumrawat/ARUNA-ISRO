/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        aruna: {
          bg:      "#0a0e1a",
          surface: "#111827",
          border:  "#1f2937",
          accent:  "#3b82f6",
          warning: "#f59e0b",
          danger:  "#ef4444",
          success: "#10b981",
        },
      },
    },
  },
  plugins: [],
};
