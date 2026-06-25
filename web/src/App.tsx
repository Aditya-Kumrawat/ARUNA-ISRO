import { Routes, Route } from "react-router-dom";
import { useSocket } from "./hooks/useSocket";
import MissionControl from "./pages/MissionControl";
import LivePhysicsStreamPage from "./pages/LivePhysicsStreamPage";
import NowcastingPage from "./pages/NowcastingPage";
import ForecastCenterPage from "./pages/ForecastCenterPage";
import ThreatIntelligencePage from "./pages/ThreatIntelligencePage";
import HistoricalReplayPage from "./pages/HistoricalReplayPage";
import MasterCatalogPage from "./pages/MasterCatalogPage";
import ValidationPerformancePage from "./pages/ValidationPerformancePage";
import GaganyaanAdvisorPage from "./pages/GaganyaanAdvisorPage";
import PipelineHealthPage from "./pages/PipelineHealthPage";

// New pages (Plan C — created in Tasks 16-20)
import { lazy, Suspense } from "react";
const ExplainabilityPage = lazy(() => import("./pages/ExplainabilityPage"));
const SolarViewPage = lazy(() => import("./pages/SolarViewPage"));
const ConfigurationPage = lazy(() => import("./pages/ConfigurationPage"));
const MissionChroniclePage = lazy(() => import("./pages/MissionChroniclePage"));

const PageLoader = () => (
  <div className="flex items-center justify-center h-full" style={{ color: 'rgba(244,168,37,0.5)' }}>
    Loading…
  </div>
);

export default function App() {
  useSocket();   // initialises socket + store wiring; call once at app root

  return (
    <Routes>
      <Route path="/"           index element={<MissionControl />} />
      <Route path="/live"       element={<LivePhysicsStreamPage />} />
      <Route path="/nowcasting" element={<NowcastingPage />} />
      <Route path="/forecast"   element={<ForecastCenterPage />} />
      <Route path="/threat"     element={<ThreatIntelligencePage />} />
      <Route path="/replay"     element={<HistoricalReplayPage />} />
      <Route path="/catalog"    element={<MasterCatalogPage />} />
      <Route path="/validation" element={<ValidationPerformancePage />} />
      <Route path="/gaganyaan"  element={<GaganyaanAdvisorPage />} />
      <Route path="/health"     element={<PipelineHealthPage />} />
      {/* New Plan C pages */}
      <Route path="/explain"    element={<Suspense fallback={<PageLoader />}><ExplainabilityPage /></Suspense>} />
      <Route path="/solar"      element={<Suspense fallback={<PageLoader />}><SolarViewPage /></Suspense>} />
      <Route path="/config"     element={<Suspense fallback={<PageLoader />}><ConfigurationPage /></Suspense>} />
      <Route path="/chronicle"  element={<Suspense fallback={<PageLoader />}><MissionChroniclePage /></Suspense>} />
    </Routes>
  );
}
