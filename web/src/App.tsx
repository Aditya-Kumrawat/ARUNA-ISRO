import { Routes, Route } from "react-router-dom";

// Pages — stubs to be implemented in Phase 12
const MissionControl     = () => <div>Mission Control — Phase 12</div>;
const LiveSolarView      = () => <div>Live Solar View — Phase 12</div>;
const PhysicsDiagnostics = () => <div>Physics Diagnostics — Phase 12</div>;
const Nowcasting         = () => <div>Nowcasting — Phase 12</div>;
const ForecastCenter     = () => <div>Forecast Center — Phase 12</div>;
const ThreatIntelligence = () => <div>Threat Intelligence — Phase 12</div>;
const HistoricalReplay   = () => <div>Historical Replay — Phase 12</div>;
const FlareCatalog       = () => <div>Master Flare Catalog — Phase 12</div>;
const PipelineHealth     = () => <div>Pipeline Health — Phase 12</div>;
const Validation         = () => <div>Validation — Phase 12</div>;
const ModelPerformance   = () => <div>Model Performance — Phase 12</div>;
const GaganyaanAdvisor   = () => <div>Gaganyaan Window Advisor — Phase 12</div>;
const Configuration      = () => <div>Configuration — Phase 12</div>;
const Logs               = () => <div>Logs — Phase 12</div>;

export default function App() {
  return (
    <Routes>
      <Route path="/"                index element={<MissionControl />} />
      <Route path="/live"            element={<LiveSolarView />} />
      <Route path="/physics"         element={<PhysicsDiagnostics />} />
      <Route path="/nowcasting"      element={<Nowcasting />} />
      <Route path="/forecast"        element={<ForecastCenter />} />
      <Route path="/threat"          element={<ThreatIntelligence />} />
      <Route path="/replay"          element={<HistoricalReplay />} />
      <Route path="/catalog"         element={<FlareCatalog />} />
      <Route path="/health"          element={<PipelineHealth />} />
      <Route path="/validation"      element={<Validation />} />
      <Route path="/models"          element={<ModelPerformance />} />
      <Route path="/gaganyaan"       element={<GaganyaanAdvisor />} />
      <Route path="/config"          element={<Configuration />} />
      <Route path="/logs"            element={<Logs />} />
    </Routes>
  );
}
