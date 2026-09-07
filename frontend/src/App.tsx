import { NavLink, Route, Routes } from "react-router-dom";
import Dashboard from "./views/Dashboard";
import WeeklyBoard from "./views/WeeklyBoard";
import SeasonPlanView from "./views/SeasonPlanView";
import MyPicks from "./views/MyPicks";
import AnalysisView from "./views/AnalysisView";
import SimulationView from "./views/SimulationView";

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/board", label: "Odds Board" },
  { to: "/plan", label: "Season Plan" },
  { to: "/picks", label: "My Picks" },
  { to: "/analysis", label: "Analysis" },
  { to: "/simulation", label: "Simulation" },
];

function navClass(isActive: boolean): string {
  return [
    "px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
    isActive ? "bg-ink text-white" : "text-slate-600 hover:bg-slate-100 hover:text-ink",
  ].join(" ");
}

export default function App() {
  return (
    <div className="min-h-full flex flex-col">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur sticky top-0 z-10">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 py-3 flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold tracking-tight">🏈 Survivor Advisor</span>
          </div>
          <nav className="flex items-center gap-1 flex-wrap">
            {NAV.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.end} className={({ isActive }) => navClass(isActive)}>
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="flex-1 mx-auto w-full max-w-6xl px-4 sm:px-6 py-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/board" element={<WeeklyBoard />} />
          <Route path="/plan" element={<SeasonPlanView />} />
          <Route path="/picks" element={<MyPicks />} />
          <Route path="/analysis" element={<AnalysisView />} />
          <Route path="/simulation" element={<SimulationView />} />
        </Routes>
      </main>

      <footer className="border-t border-slate-200 py-4 text-center text-xs text-slate-400">
        Built on nflverse data. Not betting advice.
      </footer>
    </div>
  );
}
