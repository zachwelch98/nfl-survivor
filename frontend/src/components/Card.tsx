import type { ReactNode } from "react";

export function Card({ title, subtitle, children, className = "" }: { title?: string; subtitle?: string; children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-slate-200 bg-white shadow-sm ${className}`}>
      {(title || subtitle) && (
        <div className="px-4 sm:px-5 pt-4 pb-2">
          {title && <h2 className="text-base font-semibold text-ink">{title}</h2>}
          {subtitle && <p className="text-sm text-slate-500 mt-0.5">{subtitle}</p>}
        </div>
      )}
      <div className="px-4 sm:px-5 pb-4 sm:pb-5">{children}</div>
    </div>
  );
}

export function StatTile({ label, value, tone = "default" }: { label: string; value: ReactNode; tone?: "default" | "good" | "bad" }) {
  const toneClass = tone === "good" ? "text-win" : tone === "bad" ? "text-loss" : "text-ink";
  return (
    <div className="rounded-lg bg-slate-50 border border-slate-200 px-3 py-2.5">
      <div className="text-xs text-slate-500">{label}</div>
      <div className={`text-xl font-semibold tabular-nums ${toneClass}`}>{value}</div>
    </div>
  );
}
