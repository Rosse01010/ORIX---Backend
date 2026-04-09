/**
 * Footer rendered at the bottom of the DashboardLayout. Intentionally
 * minimal — shows app version and copyright only.
 */
export default function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t border-orix-border bg-orix-surface/40 px-4 py-2 text-center text-[11px] text-zinc-500">
      ORIX v1.0.0 — © {year} All rights reserved
    </footer>
  );
}
