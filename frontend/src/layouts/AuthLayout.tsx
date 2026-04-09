import type { ReactNode } from "react";

/**
 * Layout used by unauthenticated screens (login, password reset, etc.).
 * Centers its children on a dark background with a subtle brand header.
 */
export default function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-full flex flex-col">
      <main className="flex-1 flex items-center justify-center px-4 py-6 sm:py-12">
        {children}
      </main>
      <footer className="text-center text-[11px] text-zinc-600 py-3">
        ORIX — Intelligent Video Surveillance System
      </footer>
    </div>
  );
}
