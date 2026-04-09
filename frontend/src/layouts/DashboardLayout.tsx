import { useState, type ReactNode } from "react";
import Navbar from "../components/layout/Navbar";
import Sidebar from "../components/layout/Sidebar";
import Footer from "../components/layout/Footer";

/**
 * Layout shared by every authenticated page. Provides a sticky navbar, a
 * collapsible sidebar and a persistent footer; pages render into the
 * central scrollable area.
 */
export default function DashboardLayout({ children }: { children: ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-full flex flex-col">
      <Navbar onToggleSidebar={() => setSidebarOpen((v) => !v)} />

      <div className="flex-1 flex min-h-0">
        <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

        <main className="flex-1 min-w-0 flex flex-col">
          <div className="flex-1 p-2 sm:p-4">{children}</div>
          <Footer />
        </main>
      </div>
    </div>
  );
}
