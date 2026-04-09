import { useAuthStore } from "../../store/authStore";
import { useSocket } from "../../services/socketService";
import { disconnectSocket } from "../../services/socketService";
import Button from "../common/Button";

interface Props {
  onToggleSidebar?: () => void;
}

/**
 * Top navigation bar used inside the DashboardLayout. Shows app name,
 * realtime connection state and the current user, plus a logout button.
 */
export default function Navbar({ onToggleSidebar }: Props) {
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const { connected } = useSocket();

  const onLogout = async () => {
    disconnectSocket();
    await logout();
  };

  return (
    <header className="border-b border-orix-border bg-orix-surface/60 backdrop-blur sticky top-0 z-20">
      <div className="px-3 sm:px-6 py-2 sm:py-3 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          {onToggleSidebar && (
            <button
              type="button"
              onClick={onToggleSidebar}
              className="md:hidden p-1.5 rounded hover:bg-orix-surface text-zinc-300"
              aria-label="Toggle sidebar"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>
          )}
          <h1 className="text-base sm:text-lg font-bold tracking-tight truncate">
            <span className="sm:hidden">ORIX</span>
            <span className="hidden sm:inline">ORIX Dashboard</span>
          </h1>
          <span
            className={`h-2 w-2 rounded-full shrink-0 ${
              connected ? "bg-orix-success" : "bg-orix-danger animate-pulse"
            }`}
            title={connected ? "Socket connected" : "Socket disconnected"}
          />
          <span className="hidden sm:inline text-xs text-zinc-400">
            {connected ? "Live" : "Reconnecting…"}
          </span>
        </div>
        <div className="flex items-center gap-2 sm:gap-4 shrink-0">
          <div className="hidden xs:block text-right leading-tight">
            <p className="text-sm font-medium truncate max-w-[120px]">
              {user?.username}
            </p>
            <p className="text-[11px] uppercase tracking-wider text-zinc-400">
              {user?.role}
            </p>
          </div>
          <Button variant="danger" size="sm" onClick={onLogout}>
            <span className="hidden sm:inline">Logout</span>
            <span className="sm:hidden">Exit</span>
          </Button>
        </div>
      </div>
    </header>
  );
}
