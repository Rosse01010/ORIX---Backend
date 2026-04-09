import { NavLink } from "react-router-dom";
import { useAuthStore } from "../../store/authStore";
import { cn } from "../../utils/cn";
import type { UserRole } from "../../types/User";

interface NavItem {
  to: string;
  label: string;
  icon: React.ReactNode;
  roles?: UserRole[];
}

const NAV: NavItem[] = [
  {
    to: "/dashboard",
    label: "Dashboard",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="7" height="9" />
        <rect x="14" y="3" width="7" height="5" />
        <rect x="14" y="12" width="7" height="9" />
        <rect x="3" y="16" width="7" height="5" />
      </svg>
    ),
  },
  {
    to: "/enroll",
    label: "Enroll Person",
    roles: ["admin", "operator"],
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <line x1="19" y1="8" x2="19" y2="14" />
        <line x1="22" y1="11" x2="16" y2="11" />
      </svg>
    ),
  },
  {
    to: "/users",
    label: "Users",
    roles: ["admin"],
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
    ),
  },
];

interface Props {
  open: boolean;
  onClose: () => void;
}

/**
 * Collapsible navigation sidebar. On mobile it slides in as an overlay;
 * on md+ it's a static 220px column.
 */
export default function Sidebar({ open, onClose }: Props) {
  const user = useAuthStore((s) => s.user);

  const visible = NAV.filter((item) => {
    if (!item.roles) return true;
    return !!user && item.roles.includes(user.role);
  });

  return (
    <>
      {/* Mobile backdrop */}
      {open && (
        <button
          type="button"
          aria-label="Close sidebar"
          onClick={onClose}
          className="md:hidden fixed inset-0 z-20 bg-black/60"
        />
      )}

      <aside
        className={cn(
          "fixed md:sticky top-[49px] md:top-0 z-30 h-[calc(100vh-49px)] md:h-screen w-56 shrink-0",
          "bg-orix-surface border-r border-orix-border transition-transform",
          open ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        )}
      >
        <div className="p-4 hidden md:block">
          <p className="text-xs uppercase tracking-widest text-zinc-500">
            ORIX
          </p>
          <p className="text-sm font-semibold">Control Center</p>
        </div>
        <nav className="px-2 py-3 space-y-1">
          {visible.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end
              onClick={onClose}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                  isActive
                    ? "bg-orix-accent/20 text-white"
                    : "text-zinc-300 hover:bg-orix-bg"
                )
              }
            >
              {item.icon}
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </aside>
    </>
  );
}
