import type { Alert } from "../../types/Alert";
import { cn } from "../../utils/cn";
import { formatTime } from "../../utils/format";

interface Props {
  alert: Alert;
}

const LEVEL_STYLES: Record<Alert["level"], string> = {
  info: "bg-blue-600/80 border-blue-400/60",
  warning: "bg-amber-600/80 border-amber-400/60",
  critical: "bg-red-600/80 border-red-400/60 animate-pulse",
};

export default function AlertBanner({ alert }: Props) {
  return (
    <div
      role="alert"
      className={cn(
        "rounded border text-white text-xs px-2 py-1 shadow-md backdrop-blur-sm flex justify-between gap-2",
        LEVEL_STYLES[alert.level]
      )}
    >
      <span className="truncate">{alert.message}</span>
      <span className="opacity-70 shrink-0">{formatTime(alert.timestamp)}</span>
    </div>
  );
}
