import { useEffect, useState } from "react";
import CameraGrid from "../components/cameras/CameraGrid";
import SimilarityPanel from "../components/similarity/SimilarityPanel";
import MatchedPersonCard from "../components/alerts/MatchedPersonCard";
import { useSocket } from "../services/socketService";
import { useAlertStore } from "../store/alertStore";
import { useCameraStore } from "../store/cameraStore";
import { useCandidateStore } from "../store/candidateStore";
import { formatTime } from "../utils/format";
import { notifyAlert } from "../services/alertService";
import type { FaceCandidate } from "../types/Candidate";

export default function DashboardPage() {
  const { socket } = useSocket();

  const cameras         = useCameraStore((s) => s.cameras);
  const fetchCameras    = useCameraStore((s) => s.fetchAll);
  const setCameraStatus = useCameraStore((s) => s.setStatus);

  const alerts    = useAlertStore((s) => s.alerts);
  const pushAlert = useAlertStore((s) => s.push);

  const addCandidates  = useCandidateStore((s) => s.addCandidates);
  const pendingCount   = useCandidateStore((s) => s.pending.length);

  // Currently expanded matched person (shows the social card below).
  const [expandedPersonId, setExpandedPersonId] = useState<string | null>(null);

  // Load cameras on mount
  useEffect(() => {
    fetchCameras();
  }, [fetchCameras]);

  // Global alert subscription
  useEffect(() => {
    const onAlert = (alert: Parameters<typeof pushAlert>[0]) => {
      pushAlert(alert);
      const cam = cameras.find((c) => c.id === alert.cameraId);
      notifyAlert(alert, cam?.name);
    };
    socket.on("alert", onAlert);
    return () => { socket.off("alert", onAlert); };
  }, [socket, pushAlert, cameras]);

  // Camera status updates
  useEffect(() => {
    const onStatus = (payload: { cameraId: string; status: string }) => {
      setCameraStatus(payload.cameraId, payload.status as Parameters<typeof setCameraStatus>[1]);
    };
    socket.on("camera-status", onStatus);
    return () => { socket.off("camera-status", onStatus); };
  }, [socket, setCameraStatus]);

  // ── Similarity panel: collect candidates from every detection event ────────
  useEffect(() => {
    const onDetection = (payload: {
      cameraId: string;
      boxes: unknown[];
      candidates: FaceCandidate[];
    }) => {
      if (!payload.candidates || payload.candidates.length === 0) return;

      // Stamp each candidate with the camera and timestamp
      const stamped: FaceCandidate[] = payload.candidates.map((c) => ({
        ...c,
        cameraId: payload.cameraId,
        timestamp: new Date().toISOString(),
      }));
      addCandidates(stamped);
    };

    socket.on("detection-result", onDetection);
    return () => { socket.off("detection-result", onDetection); };
  }, [socket, addCandidates]);

  return (
    <>
      <div
        className={`h-full grid gap-3 sm:gap-4 transition-all ${
          pendingCount > 0
            ? "grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px]"
            : "grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px]"
        }`}
      >
        {/* ── Camera grid ───────────────────────────────────────── */}
        <CameraGrid />

        {/* ── Recent alerts sidebar ─────────────────────────────── */}
        <aside className="card p-3 sm:p-4 flex flex-col min-h-0
                           max-h-[40vh] xl:max-h-none">
          <div className="flex items-center justify-between mb-2 sm:mb-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">
              Recent Alerts
            </h2>
            {pendingCount > 0 && (
              <span className="text-[10px] bg-orange-600 text-white px-2 py-0.5
                               rounded-full font-medium animate-pulse">
                {pendingCount} pending
              </span>
            )}
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 pr-1">
            {alerts.length === 0 ? (
              <p className="text-xs text-zinc-500">No alerts yet.</p>
            ) : (
              alerts.map((a) => {
                const isMatched =
                  !!a.personId && a.meta?.status === "matched";
                const isExpanded =
                  isMatched && expandedPersonId === a.personId;
                return (
                  <div key={a.id} className="space-y-1">
                    <button
                      type="button"
                      disabled={!isMatched}
                      onClick={() =>
                        isMatched &&
                        setExpandedPersonId((prev) =>
                          prev === a.personId ? null : a.personId ?? null
                        )
                      }
                      className={`w-full text-left rounded-md border p-2 text-xs transition-colors ${
                        isMatched
                          ? "border-orix-accent/40 bg-orix-bg/50 hover:bg-orix-bg/80 cursor-pointer"
                          : "border-orix-border bg-orix-bg/50 cursor-default"
                      }`}
                    >
                      <div className="flex justify-between gap-2 text-zinc-400">
                        <span className="uppercase truncate">{a.type}</span>
                        <span className="shrink-0">
                          {formatTime(a.timestamp)}
                        </span>
                      </div>
                      <p className="text-white mt-1 break-words">{a.message}</p>
                      {isMatched && !isExpanded && (
                        <p className="text-[10px] text-orix-accent mt-1">
                          Click to see profile →
                        </p>
                      )}
                    </button>
                    {isExpanded && a.personId && (
                      <MatchedPersonCard
                        personId={a.personId}
                        onClose={() => setExpandedPersonId(null)}
                      />
                    )}
                  </div>
                );
              })
            )}
          </div>
        </aside>
      </div>

      {/* ── Similarity panel (slides in from right when candidates exist) ── */}
      <SimilarityPanel />
    </>
  );
}
