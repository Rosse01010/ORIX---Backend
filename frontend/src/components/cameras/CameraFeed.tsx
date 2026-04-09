import { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";
import AlertBanner from "../alerts/AlertBanner";
import OverlayBoundingBox from "./OverlayBoundingBox";
import { useSocket } from "../../services/socketService";
import { buildLocalAlert, notifyAlert } from "../../services/alertService";
import { useAlertStore } from "../../store/alertStore";
import type { Camera } from "../../types/Camera";
import type { BoundingBox } from "../../types/Socket";

interface Props {
  camera: Camera;
}

/**
 * Load the tiny face detector exactly once per app lifetime. Models live
 * in /public/models; see the README for download instructions. If the
 * weights are missing we silently disable detection — the rest of the UI
 * keeps working.
 */
let modelsLoadingPromise: Promise<void> | null = null;
function ensureModelsLoaded(): Promise<void> {
  if (modelsLoadingPromise) return modelsLoadingPromise;
  modelsLoadingPromise = faceapi.nets.tinyFaceDetector
    .loadFromUri("/models")
    .catch((err) => {
      // eslint-disable-next-line no-console
      console.warn(
        "[face-api] Could not load tinyFaceDetector from /models — face detection disabled.",
        err
      );
    });
  return modelsLoadingPromise;
}

export default function CameraFeed({ camera }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { socket } = useSocket();
  const pushAlert = useAlertStore((s) => s.push);

  const [banners, setBanners] = useState<
    ReturnType<typeof buildLocalAlert>[]
  >([]);
  const [serverBoxes, setServerBoxes] = useState<BoundingBox[]>([]);
  const [overlaySize, setOverlaySize] = useState({ width: 0, height: 0 });
  const [videoError, setVideoError] = useState<string | null>(null);
  const [modelsReady, setModelsReady] = useState(false);

  // Load face-api models exactly once per app.
  useEffect(() => {
    let cancelled = false;
    ensureModelsLoaded().then(() => {
      if (!cancelled) setModelsReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Attach the stream to the <video> element and surface load failures
  // instead of showing a silent black square. Two sources are supported:
  //   - `webcam://` → local device camera via getUserMedia
  //   - anything else → treated as an http(s) video URL
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    setVideoError(null);
    let mediaStream: MediaStream | null = null;
    let cancelled = false;

    const onError = () => setVideoError("Video stream unavailable");
    const onResize = () => {
      setOverlaySize({
        width: video.clientWidth,
        height: video.clientHeight,
      });
    };
    video.addEventListener("error", onError);
    video.addEventListener("loadedmetadata", onResize);

    if (camera.streamUrl.startsWith("webcam://")) {
      if (!navigator.mediaDevices?.getUserMedia) {
        setVideoError("getUserMedia not supported");
      } else {
        navigator.mediaDevices
          .getUserMedia({ video: true, audio: false })
          .then((stream) => {
            if (cancelled) {
              stream.getTracks().forEach((t) => t.stop());
              return;
            }
            mediaStream = stream;
            video.srcObject = stream;
            video.play().catch(() => setVideoError("Autoplay blocked"));
          })
          .catch((err) => {
            // eslint-disable-next-line no-console
            console.warn("[webcam] getUserMedia failed:", err);
            setVideoError("Webcam permission denied");
          });
      }
    } else {
      video.src = camera.streamUrl;
      video.play().catch(() => setVideoError("Autoplay blocked"));
    }

    // Tell the backend we care about detections for this camera.
    socket.emit("subscribe-camera", { cameraId: camera.id });

    return () => {
      cancelled = true;
      socket.emit("unsubscribe-camera", { cameraId: camera.id });
      video.removeEventListener("error", onError);
      video.removeEventListener("loadedmetadata", onResize);
      video.pause();
      if (mediaStream) {
        mediaStream.getTracks().forEach((t) => t.stop());
        video.srcObject = null;
      } else {
        video.removeAttribute("src");
        video.load();
      }
    };
  }, [camera.id, camera.streamUrl, socket]);

  // Subscribe to backend-emitted alerts targeting this specific camera.
  // The global DashboardPage also pushes alerts to the store; here we
  // only render the transient in-frame banners.
  useEffect(() => {
    const onAlert = (alert: Parameters<typeof pushAlert>[0]) => {
      if (alert.cameraId !== camera.id) return;
      setBanners((prev) => [alert, ...prev].slice(0, 3));
    };
    socket.on("alert", onAlert);
    return () => {
      socket.off("alert", onAlert);
    };
  }, [camera.id, pushAlert, socket]);

  // Subscribe to server-pushed detection bounding boxes.
  useEffect(() => {
    const onDetection = (payload: {
      cameraId: string;
      boxes: BoundingBox[];
    }) => {
      if (payload.cameraId !== camera.id) return;
      setServerBoxes(payload.boxes);
    };
    socket.on("detection-result", onDetection);
    return () => {
      socket.off("detection-result", onDetection);
    };
  }, [camera.id, socket]);

  // In-browser face detection loop (1 FPS) as a fallback when the backend
  // does not stream detection results. Only runs once the weights loaded
  // AND when the source is the local webcam — face-api.js can't read
  // pixels from cross-origin <video> elements (tainted canvas), so for
  // remote streams we rely entirely on the backend's detection pipeline.
  const isLocalWebcam = camera.streamUrl.startsWith("webcam://");
  useEffect(() => {
    if (!modelsReady || videoError || !isLocalWebcam) return;

    const interval = setInterval(async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      const displaySize = {
        width: video.clientWidth,
        height: video.clientHeight,
      };
      faceapi.matchDimensions(canvas, displaySize);

      try {
        const detections = await faceapi.detectAllFaces(
          video,
          new faceapi.TinyFaceDetectorOptions()
        );
        const resized = faceapi.resizeResults(detections, displaySize);
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, resized);

        if (detections.length > 0) {
          const alert = buildLocalAlert({
            cameraId: camera.id,
            type: "face-detected",
            level: "info",
            message: `${detections.length} face(s) detected`,
            meta: { count: detections.length },
          });
          setBanners((prev) => [alert, ...prev].slice(0, 3));
          pushAlert(alert);
          notifyAlert(alert, camera.name);
          socket.emit("face-detected", {
            cameraId: camera.id,
            count: detections.length,
          });
        }
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn(`[${camera.id}] detection failed`, err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [camera.id, camera.name, modelsReady, videoError, socket, pushAlert]);

  return (
    <div className="card relative overflow-hidden flex flex-col">
      <div className="flex items-center justify-between gap-2 px-2 sm:px-3 py-1.5 sm:py-2 border-b border-orix-border bg-orix-bg/50">
        <div className="min-w-0 flex-1">
          <p className="text-xs sm:text-sm font-medium truncate">
            {camera.name}
          </p>
          <p className="text-[10px] sm:text-[11px] text-zinc-500 truncate">
            {camera.location}
          </p>
        </div>
        <span
          className={`text-[9px] sm:text-[10px] uppercase px-1.5 sm:px-2 py-0.5 rounded shrink-0 ${
            videoError
              ? "bg-red-900/60 text-red-200"
              : "bg-green-900/60 text-green-200"
          }`}
        >
          {videoError ? "offline" : camera.status}
        </span>
      </div>

      <div className="relative flex-1 bg-black aspect-video sm:aspect-auto sm:min-h-[200px] md:min-h-[240px]">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          muted
          playsInline
          autoPlay
        />
        {/* local face-api overlay */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
        {/* server-pushed detections overlay */}
        {overlaySize.width > 0 && serverBoxes.length > 0 && (
          <OverlayBoundingBox
            boxes={serverBoxes}
            width={overlaySize.width}
            height={overlaySize.height}
          />
        )}

        {videoError && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70 p-3">
            <div className="text-center">
              <p className="text-red-400 font-semibold text-xs sm:text-sm">
                {videoError}
              </p>
              <p className="text-[10px] sm:text-xs text-zinc-500 mt-1 break-all">
                {camera.streamUrl}
              </p>
            </div>
          </div>
        )}

        <div className="absolute bottom-1.5 sm:bottom-2 left-1.5 right-1.5 sm:left-2 sm:right-2 flex flex-col gap-1 pointer-events-none">
          {banners.map((a) => (
            <AlertBanner key={a.id} alert={a} />
          ))}
        </div>
      </div>
    </div>
  );
}
