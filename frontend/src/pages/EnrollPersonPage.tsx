import { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";
import Button from "../components/common/Button";
import { registerPerson } from "../api/recognitionApi";
import type { PersonCreatePayload } from "../api/recognitionApi";

/**
 * Consented enrollment flow.
 *
 * A person opens this page, faces the webcam, and the browser computes a
 * 128-dim face-api.js embedding locally. They fill in their own social
 * handles and we POST everything to the backend. The embedding never
 * leaves the client except as a normalized vector destined for their
 * own record — there is no auto-discovery of strangers' profiles.
 */

type ModelsState = "loading" | "ready" | "failed";

/** Lazily load face-api.js weights (all shared with CameraFeed). */
async function loadModels(): Promise<void> {
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
    faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
    faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  ]);
}

interface FormState {
  name: string;
  linkedin_url: string;
  instagram_handle: string;
  twitter_handle: string;
  notes: string;
}

const EMPTY_FORM: FormState = {
  name: "",
  linkedin_url: "",
  instagram_handle: "",
  twitter_handle: "",
  notes: "",
};

export default function EnrollPersonPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [modelsState, setModelsState] = useState<ModelsState>("loading");
  const [streamError, setStreamError] = useState<string | null>(null);
  const [embedding, setEmbedding] = useState<number[] | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load face-api weights once.
  useEffect(() => {
    let cancelled = false;
    loadModels()
      .then(() => {
        if (!cancelled) setModelsState("ready");
      })
      .catch((err) => {
        // eslint-disable-next-line no-console
        console.error("[enroll] failed to load models:", err);
        if (!cancelled) setModelsState("failed");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Attach the webcam stream.
  useEffect(() => {
    let cancelled = false;
    let mediaStream: MediaStream | null = null;

    navigator.mediaDevices
      ?.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        mediaStream = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {
            setStreamError("Autoplay blocked");
          });
        }
      })
      .catch((err) => {
        // eslint-disable-next-line no-console
        console.warn("[enroll] getUserMedia failed:", err);
        setStreamError("Webcam permission denied");
      });

    return () => {
      cancelled = true;
      if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  async function handleCapture() {
    if (modelsState !== "ready" || !videoRef.current) return;
    setCapturing(true);
    setError(null);
    setMessage(null);

    try {
      const detection = await faceapi
        .detectSingleFace(
          videoRef.current,
          new faceapi.TinyFaceDetectorOptions()
        )
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!detection) {
        setError("No face detected — try centering your face in the frame.");
        setEmbedding(null);
        return;
      }

      // face-api.js returns a Float32Array descriptor. pgvector wants a
      // plain JS array; we also L2-normalize for consistency with the
      // backend's cosine-distance search (even though face-api.js
      // descriptors are already roughly unit-norm).
      const raw = Array.from(detection.descriptor);
      const norm = Math.sqrt(raw.reduce((s, v) => s + v * v, 0)) || 1;
      const normalized = raw.map((v) => v / norm);
      setEmbedding(normalized);
      setMessage("Face captured. Fill out the form and submit.");

      // Draw the bounding box as visual feedback.
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (canvas && video) {
        const displaySize = {
          width: video.clientWidth,
          height: video.clientHeight,
        };
        faceapi.matchDimensions(canvas, displaySize);
        const resized = faceapi.resizeResults(detection, displaySize);
        const ctx = canvas.getContext("2d");
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, [resized.detection]);
        faceapi.draw.drawFaceLandmarks(canvas, [resized]);
      }
    } finally {
      setCapturing(false);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!embedding) {
      setError("Capture your face first.");
      return;
    }
    if (!form.name.trim()) {
      setError("Name is required.");
      return;
    }

    setSubmitting(true);
    setError(null);
    setMessage(null);
    try {
      const payload: PersonCreatePayload = {
        name: form.name.trim(),
        embedding,
        linkedin_url: form.linkedin_url.trim() || null,
        instagram_handle: form.instagram_handle.trim() || null,
        twitter_handle: form.twitter_handle.trim() || null,
        notes: form.notes.trim() || null,
      };
      const res = await registerPerson(payload);
      setMessage(`Enrolled "${res.name}" (id: ${res.person_id}).`);
      setForm(EMPTY_FORM);
      setEmbedding(null);
      const canvas = canvasRef.current;
      canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Unknown error submitting enrollment";
      setError(`Submit failed: ${msg}`);
    } finally {
      setSubmitting(false);
    }
  }

  function updateField<K extends keyof FormState>(key: K, value: string) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <div className="h-full grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* ── Webcam panel ─────────────────────────────────────────────── */}
      <div className="card p-4 flex flex-col gap-3">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">
            Capture
          </h2>
          <p className="text-xs text-zinc-500 mt-1">
            Position yourself in front of the camera and click{" "}
            <span className="text-zinc-300">Capture face</span>. The embedding
            is computed locally in your browser and only submitted with the
            form below — nothing is uploaded until you click{" "}
            <span className="text-zinc-300">Enroll</span>.
          </p>
        </div>

        <div className="relative bg-black rounded-md overflow-hidden aspect-video">
          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            muted
            playsInline
            autoPlay
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
          />
          {streamError && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/70">
              <p className="text-red-400 text-sm">{streamError}</p>
            </div>
          )}
          {modelsState === "loading" && !streamError && (
            <div className="absolute top-2 right-2 bg-black/70 px-2 py-1 rounded text-[10px] text-zinc-300">
              Loading models…
            </div>
          )}
          {modelsState === "failed" && (
            <div className="absolute top-2 right-2 bg-red-900/80 px-2 py-1 rounded text-[10px] text-red-200">
              Models failed to load
            </div>
          )}
        </div>

        <Button
          onClick={handleCapture}
          disabled={modelsState !== "ready" || !!streamError || capturing}
        >
          {capturing ? "Capturing…" : "Capture face"}
        </Button>

        {embedding && (
          <p className="text-xs text-green-400">
            ✓ Embedding ready ({embedding.length} dims)
          </p>
        )}
      </div>

      {/* ── Form panel ───────────────────────────────────────────────── */}
      <form
        onSubmit={handleSubmit}
        className="card p-4 flex flex-col gap-3 text-sm"
      >
        <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">
          Person details
        </h2>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">Name *</span>
          <input
            className="bg-orix-bg border border-orix-border rounded px-2 py-1.5 text-sm"
            value={form.name}
            onChange={(e) => updateField("name", e.target.value)}
            placeholder="Jane Doe"
            required
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">LinkedIn URL</span>
          <input
            className="bg-orix-bg border border-orix-border rounded px-2 py-1.5 text-sm"
            value={form.linkedin_url}
            onChange={(e) => updateField("linkedin_url", e.target.value)}
            placeholder="https://www.linkedin.com/in/jane-doe/"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">Instagram handle</span>
          <input
            className="bg-orix-bg border border-orix-border rounded px-2 py-1.5 text-sm"
            value={form.instagram_handle}
            onChange={(e) => updateField("instagram_handle", e.target.value)}
            placeholder="@janedoe"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">X / Twitter handle</span>
          <input
            className="bg-orix-bg border border-orix-border rounded px-2 py-1.5 text-sm"
            value={form.twitter_handle}
            onChange={(e) => updateField("twitter_handle", e.target.value)}
            placeholder="@janedoe"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">Notes</span>
          <textarea
            className="bg-orix-bg border border-orix-border rounded px-2 py-1.5 text-sm min-h-[80px]"
            value={form.notes}
            onChange={(e) => updateField("notes", e.target.value)}
            placeholder="Optional free-form notes (role, department, etc.)"
          />
        </label>

        <p className="text-[11px] text-zinc-500 leading-relaxed">
          By enrolling, you confirm the person in front of the camera
          consents to having their face embedding and the social handles
          above stored in this system for future identification.
        </p>

        <Button type="submit" disabled={submitting || !embedding}>
          {submitting ? "Enrolling…" : "Enroll"}
        </Button>

        {message && <p className="text-xs text-green-400">{message}</p>}
        {error && <p className="text-xs text-red-400">{error}</p>}
      </form>
    </div>
  );
}
