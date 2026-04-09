import CameraFeed from "./CameraFeed";
import { useCameraStore } from "../../store/cameraStore";

/**
 * Renders every camera returned by the backend in a responsive grid.
 * The camera list itself is owned by cameraStore so that other pages
 * (e.g. user management, reports) can share it without refetching.
 */
export default function CameraGrid() {
  const cameras = useCameraStore((s) => s.cameras);
  const loading = useCameraStore((s) => s.loading);
  const error = useCameraStore((s) => s.error);

  if (loading && cameras.length === 0) {
    return (
      <div className="card p-6 text-center text-sm text-zinc-400">
        Loading cameras…
      </div>
    );
  }

  if (error && cameras.length === 0) {
    return (
      <div className="card p-6 text-center text-sm text-red-300">{error}</div>
    );
  }

  if (cameras.length === 0) {
    return (
      <div className="card p-6 text-center text-sm text-zinc-500">
        No cameras configured.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 2xl:grid-cols-3 gap-2 sm:gap-3 lg:gap-4 auto-rows-fr">
      {cameras.map((cam) => (
        <CameraFeed key={cam.id} camera={cam} />
      ))}
    </div>
  );
}
