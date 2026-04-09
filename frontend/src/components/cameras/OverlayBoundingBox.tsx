import { useEffect, useRef } from "react";
import type { BoundingBox } from "../../types/Socket";

interface Props {
  boxes: BoundingBox[];
  /** Width of the overlay in display pixels. Should match the <video>. */
  width: number;
  /** Height of the overlay in display pixels. */
  height: number;
  color?: string;
}

/**
 * A lightweight canvas overlay that draws arbitrary bounding boxes on top
 * of a video. Used for server-pushed detections in `detection-result`
 * events, complementing the in-browser face-api overlay.
 */
export default function OverlayBoundingBox({
  boxes,
  width,
  height,
  color = "#22c55e",
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";

    for (const box of boxes) {
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      if (box.label) {
        const label =
          box.confidence != null
            ? `${box.label} ${(box.confidence * 100).toFixed(0)}%`
            : box.label;
        const textY = Math.max(12, box.y - 4);
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(box.x, textY - 12, textWidth + 8, 14);
        ctx.fillStyle = color;
        ctx.fillText(label, box.x + 4, textY);
      }
    }
  }, [boxes, width, height, color]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
    />
  );
}
