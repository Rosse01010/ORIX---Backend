import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // The ORIX backend only exposes /api/recognition/* (REST + WS).
      // `ws: true` upgrades `/api/recognition/ws` to a WebSocket tunnel
      // so the frontend can hit /api/... same-origin during dev.
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
