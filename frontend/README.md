# ORIX Frontend

Intelligent Video Surveillance System — frontend.

Real-time camera grid with face detection, role-based login and WebSocket
alerting. Built with **React 19 + TypeScript + Vite + TailwindCSS**.

## Features

- Login with three demo roles (`admin`, `operator`, `user`) and persistent session
- Role-aware protected routes via `react-router-dom`
- Responsive 2×2 camera grid (extensible to N cameras)
- In-browser face detection with [`face-api.js`](https://github.com/justadudewhohacks/face-api.js) and canvas overlays
- Realtime alert stream via Socket.IO with auto-reconnect
- Toast notifications + per-camera alert banners
- Dark, modern UI with Tailwind utility classes
- Graceful error handling when a stream or the socket is unavailable

## Getting Started

### Prerequisites

- Node.js ≥ 18
- npm ≥ 9

### Install & Run

```bash
npm install
npm run dev
```

The app starts at `http://localhost:3000`.

### Demo Credentials

| Username | Password      | Role     |
|----------|---------------|----------|
| admin    | admin123      | admin    |
| operator | operator123   | operator |
| user     | user123       | user     |

### Enabling Face Detection

`face-api.js` requires its model weights at runtime. Download the
`tinyFaceDetector` model files and place them in `public/models/`:

```
public/models/
├── tiny_face_detector_model-weights_manifest.json
└── tiny_face_detector_model-shard1
```

You can grab them from the official repo:
<https://github.com/justadudewhohacks/face-api.js/tree/master/weights>

If the models are missing the app still runs — face detection simply
stays disabled and a warning is logged.

### Backend Expected by the Frontend

The UI assumes a Socket.IO server at `http://localhost:4000` that:

- accepts a connection with `auth: { token }`
- streams `cam1`..`cam4` over HTTP at `/stream/camN`
- emits `alert` events matching `src/types/Alert.ts`
- listens for `face-detected` client events

Override the URL with an `.env.local` file:

```env
VITE_SOCKET_URL=http://localhost:4000
```

## Available Scripts

| Command          | Description                              |
|------------------|------------------------------------------|
| `npm run dev`    | Start Vite dev server on port 3000       |
| `npm run build`  | Type-check and build to `dist/`          |
| `npm run preview`| Preview the production build locally     |
| `npm run lint`   | Run ESLint                               |

## Project Structure

```
src/
├── api/            # (reserved) REST clients
├── assets/         # static assets imported from code
├── components/     # CameraGrid, CameraFeed, AlertBanner
├── layouts/        # (reserved) shared page layouts
├── pages/          # Router + page components
├── services/       # authService, socketService
├── store/          # Zustand stores (authStore)
├── types/          # User, Camera, Alert
├── utils/          # cn, format, storage helpers
├── App.tsx
├── main.tsx
└── index.css
```

## License

Private — All rights reserved.
