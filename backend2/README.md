# ORIX Face Recognition Backend

Real-time facial recognition system built with **FastAPI**, **InsightFace (RetinaFace + ArcFace)**, **Redis Streams**, **PostgreSQL + pgvector**, and **WebSockets**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Docker Network                        │
│                                                             │
│  ┌──────────────┐    Redis Streams    ┌──────────────────┐  │
│  │ camera_worker│ ──stream:frames──► │   gpu_worker     │  │
│  │ (per camera) │                    │ (InsightFace GPU) │  │
│  └──────────────┘                    └────────┬─────────┘  │
│                                               │             │
│                                    stream:vectors           │
│                                               ▼             │
│                                      ┌──────────────┐       │
│                                      │  db_worker   │       │
│                                      │ (pgvector NN)│       │
│                                      └──────┬───────┘       │
│                                             │               │
│                                    stream:events            │
│                                             ▼               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FastAPI (api)                           │   │
│  │   GET /health   POST /recognize   GET /persons      │   │
│  │   POST /persons  DELETE /persons/{id}               │   │
│  │   WS  /ws/detections  (broadcasts to React)         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Requirements

- Docker ≥ 24 + Docker Compose v2
- NVIDIA GPU + CUDA 12.3 drivers (for `worker` service)
- `nvidia-container-toolkit` installed on the host

---

## Installation

```bash
# 1. Clone and enter the project
cd backend2/

# 2. Create environment file
cp .env.example .env
# Edit .env: set CAMERA_SOURCES, DB/Redis passwords, thresholds, etc.

# 3. Build and start all services
docker compose up --build -d

# 4. Check logs
docker compose logs -f api
docker compose logs -f worker
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `CAMERA_SOURCES` | `0` | Comma-separated: `0,rtsp://user:pass@ip/stream` |
| `CAMERA_FRAME_RATE` | `15` | FPS for capture |
| `DETECTOR_BACKEND` | `insightface` | `insightface` \| `mediapipe` |
| `EMBEDDER_BACKEND` | `arcface` | `arcface` \| `facenet` |
| `INSIGHTFACE_MODEL` | `buffalo_l` | `buffalo_l` \| `buffalo_m` \| `buffalo_s` |
| `SIMILARITY_THRESHOLD` | `0.45` | Cosine similarity cutoff (0–1) |
| `DETECTION_CONFIDENCE` | `0.85` | Min face detection score |
| `GPU_DEVICE_ID` | `0` | CUDA device index |
| `USE_GPU` | `true` | `true` \| `false` |
| `GPU_WORKER_BATCH_SIZE` | `4` | Frames processed per batch |
| `STREAM_MAX_LEN` | `1000` | Max messages kept in each stream |

---

## API Reference

### Health

```
GET /health
→ {"status": "ok"}

GET /health/detailed
→ {"status": "ok", "postgres": "ok", "redis": "ok", "websocket_clients": 2, "uptime_seconds": 120.5}
```

### Persons (Known Faces)

```
GET    /api/persons                     – list all active persons
POST   /api/persons  (multipart)        – register person (name + face photo)
DELETE /api/persons/{id}               – soft-delete person
```

### Recognition (single image)

```
POST /api/recognize  (multipart: camera=string, file=image)
→ {
    "camera": "upload",
    "timestamp": "2026-04-08T12:00:00+00:00",
    "bboxes": [
      {"x": 120, "y": 60, "width": 100, "height": 100, "name": "Carlos", "confidence": 0.87}
    ]
  }
```

---

## WebSocket

Connect from your React app:

```js
const ws = new WebSocket("ws://localhost:8000/ws/detections");
// Optional: filter by camera
// const ws = new WebSocket("ws://localhost:8000/ws/detections?cameras=cam_00,cam_01");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "ping") return;
  console.log(data);
  // {
  //   "camera": "cam_00",
  //   "timestamp": "2026-04-08T12:00:00+00:00",
  //   "bboxes": [
  //     {"x": 120, "y": 60, "width": 100, "height": 100, "name": "Carlos", "confidence": 0.87},
  //     {"x": 300, "y": 80, "width": 95,  "height": 95,  "name": "Unknown", "confidence": 0.65}
  //   ]
  // }
};
```

---

## Local Demo (no Docker)

```bash
# Requires local PostgreSQL + Redis running

pip install -r requirements.txt
cp .env.example .env   # update DATABASE_URL and REDIS_URL to localhost

python run_demo.py
```

The demo script:
1. Creates DB tables and seeds a "Demo Person".
2. Starts API on `http://127.0.0.1:8000`.
3. Starts the worker supervisor (camera → GPU → DB pipeline).
4. Connects a WebSocket and prints detection events for 30 seconds.

---

## Registering a Real Person via curl

```bash
# Register
curl -X POST http://localhost:8000/api/persons \
  -F "name=Carlos" \
  -F "file=@carlos.jpg"

# List
curl http://localhost:8000/api/persons

# Recognize from image
curl -X POST http://localhost:8000/api/recognize \
  -F "camera=test_cam" \
  -F "file=@test.jpg"
```

---

## Project Structure

```
backend2/
├── .env.example
├── docker-compose.yml
├── Dockerfile.api          # lightweight Python image for FastAPI
├── Dockerfile.worker       # CUDA image for GPU workers
├── requirements.txt
├── run_demo.py             # local integration demo
├── scripts/
│   └── init_db.sql         # pgvector extension setup
└── app/
    ├── config.py           # pydantic-settings (env vars)
    ├── database.py         # async SQLAlchemy + pgvector
    ├── main.py             # FastAPI app + WebSocket endpoint
    ├── models.py           # Person + DetectionLog ORM models
    ├── routes/
    │   ├── health.py
    │   └── recognition.py
    └── websocket/
        ├── manager.py      # ConnectionManager (broadcast)
        └── notifications.py # Redis → WebSocket relay task
workers/
    ├── main_worker.py      # process supervisor
    ├── camera_worker.py    # RTSP/USB → stream:frames
    ├── gpu_worker.py       # stream:frames → detect+embed → stream:vectors
    └── db_worker.py        # stream:vectors → pgvector search → stream:events
utils/
    ├── gpu_utils.py        # InsightFace / MediaPipe factories
    ├── preprocessing.py    # frame resize, normalization
    └── logging_utils.py    # structlog configuration
```

---

## Extending the System

| Concern | Where to extend |
|---|---|
| Add a new camera | Append to `CAMERA_SOURCES` in `.env` |
| Add analytics | Subscribe a new worker to `stream:events` |
| Multi-tenant support | Add `tenant_id` column to `persons` and filter in `db_worker` |
| Tracking (person re-ID) | Add a tracking module between `gpu_worker` and `db_worker` |
| Metrics (Prometheus) | `utils/logging_utils.py` exposes a counter – wire to `/metrics` |
| GPU scaling | Add more `gpu_worker` instances with different `CONSUMER_NAME` |

---

## License

MIT – ORIX Backend Team
