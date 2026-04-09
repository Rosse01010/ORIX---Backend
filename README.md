# ORIX

Real-time facial recognition platform with consented enrollment and social-profile matching.

## Architecture

```
ORIX/
├── backend/              # FastAPI + Socket.IO API
│   ├── app/              # Application package
│   │   ├── routes/       # REST endpoints
│   │   ├── services/     # Business logic
│   │   ├── utils/        # Face quality, vector search, GPU
│   │   └── websocket/    # WS + Socket.IO managers
│   ├── workers/          # GPU & DB background workers
│   ├── tests/            # pytest suite
│   ├── Dockerfile        # API image
│   └── Dockerfile.worker # GPU worker image
├── frontend/             # React 19 + Vite + TypeScript
│   ├── src/
│   │   ├── api/          # REST clients
│   │   ├── components/   # UI components
│   │   ├── hooks/        # Custom React hooks
│   │   ├── pages/        # Page components
│   │   ├── services/     # Socket, auth, alert services
│   │   ├── store/        # Zustand stores
│   │   └── types/        # TypeScript interfaces
│   ├── public/models/    # face-api.js weights
│   └── Dockerfile        # Production Nginx image
├── docker-compose.yml    # Full-stack orchestration
└── .github/              # CI workflows, PR templates
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Socket.IO, Uvicorn |
| Database | PostgreSQL 16 + pgvector |
| Cache | Redis 7 (Streams + Pub/Sub) |
| AI | InsightFace (ArcFace), face-api.js (browser) |
| Frontend | React 19, Vite, TypeScript, Tailwind CSS, Zustand |
| Infra | Docker Compose, Nginx |

## Quick Start

```bash
# 1. Configure environment
cp backend/.env.example backend/.env

# 2. Start everything
docker compose up -d

# 3. Verify
curl http://localhost:8000/health        # Backend API
open http://localhost:3000               # Frontend UI
```

### Development (without Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:socket_app --host 0.0.0.0 --port 8000 --reload

# Frontend
cd frontend
npm install
npm run dev
```

## Key Features

- Real-time face detection via webcam (browser-side face-api.js)
- Consented enrollment with social profile links (LinkedIn, Instagram, X)
- Multi-angle face recognition (InsightFace ArcFace backend)
- Live alerts via WebSocket / Socket.IO
- Role-based access control (admin, operator, user)
- Similarity candidate panel for unknown faces

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/recognition/persons/enroll` | Enroll person (browser embedding) |
| GET | `/api/recognition/persons/{id}` | Person details + social links |
| GET | `/api/recognition/persons` | List all persons |
| POST | `/api/recognition/recognize` | Recognize faces in image |
| POST | `/auth/login` | JWT authentication |
| GET | `/cameras` | List cameras |
