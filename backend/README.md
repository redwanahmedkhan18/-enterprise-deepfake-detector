# Backend — Enterprise Deepfake Detection Platform

Phase 1 (Backend Foundation) of the roadmap. This is a real, runnable FastAPI
service — not a stub — covering auth, RBAC, job/result persistence, media
upload to object storage, and async hand-off to a Celery worker that calls
the (separately deployed) AI inference service.

## What's implemented

- **Auth**: register / login / refresh, JWT access + refresh tokens, bcrypt password hashing
- **RBAC**: `admin`, `researcher`, `enterprise_client` roles enforced via a `require_roles` dependency
- **Detection jobs**: upload → validated → stored in S3-compatible object storage → queued to Celery → result persisted
- **Layering**: API → Service → Repository → Database, per the architecture doc
- **Middleware**: request ID propagation, Redis-backed rate limiting
- **Error handling**: domain exceptions mapped to proper HTTP status codes
- **Migrations**: Alembic wired to the SQLAlchemy models
- **Docker**: `docker-compose.yml` brings up Postgres, Redis, the API, and a Celery worker

## What's intentionally a seam, not a stub

`app/services/inference_client.py` calls out to `INFERENCE_SERVICE_URL` (default
`http://localhost:8500`) via a documented `POST /detect` contract. That's the
boundary to the **AI Engine** (Phases 3–5 of the roadmap — face detection,
spatial/temporal/frequency/physiological branches, fusion, explainability),
which is deliberately a separate service so it can scale and deploy
independently, per the architecture doc. Building that model pipeline is a
distinct, much larger effort (data, training, evaluation) — say the word and
we start Phase 2/3 next.

## Running locally

```bash
cd docker
cp ../backend/.env.example ../backend/.env   # edit SECRET_KEY at minimum
docker compose up --build
```

The API will be at `http://localhost:8000`, interactive docs at `/docs`.

### First-time DB migration

```bash
docker compose exec backend alembic revision --autogenerate -m "init"
docker compose exec backend alembic upgrade head
```

### Running without Docker

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
# in another terminal:
celery -A app.workers.celery_app worker --loglevel=info
```

## Verified

The app has been smoke-tested in this environment: all routes register
correctly under `/api/v1/...` and the app boots without errors (see the
OpenAPI schema for the full endpoint list).

## Next phases (from your roadmap)

2. Dataset management & preprocessing
3. AI detection pipeline (spatial/temporal/frequency/physiological branches)
4. Model training & evaluation
5. Inference service (the thing `inference_client.py` talks to)
6. Frontend
7. Kubernetes/production deployment
8. Monitoring & scaling

Tell me which one to build next and I'll do the same thing: real code, tested
where I can test it locally.
