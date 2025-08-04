# Capstone FastAPI + HTMX + Tailwind v4 + Alpine

## Overview
FastAPI web app with a text-based form (HTMX + Alpine + Jinja) posting to a JSON API that serves a PyTorch model. The model auto-reloads when `config.yaml` changes using `watchfiles`, with an RW lock for safe hot-swaps.

## Stack
- FastAPI, Jinja2 templates, HTMX, Alpine.js
- Tailwind CSS v4 via `@tailwindcss/cli` (built asset served from static)
- PyTorch (mock fallback if torch unavailable)
- watchfiles for config-driven reloads

## App Structure
- Web UI: GET `/` renders form
- API: `POST /api/predict`, `GET /api/health`, `GET /api/model`
- Config: `config.yaml` with `model.path: model/path_to_model`

## Setup
1. Install Python deps: `uv sync`
2. Build CSS (one-time or watch):
   - `pnpm run tailwind:build`
   - `pnpm run tailwind:watch`
3. Run app: `uv run python src/app/main.py`

## Tailwind v4
- Source: `src/app/styles/input.css`
- Use v4 import: `@import "tailwindcss";`
- Output: `src/app/static/css/app.css` (served at `/static/css/app.css`)

## Model Reloading
- On startup, model loads from `config.yaml`.
- Background watcher monitors config; on change, loads new model and atomically swaps under a write lock.

## Endpoints
- `POST /api/infer` body: `{ "payload": { "feature_a": "...", ... } }`
  - Response: `{ "prediction": "..." }`
- `GET /api/health`
- `GET /api/model` returns `model_path`, `loaded_at`.

## Notes
- No auth/CSRF.
- Text-only structured fields.
