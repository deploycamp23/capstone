from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from watchfiles import awatch

from app.routers import api, web
from app.services.model_service import ModelService

static_dir = Path("src/app/static")
templates_dir = Path("src/app/templates")
config_path = Path("config.yaml")

_container: dict[str, object] = {"model_service": ModelService(config_path)}

def get_container() -> dict[str, object]:
    return _container


async def _watch_config():
    await asyncio.sleep(0.1)
    async for _ in awatch(config_path):
        try:
            svc: ModelService = _container["model_service"]  # type: ignore
            svc.load_from_config()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc: ModelService = _container["model_service"]  # type: ignore
    svc.load_from_config()
    task = asyncio.create_task(_watch_config())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(Exception):
            await task


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.include_router(web.router)
app.include_router(api.router)
