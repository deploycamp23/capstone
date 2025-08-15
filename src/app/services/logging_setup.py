from __future__ import annotations

import logging
import sys
import uuid
from typing import Literal

import structlog
from fastapi import Request


def configure_logging(mode: Literal["human", "json"]) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if mode == "json":
        processors = shared + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared + [structlog.dev.ConsoleRenderer()]
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def request_context_middleware_factory(app):
    async def middleware(request: Request, call_next):
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=str(uuid.uuid4()),
            method=request.method,
            path=request.url.path,
            client=getattr(request.client, "host", None),
        )
        log = structlog.get_logger()
        log.info("request.start")
        try:
            response = await call_next(request)
            log.info("request.end", status_code=response.status_code)
            return response
        except Exception as e:
            log.error("request.exception", exc_info=True)
            raise

    return middleware
