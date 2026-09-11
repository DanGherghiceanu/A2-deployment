"""Structured logging.

Cloud Run reads stdout. If a line is plain text it lands in the log viewer as an
undifferentiated INFO string; if it's JSON with a `severity` key, the console
colours it by level, makes the extra fields filterable, and groups the request
trace. That difference is what makes the Step 5 log screenshots readable, so the
formatter emits JSON in the shape Cloud Logging expects.

Set LOG_FORMAT=text for human-friendly output while developing locally.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

from . import config

# Cloud Logging severity names differ slightly from Python's level names.
_SEVERITY = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


class CloudLoggingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": _SEVERITY.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Anything passed as logger.info(msg, extra={"json_fields": {...}})
        # becomes a queryable field in Cloud Logging.
        extra_fields = getattr(record, "json_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    use_text = os.environ.get("LOG_FORMAT", "json").lower() == "text"

    handler = logging.StreamHandler(sys.stdout)
    if use_text:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    else:
        handler.setFormatter(CloudLoggingFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(config.LOG_LEVEL)

    # uvicorn installs its own handlers; route them through ours instead so
    # every line in the container has the same shape.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # Pillow and urllib3 are chatty at DEBUG.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
