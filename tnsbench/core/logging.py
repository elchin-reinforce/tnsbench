"""Simple logging helper for TnSBench."""
from __future__ import annotations

import logging
import os


def get_logger(name: str = "tnsbench") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = os.environ.get("TNSBENCH_LOG_LEVEL", "INFO").upper()
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
