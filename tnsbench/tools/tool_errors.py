"""Standard tool error codes."""
from __future__ import annotations

from ..core.types import ToolError


def not_found(kind: str, key: str) -> ToolError:
    return ToolError(code="NOT_FOUND", message=f"{kind} '{key}' not found.", details={"kind": kind, "key": key})


def invalid_argument(message: str, **details) -> ToolError:
    return ToolError(code="INVALID_ARGUMENT", message=message, details=details)


def conflict(message: str, **details) -> ToolError:
    return ToolError(code="CONFLICT", message=message, details=details)


def unauthorized(message: str = "Not authorized.", **details) -> ToolError:
    return ToolError(code="UNAUTHORIZED", message=message, details=details)
