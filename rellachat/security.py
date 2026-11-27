"""Shared-secret check between the Rellachat API and this service."""

import hmac

SECRET_HEADER = "X-Voice-Secret"


def secret_matches(provided: str | None, expected: str | None) -> bool:
    """Constant-time comparison; an unset expected secret never matches."""
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
