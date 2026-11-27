"""The Rellachat business tools of a voice agent (products, calendar, orders, leads).

The API sends their definitions in the agent config (OpenAI function format, the
same the chat agents use) plus ``tools_path``. Each one becomes an LLM function;
calling it POSTs ``{arguments}`` to ``{RELLACHAT_API_URL}/voice-service/{tools_path}/{name}``
and the API's result (``{ok: true, ...}`` or ``{ok: false, error}``) goes back to the model.
"""

import re
from typing import Any

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema

# The voice service's own functions: an API tool never replaces them
RESERVED_NAMES = frozenset({"end_call", "transfer_to_advisor"})

_NAME = re.compile(r"^[a-z_]{1,64}$")
_PATH = re.compile(r"^[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*$")

UNAVAILABLE_ERROR = (
    "No pude consultar el sistema en este momento. Dile a la persona que no pudiste "
    "revisarlo ahora y ofrécele intentarlo de nuevo o transferirla a un asesor."
)


def function_schemas(definitions: list[dict[str, Any]] | None) -> list[FunctionSchema]:
    """FunctionSchemas for the valid, distinct, non-reserved tool definitions, in order."""
    schemas: list[FunctionSchema] = []
    seen: set[str] = set()
    for definition in definitions or []:
        function = definition.get("function") if isinstance(definition, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not _NAME.match(name) or name in RESERVED_NAMES or name in seen:
            continue
        parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
        properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
        required = [key for key in parameters.get("required") or [] if key in properties]
        schemas.append(
            FunctionSchema(
                name=name,
                description=str(function.get("description") or ""),
                properties=properties,
                required=required,
            )
        )
        seen.add(name)
    return schemas


def valid_tools_path(path: str | None) -> bool:
    """Whether the API's tools_path is a plain relative path (no traversal, no scheme)."""
    return bool(path) and bool(_PATH.match(path)) and ".." not in path


def error_result(message: str = UNAVAILABLE_ERROR) -> dict[str, Any]:
    return {"ok": False, "error": message}
