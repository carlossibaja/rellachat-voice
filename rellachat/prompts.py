"""Prompt builders for the voice agent (system prompt, greeting, summaries)."""

import re

from rellachat.models import AgentConfig

PURPOSE_LABELS = {
    "customer_support": "atención al cliente: resolver dudas y problemas de los clientes",
    "appointments": "agendamiento: ayudar a reservar, cambiar o confirmar citas",
    "lead_qualification": "calificación de prospectos: entender la necesidad del cliente y recopilar sus datos",
    "sales": "ventas: presentar productos o servicios y ayudar al cliente a decidir su compra",
    "other": "asistencia general a quien llama",
}

LANGUAGE_NAMES = {
    "es": "español",
    "en": "inglés",
    "pt": "portugués",
    "fr": "francés",
    "it": "italiano",
    "de": "alemán",
}

VOICE_STYLE_RULES = """Reglas de estilo para voz (tus respuestas se convierten en audio):
- Usa frases cortas y naturales, como en una llamada telefónica real.
- Haz una sola pregunta a la vez y espera la respuesta.
- Lee los números, precios, fechas y horas de forma natural, como se dicen en voz alta.
- No uses markdown, viñetas, listas, emojis ni símbolos especiales.
- No inventes información. Si no sabes algo, dilo con honestidad.
- Sé cordial, claro y breve. Evita respuestas de más de dos o tres oraciones."""


CALENDAR_TOOLS = frozenset(
    {"consultar_disponibilidad", "agendar_cita", "reprogramar_cita", "cancelar_cita", "listar_citas_contacto"}
)

MESSAGING_TOOLS = frozenset({"enviar_producto", "enviar_precios", "crear_pedido"})


def tool_names(agent: AgentConfig) -> set[str]:
    """Names of the business tools the API sent with the agent."""
    names = set()
    for definition in agent.tools:
        function = definition.get("function") if isinstance(definition, dict) else None
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def business_tools_section(agent: AgentConfig, *, test_mode: bool = False) -> str | None:
    """How to use the business tools in a spoken call, then the API's own rules and data for them."""
    names = tool_names(agent)
    if not names:
        return None

    lines = [
        "Herramientas del negocio (consultan y actúan en el sistema del negocio):",
        "- Tienes todas estas herramientas, pero úsalas solo cuando sirvan a tu propósito y a las "
        "instrucciones del negocio, o cuando la persona lo pida. Por ejemplo, un agente de atención "
        "al cliente no ofrece pedidos ni ventas si no se los piden.",
        "- Antes de consultar o hacer algo que la persona pidió, di una frase corta de espera, por "
        "ejemplo \"déjame revisar\" o \"un momento, lo consulto\", y llama la herramienta en el "
        "mismo turno. Las herramientas de leads se usan en silencio, sin anunciarlas.",
        "- Nunca digas que hiciste algo hasta que la herramienta responda ok. Si responde con "
        "un error, corrígelo y vuelve a intentarlo, o explícale a la persona con palabras sencillas.",
        "- No leas ids, códigos ni formatos técnicos en voz alta.",
    ]
    if names & CALENDAR_TOOLS:
        lines.append(
            "- Puedes consultar la disponibilidad"
            + (" y agendar, reprogramar o cancelar citas" if names & {"agendar_cita", "reprogramar_cita", "cancelar_cita"} else "")
            + ". Ofrece pocas opciones de horario a la vez (dos o tres), no la lista completa."
        )
        lines.append(
            "- Antes de agendar, reprogramar o cancelar, repítele a la persona el día de la semana, "
            "la fecha y la hora (por ejemplo \"el jueves 15 de octubre a las nueve y media de la mañana\") "
            "y espera que confirme."
        )
    if names & MESSAGING_TOOLS:
        lines.append(
            "- Los productos, precios y pedidos se envían al chat de WhatsApp de la persona: "
            "díselo, por ejemplo \"te lo envío por WhatsApp\"."
        )
    if test_mode:
        lines.append(
            "- En esta llamada de prueba las acciones (citas, envíos, pedidos) se simulan; "
            "responde como si fueran reales."
        )

    parts = ["\n".join(lines)]
    if agent.tools_prompt and agent.tools_prompt.strip():
        parts.append(agent.tools_prompt.strip())
    return "\n\n".join(parts)


def language_name(code: str | None) -> str:
    """Human name (in Spanish) for a language code like ``es`` or ``es-MX``."""
    base = (code or "es").split("-")[0].lower()
    return LANGUAGE_NAMES.get(base, code or "español")


def build_system_prompt(agent: AgentConfig, *, test_mode: bool = False) -> str:
    """Build the system prompt for a call from the agent configuration."""
    identity = agent.identity_name.strip() or agent.name.strip() or "el asistente virtual"
    company = agent.company_name.strip() or "la empresa"
    purpose = PURPOSE_LABELS.get(agent.purpose, PURPOSE_LABELS["other"])

    parts: list[str] = [
        f"Eres {identity}, el agente de voz de {company}. Atiendes llamadas telefónicas por WhatsApp.",
        f"Tu propósito principal es {purpose}.",
        f"Habla siempre en {language_name(agent.language)}, salvo que quien llama use otro idioma.",
    ]

    company_lines: list[str] = []
    if agent.company_description and agent.company_description.strip():
        company_lines.append(agent.company_description.strip())
    if agent.company_website and agent.company_website.strip():
        company_lines.append(f"Sitio web: {agent.company_website.strip()}")
    if company_lines:
        parts.append("Contexto de la empresa:\n" + "\n".join(company_lines))

    if agent.instructions and agent.instructions.strip():
        parts.append("Instrucciones del negocio:\n" + agent.instructions.strip())

    parts.append(VOICE_STYLE_RULES)

    tool_lines = [
        "Herramientas:",
        "- end_call: úsala cuando la conversación terminó o la persona se despide. "
        "Antes de llamarla, despídete brevemente en el mismo turno.",
    ]
    if agent.transfer_enabled:
        tool_lines.append(
            "- transfer_to_advisor: úsala cuando la persona pida hablar con un humano o "
            "cuando no puedas resolver su solicitud. Llámala sin decir nada antes; "
            "el sistema le pedirá a la persona que espere."
        )
    else:
        tool_lines.append(
            "- No puedes transferir a un asesor humano. Si te lo piden, explica con amabilidad "
            "que en este momento no es posible y ofrece tu ayuda."
        )
    parts.append("\n".join(tool_lines))

    business = business_tools_section(agent, test_mode=test_mode)
    if business:
        parts.append(business)

    if test_mode:
        parts.append(
            "Esta es una llamada de prueba desde el navegador. Si corresponde transferir, "
            "di que en una llamada real transferirías a un asesor en este momento."
        )

    return "\n\n".join(parts)


# {{cliente}} / {{nombre}} in a configured greeting: the caller's first name
_CLIENT_PLACEHOLDER = re.compile(r"\{\{\s*(cliente|nombre)\s*\}\}", re.IGNORECASE)


def _fill_client(greeting: str, contact_name: str | None) -> str:
    first_name = (contact_name or "").strip().split(" ")[0]
    if first_name:
        return _CLIENT_PLACEHOLDER.sub(first_name, greeting)
    # Unknown caller: drop the placeholder and the space before it ("hola {{cliente}}," -> "hola,")
    return re.sub(r"\s*" + _CLIENT_PLACEHOLDER.pattern, "", greeting, flags=re.IGNORECASE)


def build_greeting(agent: AgentConfig, contact_name: str | None = None) -> str:
    """Opening line: the configured greeting (with {{cliente}} filled) or a default from identity/company."""
    if agent.greeting and agent.greeting.strip():
        return _fill_client(agent.greeting.strip(), contact_name)
    identity = agent.identity_name.strip() or agent.name.strip()
    company = agent.company_name.strip()
    if identity and company:
        return f"Hola, te habla {identity} de {company}. ¿En qué te puedo ayudar?"
    if company:
        return f"Hola, gracias por llamar a {company}. ¿En qué te puedo ayudar?"
    if identity:
        return f"Hola, te habla {identity}. ¿En qué te puedo ayudar?"
    return "Hola, gracias por llamar. ¿En qué te puedo ayudar?"


HOLD_MESSAGE = "Claro, dame un momento por favor, te comunico con un asesor. No cuelgues."

NO_ADVISOR_INSTRUCTION = (
    "[Sistema] Ningún asesor contestó la transferencia. Dile a la persona, en una frase "
    "breve, que en este momento no hay asesores disponibles y continúa ayudándole tú. "
    "No vuelvas a transferir a menos que la persona insista."
)

CALL_STARTED_MARKER = "[La llamada se conectó]"


def format_transcript(transcript: list[dict]) -> str:
    """Render transcript entries as plain lines for summarization."""
    labels = {"user": "Cliente", "assistant": "Agente IA", "advisor": "Asesor"}
    lines = []
    for entry in transcript:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{labels.get(entry.get('role'), entry.get('role'))}: {text}")
    return "\n".join(lines)


SUMMARY_SYSTEM_PROMPT = (
    "Resumes llamadas telefónicas de atención al cliente. Escribe en español, en texto plano, "
    "sin markdown, en un máximo de cuatro oraciones: motivo de la llamada, datos importantes "
    "que dio el cliente y cómo terminó o qué queda pendiente."
)

TRANSFER_SUMMARY_SYSTEM_PROMPT = (
    "Preparas a un asesor humano que va a recibir una llamada transferida. Escribe en español, "
    "en texto plano, sin markdown, en un máximo de tres oraciones: quién llama si se sabe, qué "
    "necesita y qué se intentó hasta ahora."
)
