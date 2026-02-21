"""
voice_note_service_cli_responses.py

Production-hardened service layer for extracting a VoiceNote schema using:
- LangChain's ChatOpenAI configured to use the OpenAI *Responses API* (explicit)
- Structured outputs via JSON Schema (method="json_schema", strict=True)
- Pydantic v2 for validation
- Optional FastAPI-ready endpoint example
- A CLI-friendly main() with mock inputs so you can run from the command line

Run from CLI:
  export OPENAI_API_KEY="..."
  python voice_note_service_cli_responses.py

Default model: gpt-5-mini
Override model:
  export OPENAI_MODEL="gpt-4.1-mini"  # or another model you have access to
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union, Literal, TypedDict

from pydantic import BaseModel, Field, ConfigDict, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"^Pydantic serializer warnings:",
    category=UserWarning,
    module=r"pydantic\..*",
)g

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOGGER_NAME = "voice_note_service"
logger = logging.getLogger(LOGGER_NAME)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for CLI usage."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# ---------------------------------------------------------------------
# Output schema (STRICT)
# ---------------------------------------------------------------------
class VoiceNote(BaseModel):
    # Enforce "additionalProperties: false" behavior at validation time.
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="A title for the voice note")
    summary: str = Field(description="A short one sentence summary of the voice note.")
    actionItems: list[str] = Field(description="A list of action items from the voice note")


# ---------------------------------------------------------------------
# Service-layer response models
# ---------------------------------------------------------------------
class ServiceError(BaseModel):
    type: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ServiceResult(BaseModel):
    success: Literal[True, False]
    data: Optional[Dict[str, Any]] = None  # VoiceNote JSON dict on success
    error: Optional[ServiceError] = None


# ---------------------------------------------------------------------
# Message input types
# ---------------------------------------------------------------------
class DictMessage(TypedDict):
    role: str
    content: str


MessageInput = Union[
    BaseMessage,               # single LC message
    Sequence[BaseMessage],      # list/sequence of LC messages
    DictMessage,               # single dict message
    Sequence[DictMessage],     # list/sequence of dict messages
]


# ---------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------
def _is_dict_message(obj: Any) -> bool:
    return isinstance(obj, dict) and "role" in obj and "content" in obj


def _dict_to_basemessage(m: DictMessage) -> BaseMessage:
    role = m["role"]
    content = m["content"]

    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)

    raise ValueError(f"Unsupported role: {role!r}. Expected 'system'|'user'|'assistant'.")


def normalize_messages(messages: MessageInput) -> List[BaseMessage]:
    """Normalize inputs to a list[BaseMessage]."""
    if isinstance(messages, BaseMessage):
        return [messages]

    if _is_dict_message(messages):
        return [_dict_to_basemessage(messages)]  # type: ignore[arg-type]

    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        if len(messages) == 0:
            raise ValueError("messages must not be empty.")

        first = messages[0]  # type: ignore[index]
        if isinstance(first, BaseMessage):
            return list(messages)  # type: ignore[arg-type]

        if _is_dict_message(first):
            return [_dict_to_basemessage(m) for m in messages]  # type: ignore[arg-type]

    raise TypeError(
        "Unsupported messages type. Provide a BaseMessage, a list of BaseMessage, "
        "a dict {'role','content'}, or a list of such dicts."
    )


# ---------------------------------------------------------------------
# LLM factory + structured wrapper (Responses API, explicit)
# ---------------------------------------------------------------------
def build_structured_llm(*, model: str = "gpt-5-mini", temperature: float = 0.0) -> Any:
    """
    Best practice: explicitly route through the OpenAI Responses API so you don't
    accidentally hit /v1/chat/completions.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        use_responses_api=True,   # <-- explicit routing
    )

    return llm.with_structured_output(
        VoiceNote,
        method="json_schema",
        strict=True,
    )


# ---------------------------------------------------------------------
# Retry + logging helpers
# ---------------------------------------------------------------------
def _summarize_validation_error(e: ValidationError, max_items: int = 3) -> Dict[str, Any]:
    errs = e.errors()
    sample = [
        {"loc": item.get("loc"), "msg": item.get("msg"), "type": item.get("type")}
        for item in errs[:max_items]
    ]
    return {"error_count": len(errs), "sample": sample}


def _repair_system_message() -> SystemMessage:
    return SystemMessage(
        content=(
            "Return ONLY a JSON object that exactly matches the VoiceNote schema.\n"
            "Keys must be exactly: title, summary, actionItems.\n"
            "Do not add extra keys. Do not include markdown, explanations, or code fences."
        )
    )


# ---------------------------------------------------------------------
# Service function
# ---------------------------------------------------------------------
def extract_voice_note_service(
    messages: MessageInput,
    *,
    model: str = "gpt-5-mini",
    max_retries: int = 2,
    add_repair_instruction_on_retry: bool = True,
) -> ServiceResult:
    """
    - Normalizes messages
    - Invokes structured LLM with retries on ValidationError
    - Logs validation failures
    - Returns ServiceResult error for validation failures
    - Unexpected exceptions are not swallowed
    """
    lc_messages = normalize_messages(messages)
    structured_llm = build_structured_llm(model=model, temperature=0.0)

    last_validation_error: Optional[ValidationError] = None

    for attempt in range(max_retries + 1):
        try:
            run_messages = lc_messages
            if attempt > 0 and add_repair_instruction_on_retry:
                run_messages = [_repair_system_message()] + lc_messages

            voice_note: VoiceNote = structured_llm.invoke(run_messages)
            return ServiceResult(success=True, data=voice_note.model_dump())

        except ValidationError as e:
            last_validation_error = e
            logger.warning(
                "VoiceNote validation failed (attempt %s/%s): %s",
                attempt + 1,
                max_retries + 1,
                _summarize_validation_error(e),
                exc_info=True,
            )

    if last_validation_error is not None:
        return ServiceResult(
            success=False,
            error=ServiceError(
                type="ValidationError",
                message="LLM output did not match the VoiceNote schema.",
                details={"validation": _summarize_validation_error(last_validation_error, max_items=10)},
            ),
        )

    return ServiceResult(success=False, error=ServiceError(type="UnknownError", message="Unknown failure."))


# ---------------------------------------------------------------------
# CLI-friendly main() with mock inputs
# ---------------------------------------------------------------------
def main() -> None:
    configure_logging(logging.INFO)

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set. Export it before running this script.")
        logger.error("Example (bash): export OPENAI_API_KEY='...your key...'")
        return

    model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

    mock_messages: List[DictMessage] = [
        {"role": "system", "content": "Extract a voice note into the VoiceNote schema."},
        {
            "role": "user",
            "content": (
                "Met with Sam about the v2 launch. We need to ship by Friday. "
                "I will email design today. Sam will check analytics and report back tomorrow."
            ),
        },
    ]

    result = extract_voice_note_service(
        mock_messages,
        model=model,
        max_retries=2,
        add_repair_instruction_on_retry=True,
    )

    print(json.dumps(result.model_dump(), indent=2))


# ---------------------------------------------------------------------
# Optional FastAPI drop-in example
# ---------------------------------------------------------------------
try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
except Exception:
    FastAPI = None  # type: ignore
    JSONResponse = None  # type: ignore

if FastAPI is not None:
    app = FastAPI()

    class ExtractRequest(BaseModel):
        messages: Union[DictMessage, List[DictMessage]]

    @app.post("/voice-note", response_model=ServiceResult)
    def create_voice_note(req: ExtractRequest):
        result = extract_voice_note_service(
            req.messages,
            model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
            max_retries=2,
        )
        if result.success:
            return result
        return JSONResponse(status_code=422, content=result.model_dump())


if __name__ == "__main__":
    main()
