"""The optional 'G' in RAG: turn retrieved passages into a written answer.

Dense and hybrid retrieval run offline. The query_rewrite architecture also
calls Gemini, in architectures/rewrite.py. This module handles only the final
answer; `scripts/ask.py --no-answer` prints retrieved passages instead.

The prompt does one job: keep the model inside the retrieved text. That is what
"grounded" means — the answer has to come from the passages, and if the answer
is not there the model must say so rather than fall back on memory.
"""

from __future__ import annotations

import getpass
import os
import subprocess
import time

from .index import Hit

MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.7-flash"
MAX_ATTEMPTS = 3
KEYCHAIN_SERVICE = "hotpotqa-rag-gemini"

SYSTEM_PROMPT = """\
You answer questions using only the numbered passages provided.

- Answer in as few words as the question allows — usually a name, date, or short phrase.
- Cite the passage numbers you used, like [1] or [2][4].
- If the passages do not contain the answer, say exactly: Not in the retrieved passages.
- Never use knowledge from outside the passages."""


class GenerationUnavailable(RuntimeError):
    """Raised when the answer step cannot run — missing package or credentials."""


def _load_api_key() -> str:
    """Load Gemini credentials from the environment or the macOS Keychain."""
    if api_key := os.getenv("GEMINI_API_KEY"):
        return api_key.strip()

    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                getpass.getuser(),
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    else:
        if api_key := result.stdout.strip():
            return api_key

    raise GenerationUnavailable(
        "no Gemini credentials found; set GEMINI_API_KEY or save the key in "
        f"the macOS Keychain service {KEYCHAIN_SERVICE!r}"
    )


def format_context(hits: list[Hit]) -> str:
    """Lay the retrieved passages out as a numbered list for the prompt."""
    return "\n\n".join(
        f"[{i}] {hit.passage.title}\n{hit.passage.text}" for i, hit in enumerate(hits, start=1)
    )


def generate_answer(question: str, hits: list[Hit], *, model: str = MODEL) -> str:
    """Ask Gemini to answer `question` from `hits`."""
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise GenerationUnavailable(
            "The `google-genai` package is not installed. Run: .venv/bin/pip install google-genai"
        ) from exc

    api_key = _load_api_key()
    user_message = f"Passages:\n\n{format_context(hits)}\n\nQuestion: {question}"

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=256,
        # This RAG does not give Gemini any tools. Explicitly disabling AFC
        # avoids the SDK warning about direct automatic function calling.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    response = None
    last_error = None
    models = tuple(dict.fromkeys((model, FALLBACK_MODEL)))
    for candidate_model in models:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=user_message,
                    config=config,
                )
                break
            except errors.APIError as exc:
                last_error = exc
                # A 503 means temporary model saturation. Retry briefly, then
                # move to the fallback model. Other errors need user action.
                if exc.code != 503:
                    raise GenerationUnavailable(f"Gemini API error: {exc}") from exc
                if attempt + 1 < MAX_ATTEMPTS:
                    time.sleep(2**attempt)
        if response is not None:
            break

    if response is None:
        raise GenerationUnavailable(f"Gemini API unavailable: {last_error}")

    try:
        answer = response.text.strip()
    except (AttributeError, ValueError) as exc:
        raise GenerationUnavailable("Gemini returned no text answer") from exc
    if not answer:
        raise GenerationUnavailable("Gemini returned an empty answer")
    return answer
