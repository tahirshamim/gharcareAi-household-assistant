"""Configuration check: `python -m gharcare.doctor`. Reports what is set up and what is missing.
It never prints secret values (only whether a key is present)."""
from __future__ import annotations

import importlib.util
import os
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from gharcare.config import Settings
from gharcare.env import load_env_file
from gharcare.errors import GharCareError, LLMError
from gharcare.llm import LLMMessage, LLMProvider, build_llm
from gharcare.logging_utils import redact, secrets_from_env


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def code_location_check(base: Path, loaded_from: Path) -> Check:
    """Warn when `import gharcare` loads a different folder than the project in the current directory: an
    editable install keeps pointing at whichever unzipped folder it was made in."""
    local = (base / "src" / "gharcare").resolve()
    if not local.is_dir() or local == loaded_from.resolve():
        return Check("gharcare code", True, str(loaded_from), required=False)
    return Check("gharcare code", False, f"loading {loaded_from}, NOT this folder's {local}. Run: pip install -e .",
                 required=False)


def diagnose(env: Mapping[str, str], base_dir: Path | None = None,
             find_spec: Callable[[str], object] = importlib.util.find_spec) -> list[Check]:
    base = Path(base_dir or ".")
    settings = Settings.from_env(env)
    checks: list[Check] = []

    if (base / ".env").exists():
        checks.append(Check(".env file", True, "found"))
    elif (base / ".env.txt").exists():
        checks.append(Check(".env file", False, "found '.env.txt' instead: rename it to '.env' (Notepad adds .txt). "
                                               "In PowerShell: Rename-Item .env.txt .env", required=False))
    else:
        checks.append(Check(".env file", False, "not found (copy .env.example to .env), using system environment only", required=False))

    checks.append(code_location_check(base, Path(__file__).resolve().parent))

    provider = settings.llm_provider
    checks.append(Check("LLM provider", provider in ("gemini", "ollama"), provider))
    if provider == "gemini":
        has_key = bool(env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"))
        checks.append(Check("GEMINI_API_KEY", has_key, "set" if has_key else "missing: paste your key into .env"))
        checks.append(Check("langchain-google-genai", bool(find_spec("langchain_google_genai")), "installed" if find_spec("langchain_google_genai") else "missing: pip install langchain-google-genai"))
    elif provider == "ollama":
        checks.append(Check("langchain-ollama", bool(find_spec("langchain_ollama")), "installed" if find_spec("langchain_ollama") else "missing: pip install langchain-ollama"))
    checks.append(Check("GHARCARE_LLM_MODEL", bool(settings.llm_model),
                        settings.llm_model or "missing: set a model name (for Gemini, a current Flash model from AI Studio)"))

    needs = [("fastapi", "fastapi", "needed for the web app: pip install -r requirements-app.txt", False),
             ("uvicorn", "uvicorn", "needed for the web app: pip install -r requirements-app.txt", False),
             ("multipart", "python-multipart", "needed for uploads in the web app: pip install python-multipart", False),
             ("langgraph", "langgraph", "recommended: pip install langgraph (a built-in runner is used otherwise)", False),
             ("pypdf", "pypdf", "needed for PDF uploads: pip install pypdf", False),
             ("sentence_transformers", "sentence-transformers", "pip install sentence-transformers", settings.embedder == "sentence-transformers"),
             ("chromadb", "chromadb", "pip install chromadb", settings.vector_store == "chroma")]
    for module, label, hint, required in needs:
        ok = bool(find_spec(module))
        checks.append(Check(label, ok, "installed" if ok else f"missing: {hint}", required))
    checks.append(Check("embedder / vector store", True, f"{settings.embedder} / {settings.vector_store}", False))
    return checks


def live_check(llm: LLMProvider, secrets: tuple[str, ...] = ()) -> Check:
    """Makes ONE tiny real call so a wrong key / model name / quota problem shows its actual error."""
    try:
        reply = llm.generate([LLMMessage("user", "Reply with the single word OK.")])
    except LLMError as exc:
        return Check("live LLM call", False, redact(f"{type(exc).__name__}: {exc}", secrets)[:400])
    return Check("live LLM call", True, f"the model replied: {reply.text[:40]!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gharcare.doctor")
    parser.add_argument("--live", action="store_true", help="also make one real LLM call (uses a little of your quota)")
    args = parser.parse_args(argv)
    load_env_file()
    checks = diagnose(os.environ)
    failed = [c for c in checks if c.required and not c.ok]
    if args.live:
        secrets = tuple(secrets_from_env(os.environ))
        if failed:
            checks.append(Check("live LLM call", False, "skipped: fix the FAIL items above first", required=False))
        else:
            try:
                checks.append(live_check(build_llm(Settings.from_env(os.environ)), secrets))
            except GharCareError as exc:
                checks.append(Check("live LLM call", False, redact(str(exc), secrets)[:400]))
    for c in checks:
        print(f"[{'OK  ' if c.ok else ('FAIL' if c.required else 'warn')}] {c.name}: {c.detail}")
    failed = [c for c in checks if c.required and not c.ok]
    print("\nAll required settings are in place." if not failed else f"\n{len(failed)} required item(s) need attention (see FAIL above).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
