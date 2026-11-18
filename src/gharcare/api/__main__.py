"""Run the app:  python -m gharcare.api      (settings come from .env; see .env.example)"""
from __future__ import annotations

import os

from gharcare.api.service import AppService
from gharcare.config import Settings
from gharcare.domain.validation import validate_identifier
from gharcare.env import load_env_file
from gharcare.errors import GharCareError
from gharcare.logging_utils import configure_logging


def main() -> None:
    load_env_file()
    configure_logging()
    try:
        import uvicorn
        from gharcare.api.app import create_app
    except ImportError as exc:
        raise SystemExit(f"The web app needs extra packages: pip install -r requirements-app.txt  ({exc})")
    settings = Settings.from_env()
    user_id = validate_identifier(os.environ.get("GHARCARE_USER_ID", "demo"), "GHARCARE_USER_ID")
    token = os.environ.get("GHARCARE_APP_TOKEN") or None
    host, port = os.environ.get("GHARCARE_HOST", "127.0.0.1"), int(os.environ.get("GHARCARE_PORT", "8000"))
    if host not in ("127.0.0.1", "localhost") and not token:
        raise SystemExit("Refusing to listen on a non-local address without GHARCARE_APP_TOKEN: your household documents would be open to the network.")
    try:
        service = AppService.create(settings)
    except GharCareError as exc:        # e.g. Chroma selected but not installed: one clear line, not a traceback
        raise SystemExit(f"GharCare could not start: {exc}")
    print(f"GharCare AI: http://{host}:{port}   (user '{user_id}', AI: {'configured' if service.llm_configured else 'NOT configured: limited mode'}, "
          f"workflow: {service.assistant.backend}, access token: {'required' if token else 'off (local only)'})")
    uvicorn.run(create_app(service, user_id=user_id, token=token, max_upload_bytes=settings.max_upload_bytes), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
