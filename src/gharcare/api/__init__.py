"""Web API. Import AppService for the framework-free logic; gharcare.api.app needs FastAPI (pip install -r requirements-app.txt)."""
from gharcare.api.service import AppService

__all__ = ["AppService"]
