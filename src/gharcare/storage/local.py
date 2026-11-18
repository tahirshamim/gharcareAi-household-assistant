"""Local-filesystem ArtifactStore."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from gharcare.domain.validation import validate_identifier
from gharcare.errors import DomainValidationError, NotFoundError, StorageError
from gharcare.storage.base import ArtifactRef, ArtifactStore

_EXTENSION = re.compile(r"^\.[a-z0-9]{1,10}$")
_REPORT_EXTENSIONS = {".md", ".txt", ".html", ".json", ".pdf"}
_VERSION_FILE = re.compile(r"^v(\d{4,})\.json$")


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: str | Path, max_upload_bytes: int = 20 * 1024 * 1024) -> None:
        self._root = Path(root).resolve()
        self._max_upload_bytes = max_upload_bytes
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"cannot create artifact root {self._root}: {exc}") from exc

    # ---- originals -----------------------------------------------------------------------
    def save_original(self, user_id: str, filename: str, content: bytes) -> ArtifactRef:
        validate_identifier(user_id, "user_id")
        if not isinstance(content, (bytes, bytearray)) or len(content) == 0:
            raise DomainValidationError("uploaded file is empty")
        if len(content) > self._max_upload_bytes:
            raise DomainValidationError(
                f"file is too large ({len(content)} bytes; limit {self._max_upload_bytes})"
            )
        digest = hashlib.sha256(content).hexdigest()
        # The user's filename is never used on disk: only a whitelisted extension survives.
        suffix = Path(filename or "").suffix.lower()
        suffix = suffix if _EXTENSION.match(suffix) else ""
        relative = f"{user_id}/originals/{digest[:2]}/{digest}{suffix}"
        target = self._root / relative
        if target.exists():
            return ArtifactRef(relative, digest, len(content), created=False)
        self._atomic_write(target, bytes(content), read_only=True)
        return ArtifactRef(relative, digest, len(content), created=True)

    # ---- reading -------------------------------------------------------------------------
    def _resolve(self, user_id: str, relative_path: str) -> Path:
        validate_identifier(user_id, "user_id")
        user_root = (self._root / user_id).resolve()
        candidate = (self._root / str(relative_path)).resolve()
        if not candidate.is_relative_to(user_root):
            raise StorageError("path is outside the user's storage area")
        return candidate

    def read(self, user_id: str, relative_path: str) -> bytes:
        path = self._resolve(user_id, relative_path)
        if not path.is_file():
            raise NotFoundError(f"artifact not found: {relative_path}")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise StorageError(f"could not read artifact: {exc}") from exc

    def exists(self, user_id: str, relative_path: str) -> bool:
        try:
            return self._resolve(user_id, relative_path).is_file()
        except StorageError:
            return False

    # ---- extraction results (versioned) --------------------------------------------------
    def _extracted_dir(self, user_id: str, document_id: str) -> Path:
        validate_identifier(user_id, "user_id")
        validate_identifier(document_id, "document_id")
        return self._root / user_id / "extracted" / document_id

    def _versions(self, directory: Path) -> list[int]:
        if not directory.is_dir():
            return []
        return sorted(int(m.group(1)) for p in directory.iterdir() if (m := _VERSION_FILE.match(p.name)))

    def save_extracted(self, user_id: str, document_id: str, data: Mapping[str, Any]) -> ArtifactRef:
        directory = self._extracted_dir(user_id, document_id)
        payload = json.dumps(dict(data), ensure_ascii=False, indent=2, default=str).encode("utf-8")
        for _ in range(5):  # retry if a concurrent writer takes the same version number
            versions = self._versions(directory)
            number = (versions[-1] + 1) if versions else 1
            target = directory / f"v{number:04d}.json"
            try:
                self._exclusive_write(target, payload)
            except FileExistsError:
                continue
            relative = target.relative_to(self._root).as_posix()
            return ArtifactRef(relative, hashlib.sha256(payload).hexdigest(), len(payload), True)
        raise StorageError("could not allocate a new extraction version")

    def load_latest_extracted(self, user_id: str, document_id: str) -> dict[str, Any] | None:
        directory = self._extracted_dir(user_id, document_id)
        versions = self._versions(directory)
        if not versions:
            return None
        try:
            return json.loads((directory / f"v{versions[-1]:04d}.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError(f"extraction file is unreadable: {exc}") from exc

    # ---- reports -------------------------------------------------------------------------
    def save_report(self, user_id: str, name: str, content: bytes, extension: str = ".md") -> ArtifactRef:
        validate_identifier(user_id, "user_id")
        extension = extension.lower()
        if extension not in _REPORT_EXTENSIONS:
            raise DomainValidationError(f"report extension must be one of {sorted(_REPORT_EXTENSIONS)}")
        slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:60] or "report"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        directory = self._root / user_id / "reports"
        for attempt in range(1, 100):
            suffix = "" if attempt == 1 else f"-{attempt}"
            target = directory / f"{stamp}_{slug}{suffix}{extension}"
            try:
                self._exclusive_write(target, bytes(content))
            except FileExistsError:
                continue
            return ArtifactRef(target.relative_to(self._root).as_posix(),
                               hashlib.sha256(content).hexdigest(), len(content), True)
        raise StorageError("could not allocate a unique report filename")

    # ---- low-level writes ----------------------------------------------------------------
    @staticmethod
    def _exclusive_write(target: Path, payload: bytes) -> None:
        """Create-only write: raises FileExistsError instead of ever overwriting."""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "xb") as handle:
                handle.write(payload)
        except FileExistsError:
            raise
        except OSError as exc:
            raise StorageError(f"could not write {target.name}: {exc}") from exc

    @staticmethod
    def _atomic_write(target: Path, payload: bytes, *, read_only: bool) -> None:
        """Write to a temp file then rename, so a crash never leaves a half-written original."""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                if read_only:
                    os.chmod(tmp_name, 0o444)
                os.replace(tmp_name, target)
            except BaseException:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
                raise
        except OSError as exc:
            raise StorageError(f"could not store file: {exc}") from exc
