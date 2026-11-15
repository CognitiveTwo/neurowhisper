"""macOS standard directory layout and cloud-sync storage resolution.

Separates read-only bundled assets (``neurowhisper.app/Contents/Frameworks``
when frozen via PyInstaller, or ``macos/resources`` during development) from
writable user directories under ``~/Library/``:

- ``~/Library/Application Support/neurowhisper`` (configs, stats, transcripts,
  and ``storage.json`` bootstrap; overridable by ``sync_dir`` for Dropbox/iCloud)
- ``~/Library/Caches/neurowhisper/models`` (downloaded local STT weights)
- ``~/Library/Logs/neurowhisper`` (``app.log``, ``crash.log``, ``hotkey.log``)

Environment overrides (for isolated testing and power users):
- ``NEUROWHISPER_HOME``
- ``NEUROWHISPER_CACHE_DIR``
- ``NEUROWHISPER_LOG_DIR``
- ``NEUROWHISPER_SOCKET_PATH``
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Optional, Union
import uuid

from . import APP_NAME

STORAGE_BOOTSTRAP_FILENAME = "storage.json"
CONFIG_FILENAME = "whisper_config.json"
STATS_FILENAME = "whisper_stats.json"
AI_INSIGHTS_FILENAME = "ai_insights.json"
TRANSCRIPTIONS_DIRNAME = "transcriptions"
SOCKET_FILENAME = "control.sock"

# macOS AF_UNIX sockaddr_un.sun_path is 104 bytes (including NUL terminator).
# Anything longer than 96 bytes falls back to /tmp/neurowhisper-<uid>/control.sock.
MAX_SAFE_UNIX_SOCKET_LEN = 96


def get_hostname() -> str:
    """Normalized machine identifier compatible with Windows/Linux peer sync."""
    return platform.node().replace(".", "_") or "machine"


def write_json_atomic(path: Union[str, os.PathLike], data: Any, indent: Optional[int] = 2) -> None:
    """Atomically write JSON to ``path`` using a PID+UUID sibling temp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent)
            fh.flush()
            os.fsync(fh.fileno())
        last_err: Optional[BaseException] = None
        for attempt in range(3):
            try:
                os.replace(tmp_path, target)
                return
            except PermissionError as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(0.15)
        if last_err is not None:
            raise last_err
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def get_resource_dir() -> Path:
    """Return the read-only resource directory containing bundled icons/plists."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2] / "resources"


def find_resource_file(filename: str) -> Path:
    """Locate a resource file inside ``get_resource_dir()`` (flat or ``resources/`` subdir)."""
    base = get_resource_dir()
    direct = base / filename
    if direct.exists():
        return direct
    nested = base / "resources" / filename
    if nested.exists():
        return nested
    return direct


def get_app_support_dir() -> Path:
    """Base writable application support directory (always machine-local)."""
    env_home = os.environ.get("NEUROWHISPER_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / "Library" / "Application Support" / APP_NAME


def get_cache_dir() -> Path:
    """Local cache directory (never synced to Dropbox/iCloud)."""
    env_cache = os.environ.get("NEUROWHISPER_CACHE_DIR", "").strip()
    if env_cache:
        return Path(env_cache).expanduser()
    env_home = os.environ.get("NEUROWHISPER_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser() / "cache"
    return Path.home() / "Library" / "Caches" / APP_NAME


def get_models_dir() -> Path:
    """Directory holding downloaded speech-to-text weights."""
    return get_cache_dir() / "models"


def get_logs_dir() -> Path:
    """Local rotating log directory."""
    env_logs = os.environ.get("NEUROWHISPER_LOG_DIR", "").strip()
    if env_logs:
        return Path(env_logs).expanduser()
    env_home = os.environ.get("NEUROWHISPER_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser() / "logs"
    return Path.home() / "Library" / "Logs" / APP_NAME


def _fallback_tmp_socket_path() -> Path:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    sock_dir = Path(f"/tmp/{APP_NAME}-{uid}")
    sock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(sock_dir, 0o700)
    except OSError:
        pass
    return sock_dir / SOCKET_FILENAME


def get_socket_path() -> Path:
    """Return the AF_UNIX control socket path, guarding macOS's 104-byte limit."""
    env_sock = os.environ.get("NEUROWHISPER_SOCKET_PATH", "").strip()
    if env_sock:
        custom = Path(env_sock).expanduser()
        if len(os.fsencode(str(custom))) <= MAX_SAFE_UNIX_SOCKET_LEN:
            return custom
        return _fallback_tmp_socket_path()

    default_sock = get_app_support_dir() / SOCKET_FILENAME
    if len(os.fsencode(str(default_sock))) > MAX_SAFE_UNIX_SOCKET_LEN:
        return _fallback_tmp_socket_path()
    return default_sock


def get_storage_bootstrap_path() -> Path:
    """Path to ``storage.json`` inside ``app_support_dir``."""
    return get_app_support_dir() / STORAGE_BOOTSTRAP_FILENAME


def ensure_storage_bootstrap() -> Path:
    """Ensure ``storage.json`` exists in ``app_support_dir`` with ``{"sync_dir": ""}``."""
    bootstrap = get_storage_bootstrap_path()
    if not bootstrap.exists():
        try:
            write_json_atomic(bootstrap, {"sync_dir": ""}, indent=2)
        except OSError:
            pass
    return bootstrap


def get_sync_dir() -> Optional[Path]:
    """Return the configured custom sync folder as an expanded ``Path``, or ``None``."""
    bootstrap = ensure_storage_bootstrap()
    try:
        with open(bootstrap, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            raw = str(payload.get("sync_dir", "") or "").strip()
            if raw:
                return Path(raw).expanduser()
    except Exception:
        pass
    return None


def set_sync_dir(path: Union[str, os.PathLike, None]) -> Optional[Path]:
    """Persist ``sync_dir`` in ``storage.json`` atomically. Pass ``""`` or ``None`` to reset."""
    raw = str(path).strip() if path is not None else ""
    normalized = str(Path(raw).expanduser()) if raw else ""
    bootstrap = get_storage_bootstrap_path()
    write_json_atomic(bootstrap, {"sync_dir": normalized}, indent=2)
    return Path(normalized) if normalized else None


def get_data_dir() -> Path:
    """Active user data directory: ``sync_dir`` when configured, else ``app_support_dir``."""
    sync = get_sync_dir()
    if sync is not None:
        return sync
    return get_app_support_dir()


def get_transcriptions_dir() -> Path:
    """Active transcript archive directory (``<data_dir>/transcriptions``)."""
    return get_data_dir() / TRANSCRIPTIONS_DIRNAME


def get_config_file() -> Path:
    """Shared config file (``<data_dir>/whisper_config.json``)."""
    return get_data_dir() / CONFIG_FILENAME


def get_local_config_file(hostname: Optional[str] = None) -> Path:
    """Per-machine config overlay holding hardware settings & OpenAI API key."""
    host = hostname or get_hostname()
    return get_data_dir() / f"whisper_config.{host}.json"


def get_stats_file(hostname: Optional[str] = None) -> Path:
    """Per-machine stats file (``<data_dir>/whisper_stats.<HOST>.json``)."""
    host = hostname or get_hostname()
    return get_data_dir() / f"whisper_stats.{host}.json"


def get_legacy_stats_file() -> Path:
    """Legacy shared stats file path used for one-time baseline migration."""
    return get_data_dir() / STATS_FILENAME


def get_ai_insights_file() -> Path:
    """AI Daily Brief cache file (``<data_dir>/ai_insights.json``)."""
    return get_data_dir() / AI_INSIGHTS_FILENAME


def ensure_runtime_dirs() -> None:
    """Create standard runtime directories on startup."""
    for d in (
        get_app_support_dir(),
        get_data_dir(),
        get_transcriptions_dir(),
        get_models_dir(),
        get_logs_dir(),
    ):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    ensure_storage_bootstrap()
