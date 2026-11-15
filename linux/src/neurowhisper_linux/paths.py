"""XDG-compliant storage, resource discovery, sync-folder routing, and CUDA library discovery.

Separates read-only bundled assets (inside a mounted AppImage / PyInstaller bundle
or ``linux/resources``) from writable user directories under XDG Base Directory
conventions, while supporting an optional user-configured cloud Sync Folder
(Dropbox / Syncthing / Nextcloud) that remains 100% compatible with Windows and
macOS ``neurowhisper`` archives and per-machine stats files.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import site
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from . import APP_NAME

# Maximum safe Unix domain socket path length on Linux (sun_path is 108 bytes incl. NUL).
MAX_UNIX_SOCKET_PATH_LEN = 96

CONFIG_FILE_NAME = "whisper_config.json"
STATS_FILE_NAME = "whisper_stats.json"
AI_INSIGHTS_FILE_NAME = "ai_insights.json"
STORAGE_FILE_NAME = "storage.json"
TRANSCRIPTIONS_DIR_NAME = "transcriptions"
ICON_PNG_NAME = "neurowhisper.png"
DESKTOP_FILE_NAME = "neurowhisper.desktop"


def write_json_atomic(path: Path | str, data: object, indent: Optional[int] = 2) -> None:
    """Atomically write JSON to ``path`` using a PID+UUID sibling temp file and ``os.replace``."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=False)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        last_error: Optional[BaseException] = None
        for attempt in range(3):
            try:
                os.replace(tmp_path, target)
                return
            except PermissionError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.15)
        if last_error is not None:
            raise last_error
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def get_hostname() -> str:
    """Normalized machine hostname used for conflict-free multi-machine sync files."""
    return platform.node().replace(".", "_").strip() or "machine"


def get_resource_dir() -> Path:
    """Return the read-only directory containing bundled assets (``neurowhisper.png``, ``.desktop``)."""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            base = Path(meipass)
            nested = base / "resources"
            if nested.is_dir():
                return nested
            return base
        exe_dir = Path(sys.executable).resolve().parent
        for candidate in (exe_dir / "resources", exe_dir / "_internal" / "resources", exe_dir):
            if candidate.is_dir():
                return candidate
        return exe_dir
    # Source checkout: linux/src/neurowhisper_linux/paths.py -> linux/resources
    return Path(__file__).resolve().parents[2] / "resources"


def get_icon_png_path() -> Path:
    """Path to ``neurowhisper.png`` inside the bundle or ``linux/resources``."""
    res_dir = get_resource_dir()
    primary = res_dir / ICON_PNG_NAME
    if primary.is_file():
        return primary
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        alt = Path(getattr(sys, "_MEIPASS")) / ICON_PNG_NAME
        if alt.is_file():
            return alt
    return primary


def get_desktop_template_path() -> Path:
    """Path to ``neurowhisper.desktop`` template inside the bundle or ``linux/resources``."""
    return get_resource_dir() / DESKTOP_FILE_NAME


def _get_isolated_home() -> Optional[Path]:
    raw = os.environ.get("NEUROWHISPER_HOME", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def get_config_dir() -> Path:
    """User configuration directory (default ``~/.config/neurowhisper``)."""
    iso = _get_isolated_home()
    if iso is not None:
        return iso / "config"
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else (Path.home() / ".config")
    return base / APP_NAME


def get_data_dir() -> Path:
    """User data directory (default ``~/.local/share/neurowhisper``)."""
    iso = _get_isolated_home()
    if iso is not None:
        return iso / "data"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else (Path.home() / ".local" / "share")
    return base / APP_NAME


def get_cache_dir() -> Path:
    """User cache directory (default ``~/.cache/neurowhisper``)."""
    iso = _get_isolated_home()
    if iso is not None:
        return iso / "cache"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else (Path.home() / ".cache")
    return base / APP_NAME


def get_xdg_data_home() -> Path:
    """Base XDG data home used for ``applications/`` and ``icons/`` desktop integration."""
    iso = _get_isolated_home()
    if iso is not None:
        return iso / "xdg-data"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    return Path(xdg).expanduser() if xdg else (Path.home() / ".local" / "share")


def get_models_dir() -> Path:
    """Directory storing downloaded Whisper / ONNX / OpenVINO model weights."""
    return get_cache_dir() / "models"


def get_logs_dir() -> Path:
    """Directory storing rotating ``app.log``, ``crash.log``, and ``hotkey.log``."""
    return get_cache_dir() / "logs"


def get_cuda_libs_dir() -> Path:
    """User-space directory for optional CUDA 12 runtime libraries (``~/.local/share/neurowhisper/cuda-libs``)."""
    return get_data_dir() / "cuda-libs"


def _ensure_private_dir(directory: Path) -> bool:
    """Create ``directory`` with ``0700`` permissions and verify write/search access."""
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        return directory.is_dir() and os.access(directory, os.W_OK | os.X_OK)
    except OSError:
        return False


def get_socket_path() -> Path:
    """Resolve the single-instance & CLI IPC ``AF_UNIX`` control socket path.

    Priority:
    1. ``$NEUROWHISPER_SOCKET_PATH`` when explicitly set
    2. ``$NEUROWHISPER_HOME/run/control.sock`` when ``NEUROWHISPER_HOME`` is set (and <96 chars)
    3. ``$XDG_RUNTIME_DIR/neurowhisper/control.sock`` when writable and <96 chars
    4. ``/tmp/neurowhisper-<uid>/control.sock`` (with ``0700`` directory permissions)
    """
    explicit = os.environ.get("NEUROWHISPER_SOCKET_PATH", "").strip()
    if explicit:
        sock = Path(explicit).expanduser()
        _ensure_private_dir(sock.parent)
        return sock

    iso = _get_isolated_home()
    if iso is not None:
        iso_sock = iso / "run" / "control.sock"
        if len(str(iso_sock)) < MAX_UNIX_SOCKET_PATH_LEN and _ensure_private_dir(iso_sock.parent):
            return iso_sock
        digest = hashlib.sha256(str(iso.resolve()).encode("utf-8")).hexdigest()[:10]
        short_dir = Path(f"/tmp/nw-{os.getuid()}-{digest}")
        _ensure_private_dir(short_dir)
        return short_dir / "control.sock"

    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if xdg_runtime:
        rt_base = Path(xdg_runtime).expanduser()
        if rt_base.is_dir() and os.access(rt_base, os.W_OK | os.X_OK):
            candidate = rt_base / APP_NAME / "control.sock"
            if len(str(candidate)) < MAX_UNIX_SOCKET_PATH_LEN and _ensure_private_dir(candidate.parent):
                return candidate

    fallback_dir = Path(f"/tmp/{APP_NAME}-{os.getuid()}")
    _ensure_private_dir(fallback_dir)
    return fallback_dir / "control.sock"


# ---------------------------------------------------------------------------
# Bootstrap storage.json & optional cloud Sync Folder routing
# ---------------------------------------------------------------------------

def get_storage_file() -> Path:
    """Bootstrap pointer file inside ``config_dir`` storing ``{"sync_dir": "..."}``."""
    return get_config_dir() / STORAGE_FILE_NAME


def get_sync_dir() -> Optional[Path]:
    """Return the user-configured sync directory ``Path`` if set, else ``None``."""
    env_sync = os.environ.get("NEUROWHISPER_SYNC_DIR", "").strip()
    if env_sync:
        return Path(env_sync).expanduser()

    storage_path = get_storage_file()
    if not storage_path.is_file():
        return None
    try:
        with open(storage_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            raw = str(payload.get("sync_dir", "") or "").strip()
            if raw:
                return Path(raw).expanduser()
    except Exception:
        return None
    return None


def set_sync_dir(path: str | os.PathLike | None) -> Optional[Path]:
    """Persist ``sync_dir`` to ``storage.json`` atomically. Pass ``""`` or ``None`` to clear."""
    raw = str(path or "").strip()
    if raw:
        resolved = Path(raw).expanduser()
        resolved.mkdir(parents=True, exist_ok=True)
        write_json_atomic(get_storage_file(), {"sync_dir": str(resolved)}, indent=2)
        return resolved
    write_json_atomic(get_storage_file(), {"sync_dir": ""}, indent=2)
    return None


def get_active_config_dir() -> Path:
    """Directory holding ``whisper_config.json`` and ``whisper_config.<HOST>.json``."""
    sync_dir = get_sync_dir()
    return sync_dir if sync_dir is not None else get_config_dir()


def get_active_data_dir() -> Path:
    """Directory holding ``transcriptions/``, ``whisper_stats.<HOST>.json``, and ``ai_insights.json``."""
    sync_dir = get_sync_dir()
    return sync_dir if sync_dir is not None else get_data_dir()


def get_shared_config_file() -> Path:
    return get_active_config_dir() / CONFIG_FILE_NAME


def get_local_config_file(hostname: Optional[str] = None) -> Path:
    host = hostname or get_hostname()
    return get_active_config_dir() / f"whisper_config.{host}.json"


def get_stats_dir() -> Path:
    return get_active_data_dir()


def get_stats_file(hostname: Optional[str] = None) -> Path:
    host = hostname or get_hostname()
    return get_active_data_dir() / f"whisper_stats.{host}.json"


def get_legacy_stats_file() -> Path:
    return get_active_data_dir() / STATS_FILE_NAME


def get_ai_insights_file() -> Path:
    return get_active_data_dir() / AI_INSIGHTS_FILE_NAME


def get_transcriptions_dir() -> Path:
    return get_active_data_dir() / TRANSCRIPTIONS_DIR_NAME


def ensure_user_dirs() -> None:
    """Ensure writable XDG / active directories exist."""
    for d in (
        get_config_dir(),
        get_data_dir(),
        get_cache_dir(),
        get_models_dir(),
        get_logs_dir(),
        get_active_config_dir(),
        get_active_data_dir(),
        get_transcriptions_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dynamic CUDA 12 shared-library directory discovery
# ---------------------------------------------------------------------------

def _expand_nvidia_lib_subdirs(root: Path) -> List[Path]:
    """Return ``root`` plus any nested ``nvidia/*/lib`` directories inside ``root``."""
    found: List[Path] = []
    if not root.is_dir():
        return found
    found.append(root)
    for sub_name in ("lib", "lib64"):
        sub = root / sub_name
        if sub.is_dir():
            found.append(sub)
    nvidia_dir = root / "nvidia"
    if nvidia_dir.is_dir():
        for child in sorted(nvidia_dir.glob("*/lib")):
            if child.is_dir():
                found.append(child)
    # Support pip --prefix layouts such as cuda-libs/lib/python3.x/site-packages/nvidia/*/lib
    for nested_nv in sorted(root.glob("lib/python*/site-packages/nvidia/*/lib")):
        if nested_nv.is_dir():
            found.append(nested_nv)
    return found


def discover_cuda_lib_dirs() -> List[str]:
    """Discover existing directories that may provide CUDA 12 / cuBLAS / cuDNN ``.so`` files.

    Scans in priority order:
    1. ``$NEUROWHISPER_CUDA_LIB_DIR`` (colon-separated paths, incl. nested ``nvidia/*/lib``)
    2. ``~/.local/share/neurowhisper/cuda-libs`` (incl. ``pip install --target`` ``nvidia/*/lib``)
    3. Active Python / venv ``site-packages/nvidia/*/lib``
    4. Standard host CUDA toolkit locations (``/usr/local/cuda/lib64``, ``/opt/cuda/lib64``)
    """
    candidates: List[Path] = []

    env_dirs = os.environ.get("NEUROWHISPER_CUDA_LIB_DIR", "").strip()
    if env_dirs:
        for raw_part in env_dirs.split(os.pathsep):
            part = raw_part.strip()
            if part:
                candidates.extend(_expand_nvidia_lib_subdirs(Path(part).expanduser()))

    candidates.extend(_expand_nvidia_lib_subdirs(get_cuda_libs_dir()))

    # Active Python environment / user site-packages nvidia wheels
    site_roots: List[Path] = []
    for entry in list(sys.path):
        if entry:
            site_roots.append(Path(entry))
    try:
        for sp in site.getsitepackages():
            site_roots.append(Path(sp))
    except Exception:
        pass
    try:
        usp = site.getusersitepackages()
        if isinstance(usp, str) and usp:
            site_roots.append(Path(usp))
    except Exception:
        pass

    seen_site = set()
    for sr in site_roots:
        key = str(sr)
        if key in seen_site:
            continue
        seen_site.add(key)
        nv_root = sr / "nvidia"
        if nv_root.is_dir():
            for lib_dir in sorted(nv_root.glob("*/lib")):
                if lib_dir.is_dir():
                    candidates.append(lib_dir)

    # Host CUDA toolkit paths
    for sys_cuda in (
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/lib"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda/targets/sbsa-linux/lib"),
        Path("/opt/cuda/lib64"),
        Path("/opt/cuda/lib"),
    ):
        if sys_cuda.is_dir():
            candidates.append(sys_cuda)

    deduped: List[str] = []
    seen_real = set()
    for cand in candidates:
        try:
            if not cand.is_dir():
                continue
            resolved = str(cand.resolve())
        except OSError:
            continue
        if resolved not in seen_real:
            seen_real.add(resolved)
            deduped.append(resolved)
    return deduped
