"""Single-instance & Wayland/X11 CLI control over an ``AF_UNIX`` stream socket.

Allows external callers — including native Wayland compositor shortcuts such as
``neurowhisper --toggle-live`` or ``neurowhisper --toggle-batch`` — to control a
running ``neurowhisper`` instance in <5 ms with strict ``0700``/``0600`` permissions.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import paths

log = logging.getLogger("neurowhisper.ipc")

SUPPORTED_IPC_COMMANDS = (
    "toggle_live",
    "toggle_batch",
    "stop",
    "show",
    "status",
    "ping",
)


def send_ipc_command(
    cmd: str,
    timeout: float = 2.0,
    socket_path: Optional[Path | str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send a JSON command to a running ``neurowhisper`` IPC server and return its reply."""
    command = (cmd or "").strip().lower()
    if command not in SUPPORTED_IPC_COMMANDS:
        return {"ok": False, "error": f"Unsupported command: {cmd}"}

    sock_file = Path(socket_path).expanduser() if socket_path is not None else paths.get_socket_path()
    if not sock_file.exists():
        return {"ok": False, "error": "neurowhisper is not running (socket not found)"}

    req: Dict[str, Any] = {"cmd": command}
    if payload:
        req.update(payload)
    wire = (json.dumps(req) + "\n").encode("utf-8")

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(sock_file))
            client.sendall(wire)
            chunks = []
            while True:
                data = client.recv(4096)
                if not data:
                    break
                chunks.append(data)
                if b"\n" in data:
                    break
            raw_reply = b"".join(chunks).decode("utf-8", errors="replace").strip()
            if not raw_reply:
                return {"ok": False, "error": "Empty response from IPC server"}
            first_line = raw_reply.splitlines()[0]
            parsed = json.loads(first_line)
            if isinstance(parsed, dict):
                return parsed
            return {"ok": True, "result": parsed}
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        return {"ok": False, "error": f"neurowhisper is not running ({exc})"}
    except socket.timeout:
        return {"ok": False, "error": "Timed out waiting for neurowhisper IPC response"}
    except OSError as exc:
        return {"ok": False, "error": f"IPC socket error: {exc}"}
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Invalid IPC JSON reply: {exc}"}


def is_instance_running(
    socket_path: Optional[Path | str] = None,
    timeout: float = 0.75,
) -> bool:
    """Return ``True`` if an active ``neurowhisper`` IPC server responds to ``ping``."""
    resp = send_ipc_command("ping", timeout=timeout, socket_path=socket_path)
    return bool(resp.get("ok"))


class IPCServer:
    """Background daemon thread listening on ``paths.get_socket_path()`` for CLI commands."""

    def __init__(
        self,
        handler: Callable[[str], Dict[str, Any]],
        socket_path: Optional[Path | str] = None,
    ) -> None:
        self._handler = handler
        self.socket_path: Path = (
            Path(socket_path).expanduser() if socket_path is not None else paths.get_socket_path()
        )
        self._server_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._owns_socket = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def start(self) -> bool:
        """Bind the ``AF_UNIX`` control socket and start the accept loop.

        Returns ``False`` if another live ``neurowhisper`` instance already owns the socket.
        """
        with self._lock:
            if self.is_running:
                return True

            parent = self.socket_path.parent
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(parent, 0o700)
            except OSError:
                pass

            if self.socket_path.exists():
                if is_instance_running(socket_path=self.socket_path, timeout=0.5):
                    log.info("Active neurowhisper instance already listening on %s", self.socket_path)
                    return False
                try:
                    self.socket_path.unlink()
                except OSError as exc:
                    log.warning("Could not unlink stale socket %s: %s", self.socket_path, exc)
                    return False

            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                srv.bind(str(self.socket_path))
                try:
                    os.chmod(self.socket_path, 0o600)
                except OSError:
                    pass
                srv.listen(8)
                srv.settimeout(0.4)
            except OSError as exc:
                srv.close()
                log.warning("Failed to bind IPC socket %s: %s", self.socket_path, exc)
                return False

            self._stop_event.clear()
            self._server_sock = srv
            self._owns_socket = True
            self._thread = threading.Thread(
                target=self._serve_loop,
                name="neurowhisper-ipc",
                daemon=True,
            )
            self._thread.start()
            atexit.register(self.stop)
            log.info("IPC control socket listening on %s", self.socket_path)
            return True

    def stop(self) -> None:
        """Stop the IPC accept loop and remove the ``AF_UNIX`` socket file."""
        with self._lock:
            self._stop_event.set()
            srv = self._server_sock
            self._server_sock = None
            if srv is not None:
                try:
                    srv.close()
                except OSError:
                    pass
            thread = self._thread
            self._thread = None

        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.5)

        with self._lock:
            if self._owns_socket:
                self._owns_socket = False
                try:
                    if self.socket_path.exists():
                        self.socket_path.unlink()
                except OSError:
                    pass

    def _serve_loop(self) -> None:
        while not self._stop_event.is_set():
            srv = self._server_sock
            if srv is None:
                break
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                try:
                    conn.settimeout(2.0)
                    raw = b""
                    while len(raw) < 8192:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        raw += chunk
                        if b"\n" in chunk:
                            break
                    reply = self._dispatch_raw(raw.decode("utf-8", errors="replace").strip())
                except Exception as exc:
                    log.debug("IPC connection handling error: %s", exc)
                    reply = {"ok": False, "error": str(exc)}

                try:
                    conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                except OSError:
                    pass

    def _dispatch_raw(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"ok": False, "error": "Empty command"}
        first_line = text.splitlines()[0].strip()
        cmd = ""
        if first_line.startswith("{"):
            try:
                obj = json.loads(first_line)
                if isinstance(obj, dict):
                    cmd = str(obj.get("cmd", "")).strip().lower()
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"Malformed JSON: {exc}"}
        else:
            cmd = first_line.lower()

        if cmd not in SUPPORTED_IPC_COMMANDS:
            return {"ok": False, "error": f"Unknown command: {cmd}"}

        if cmd == "ping":
            try:
                extra = self._handler("ping") or {}
                if isinstance(extra, dict):
                    res = {"ok": True, "pid": os.getpid()}
                    res.update(extra)
                    return res
            except Exception:
                pass
            return {"ok": True, "pid": os.getpid()}

        try:
            response = self._handler(cmd)
            if isinstance(response, dict):
                if "ok" not in response:
                    response = {"ok": True, **response}
                return response
            return {"ok": True, "result": response}
        except Exception as exc:
            log.exception("IPC handler error for %s", cmd)
            return {"ok": False, "error": str(exc)}
