"""Single-instance lock and CLI control over an AF_UNIX domain socket."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import socket
import threading
from typing import Any, Callable, Dict, Optional

from . import paths

logger = logging.getLogger(__name__)

SUPPORTED_IPC_COMMANDS = frozenset({
    "toggle_live",
    "toggle_batch",
    "stop",
    "show",
    "status",
    "ping",
    "quit",
})


class IPCAlreadyRunningError(RuntimeError):
    """Raised when another neurowhisper instance is already listening on the IPC socket."""


IPCHandler = Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]


def send_ipc_command(
    cmd: str,
    socket_path: Optional[Path] = None,
    timeout: float = 2.0,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send a newline-delimited JSON command to a running neurowhisper instance."""
    sock_file = Path(socket_path) if socket_path is not None else paths.get_socket_path()
    if not sock_file.exists():
        return {"ok": False, "error": "socket_not_found", "socket": str(sock_file)}

    message: Dict[str, Any] = {}
    if payload:
        message.update(payload)
    message["cmd"] = str(cmd).strip()

    raw = (json.dumps(message) + "\n").encode("utf-8")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(sock_file))
            client.sendall(raw)
            chunks = []
            while True:
                data = client.recv(4096)
                if not data:
                    break
                chunks.append(data)
                if b"\n" in data:
                    break
        if not chunks:
            return {"ok": False, "error": "empty_response"}
        line = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
        if not line:
            return {"ok": False, "error": "empty_response"}
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            return parsed
        return {"ok": True, "result": parsed}
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        return {"ok": False, "error": "not_running", "detail": str(exc)}
    except socket.timeout:
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def is_instance_running(socket_path: Optional[Path] = None, timeout: float = 0.75) -> bool:
    """Return True if a live neurowhisper instance responds to ``ping``."""
    resp = send_ipc_command("ping", socket_path=socket_path, timeout=timeout)
    return bool(resp.get("ok"))


class IPCServer:
    """Daemon AF_UNIX socket server handling CLI toggles and duplicate-launch activation."""

    def __init__(
        self,
        handler: IPCHandler,
        socket_path: Optional[Path] = None,
    ) -> None:
        self._handler = handler
        self.socket_path: Path = Path(socket_path) if socket_path is not None else paths.get_socket_path()
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._owns_socket = False
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def start(self, check_existing: bool = True) -> None:
        """Bind the AF_UNIX socket (0700 parent dir, 0600 socket) and start the listener thread."""
        with self._lock:
            if self.running:
                return

            parent = self.socket_path.parent
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(parent, 0o700)
            except OSError:
                pass

            if self.socket_path.exists():
                if check_existing and is_instance_running(self.socket_path):
                    raise IPCAlreadyRunningError(
                        f"Another neurowhisper instance is already listening at {self.socket_path}"
                    )
                try:
                    self.socket_path.unlink()
                except OSError as exc:
                    logger.warning("Failed to unlink stale IPC socket %s: %s", self.socket_path, exc)

            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.settimeout(0.25)
            prev_umask = os.umask(0o177)
            try:
                srv.bind(str(self.socket_path))
            finally:
                os.umask(prev_umask)

            try:
                os.chmod(self.socket_path, 0o600)
            except OSError:
                pass

            srv.listen(8)
            self._sock = srv
            self._owns_socket = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._serve_loop,
                name="neurowhisper-ipc",
                daemon=True,
            )
            self._thread.start()

    def _serve_loop(self) -> None:
        srv = self._sock
        if srv is None:
            return
        while not self._stop_event.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                continue
            try:
                self._handle_client(conn)
            except Exception as exc:
                logger.debug("IPC client error: %s", exc)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle_client(self, conn: socket.socket) -> None:
        conn.settimeout(1.5)
        chunks = []
        total = 0
        while total < 16384:
            data = conn.recv(4096)
            if not data:
                break
            chunks.append(data)
            total += len(data)
            if b"\n" in data:
                break

        raw_line = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
        if not raw_line:
            self._send_json(conn, {"ok": False, "error": "empty_request"})
            return

        payload: Dict[str, Any]
        try:
            parsed = json.loads(raw_line)
            if isinstance(parsed, dict):
                payload = parsed
                cmd = str(parsed.get("cmd", "")).strip()
            else:
                cmd = str(parsed).strip()
                payload = {"cmd": cmd}
        except json.JSONDecodeError:
            cmd = raw_line
            payload = {"cmd": cmd}

        if not cmd or cmd not in SUPPORTED_IPC_COMMANDS:
            self._send_json(conn, {"ok": False, "error": f"unknown_command:{cmd}"})
            return

        if cmd == "ping":
            self._send_json(conn, {"ok": True, "cmd": "ping", "pid": os.getpid()})
            return

        try:
            result = self._handler(cmd, payload) or {}
            response: Dict[str, Any] = {"ok": True, "cmd": cmd}
            if isinstance(result, dict):
                response.update(result)
            self._send_json(conn, response)
        except Exception as exc:
            logger.exception("Error handling IPC command %r", cmd)
            self._send_json(conn, {"ok": False, "cmd": cmd, "error": str(exc)})

    @staticmethod
    def _send_json(conn: socket.socket, data: Dict[str, Any]) -> None:
        encoded = (json.dumps(data) + "\n").encode("utf-8")
        conn.sendall(encoded)

    def stop(self) -> None:
        """Stop the server thread and clean up the socket file."""
        with self._lock:
            self._stop_event.set()
            srv = self._sock
            self._sock = None
            if srv is not None:
                try:
                    srv.close()
                except OSError:
                    pass
            thread = self._thread
            self._thread = None
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=1.5)
            if self._owns_socket:
                self._owns_socket = False
                try:
                    if self.socket_path.exists():
                        self.socket_path.unlink()
                except OSError:
                    pass

    close = stop
