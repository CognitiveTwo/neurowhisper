#!/usr/bin/env python3
"""Headless smoke test for neurowhisper.

Boots the real application into a throwaway directory with no config, no stats
and no model, lets the Tk event loop run for a few seconds, and checks that:

  * ``app_config``, ``platform_utils`` and ``backends`` import cleanly
  * ``detect_best_backend()`` returns a populated ``(backend, device, reason)``
  * ``WhisperApp`` constructs and runs without raising inside a Tk callback
  * the status bar leaves the initial "Loading..." state - either it is
    downloading, ready, or it says the model load failed. Any of those is fine;
    a silent failure is not.
  * ``on_close()`` completes within 5 seconds

Standard library only. Exits non-zero on failure.

The ``keyboard`` module is stubbed before the app is imported so this never
installs a global keyboard hook (which would fight with a real running
instance, and would need root on Linux).
"""

import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GUI_SETTLE_SECONDS = 8.0
CLOSE_BUDGET_SECONDS = 5.0

# Files/directories the app needs in its working directory to start up.
APP_FILES = ("whisper_gui.pyw", "app_config.py", "platform_utils.py", "icon.ico")
APP_DIRS = ("backends",)

_failures = []


def fail(message):
    _failures.append(message)
    print("FAIL: %s" % message, flush=True)


def ok(message):
    print("ok  : %s" % message, flush=True)


def note(message):
    print("note: %s" % message, flush=True)


# --------------------------------------------------------------------------- #
# Stubs - must be installed before whisper_gui (or any backend) is imported.
# --------------------------------------------------------------------------- #

def install_stubs():
    """Replace the modules that would touch real hardware."""
    kb = types.ModuleType("keyboard")

    def _noop(*args, **kwargs):
        return None

    for name in (
        "remove_hotkey", "write", "send", "press_and_release", "hook", "unhook",
        "on_release", "on_press", "unhook_all", "unhook_all_hotkeys", "wait",
        "release", "press", "read_event",
    ):
        setattr(kb, name, _noop)
    kb.add_hotkey = lambda *a, **k: object()
    kb.is_pressed = lambda *a, **k: False
    sys.modules["keyboard"] = kb

    # pynput is only used on macOS, but stub it anywhere that is not Windows so
    # a Linux CI runner never tries to open an X display for it.
    if sys.platform != "win32":
        pynput = types.ModuleType("pynput")
        pynput_kb = types.ModuleType("pynput.keyboard")

        class _Controller(object):
            def type(self, *a, **k):
                return None

            def press(self, *a, **k):
                return None

            def release(self, *a, **k):
                return None

        class _Listener(object):
            def __init__(self, *a, **k):
                pass

            def start(self):
                return None

            def stop(self):
                return None

            def join(self, *a, **k):
                return None

        class _Key(object):
            pass

        pynput_kb.Controller = _Controller
        pynput_kb.Listener = _Listener
        pynput_kb.GlobalHotKeys = _Listener
        pynput_kb.Key = _Key
        pynput_kb.KeyCode = _Key
        pynput.keyboard = pynput_kb
        sys.modules["pynput"] = pynput
        sys.modules["pynput.keyboard"] = pynput_kb

    # sounddevice: pretend the machine has no microphone at all. CI runners have
    # no audio hardware, and the app must survive that.
    try:
        import sounddevice as sd  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the runner
        note("real sounddevice unavailable (%s); using a stub" % exc)
        sd = types.ModuleType("sounddevice")
        sd.default = types.SimpleNamespace(device=None, samplerate=None,
                                           channels=None, dtype=None)
        sd.PortAudioError = type("PortAudioError", (Exception,), {})
        sd.check_input_settings = lambda *a, **k: None
        sd.sleep = lambda *a, **k: None
        sd.stop = lambda *a, **k: None
        sys.modules["sounddevice"] = sd

    sd.query_devices = lambda *a, **k: []
    sd.query_hostapis = lambda *a, **k: []

    def _no_input_stream(*args, **kwargs):
        raise RuntimeError("smoke test: no audio input device available")

    sd.InputStream = _no_input_stream


def load_whisper_gui(app_dir):
    """Import whisper_gui.pyw as a module (the .pyw suffix blocks plain import)."""
    import importlib.util

    path = os.path.join(app_dir, "whisper_gui.pyw")
    spec = importlib.util.spec_from_file_location("whisper_gui", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["whisper_gui"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def check_imports():
    import app_config
    ok("imported app_config (%d default config keys)" % len(app_config.DEFAULT_CONFIG))

    import platform_utils  # noqa: F401
    ok("imported platform_utils")

    import backends
    result = backends.detect_best_backend()
    if not isinstance(result, tuple):
        fail("detect_best_backend() returned %r, expected a tuple" % (type(result),))
        return
    if len(result) != 3:
        fail("detect_best_backend() returned %d values, expected 3" % len(result))
        return
    name, device, reason = result
    if not isinstance(reason, str) or not reason.strip():
        fail("detect_best_backend() reason is empty: %r" % (reason,))
        return
    ok("detect_best_backend() -> backend=%r device=%r reason=%r"
       % (name, device, reason))


def make_temp_app_dir():
    tmp = tempfile.mkdtemp(prefix="neurowhisper_smoke_")
    for name in APP_FILES:
        src = os.path.join(REPO_ROOT, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp, name))
        else:
            note("optional file missing from the repo: %s" % name)
    for name in APP_DIRS:
        src = os.path.join(REPO_ROOT, name)
        if not os.path.isdir(src):
            fail("required directory missing from the repo: %s" % name)
            continue
        shutil.copytree(src, os.path.join(tmp, name),
                        ignore=shutil.ignore_patterns("__pycache__"))
    return tmp


def tk_usable():
    """True if a Tk root can actually be created here."""
    try:
        import tkinter as tk
    except Exception as exc:
        note("tkinter is not importable (%s)" % exc)
        return False
    try:
        probe = tk.Tk()
    except Exception as exc:
        note("tkinter cannot create a root window (%s)" % exc)
        return False
    probe.destroy()
    return True


def check_gui(app_dir):
    if not tk_usable():
        print("SKIP: GUI checks skipped - no usable Tk display. "
              "On a headless Linux box run this under xvfb-run.", flush=True)
        return

    import tkinter as tk

    module = load_whisper_gui(app_dir)
    ok("imported whisper_gui.pyw")

    errors = []

    def report_callback_exception(exc, val, tb):
        errors.append("".join(traceback.format_exception(exc, val, tb)))

    root = tk.Tk()
    root.report_callback_exception = report_callback_exception

    app = module.WhisperApp(root)
    ok("constructed WhisperApp")

    root.after(int(GUI_SETTLE_SECONDS * 1000), root.quit)
    root.mainloop()
    ok("Tk event loop ran for %.0f s" % GUI_SETTLE_SECONDS)

    status = ""
    try:
        status = app.status_var.get()
    except Exception as exc:
        fail("could not read status_var: %s" % exc)
    print("status bar: %r" % status, flush=True)

    if errors:
        fail("%d Tk callback error(s) during startup" % len(errors))
        for text in errors:
            print(text, flush=True)
    else:
        ok("no Tk callback errors")

    # Explicit pass conditions, not "anything but Loading...". With
    # HF_HUB_OFFLINE=1 and no model on disk the load MUST fail, and the app
    # must say so; the only other accepted outcome is a model that happened to
    # be present already, which loads and reports Ready.
    normalised = status.strip().lower()
    failure_markers = (
        "model load failed",
        "is not downloaded yet",
        "no internet connection",
    )
    if any(marker in normalised for marker in failure_markers):
        ok("model-load failure is reported in the status bar: %r" % status)
    elif normalised.startswith("ready"):
        ok("a model was already present and loaded: %r" % status)
    elif normalised in ("", "loading...", "loading"):
        fail("status bar is still %r after %.0f s - the model load neither "
             "progressed nor reported a failure" % (status, GUI_SETTLE_SECONDS))
    else:
        fail("status bar reads %r - expected the offline model-load failure "
             "message or Ready" % status)

    # on_close() must not hang. A watchdog kills the process if it does, so a
    # deadlock surfaces as a test failure instead of a stuck CI job.
    def _watchdog():
        print("FAIL: on_close() did not return within %.0f s"
              % CLOSE_BUDGET_SECONDS, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)

    timer = threading.Timer(CLOSE_BUDGET_SECONDS, _watchdog)
    timer.daemon = True
    timer.start()
    started = time.monotonic()
    app.on_close()
    elapsed = time.monotonic() - started
    timer.cancel()
    ok("on_close() returned in %.2f s" % elapsed)


def main():
    # Status strings contain non-ASCII characters; never let a legacy console
    # encoding turn a passing test into a UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Suppress modal dialogs: a messagebox on a CI runner blocks forever.
    os.environ["NEUROWHISPER_NO_DIALOGS"] = "1"
    os.environ.setdefault("PYTHONUTF8", "1")
    # Never pull the ~460 MB model on CI. Offline mode makes huggingface_hub
    # fail immediately instead, which is exactly the path this test is here to
    # check: the app must REPORT that failure rather than sit on "Loading...".
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    install_stubs()

    app_dir = make_temp_app_dir()
    print("app copy: %s" % app_dir, flush=True)

    original_cwd = os.getcwd()
    os.chdir(app_dir)
    sys.path.insert(0, app_dir)
    try:
        check_imports()
        check_gui(app_dir)
    except Exception:
        fail("unhandled exception:\n%s" % traceback.format_exc())
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(app_dir, ignore_errors=True)

    print("", flush=True)
    if _failures:
        print("SMOKE TEST FAILED (%d problem(s))" % len(_failures), flush=True)
        for item in _failures:
            print("  - %s" % item, flush=True)
        return 1
    print("SMOKE TEST PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
