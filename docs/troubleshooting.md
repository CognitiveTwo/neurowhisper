# NeuroWhisper Troubleshooting Notes

Running log of performance issues, root causes, and fixes. Append new entries at the top.

---

## 2026-05-07 — System-wide input lag from keyboard-hook watchdog churn

### Symptom
Whole machine felt sluggish (typing lag, delayed mouse response) while NeuroWhisper was running. Same family of symptom as the 2025 GUI-loop incident noted in `readme.md` (line 76).

### Diagnosis
`logs/hotkey.log` showed the keyboard-listener watchdog force-restarting the Windows hook every 30–60 seconds:

```
Hook possibly dead (1/2): hook silent 30s but system idle only 0.0s
>>> FORCE RESTART of keyboard listener <<<
Listener restarted (total: 764)
```

- 700–900 restarts per session across `hotkey.log`, `hotkey.log.1`, `hotkey.log.2`.
- Each restart tears down and reinstalls a `WH_KEYBOARD_LL` global hook.
- Every keystroke on the machine is marshaled through that hook, so constant teardown/reinstall = system-wide input lag.

### Root cause
`HotkeyManager._is_hook_alive()` in `whisper_gui.pyw` (~line 304) infers liveness from:

```
hook_silent = now - self._last_hook_event   # only updated on KEY events
system_idle = GetLastInputInfo()             # counts MOUSE input too
if hook_silent > 30 and system_idle < 30 → declare dead
```

`GetLastInputInfo` resets on mouse movement. Normal "user moves mouse, doesn't type" behavior trips the heuristic. False positive → force restart → repeat. The signature in the log is `hook silent 30s but system idle only 0.0s` — confirms mouse-only activity, not a real dead hook.

### Fix applied
Raised the silence threshold so the heuristic only catches genuinely dead hooks. `_listener_thread_health()` remains the reliable primary signal (unchanged).

```python
HOOK_DEAD_THRESHOLD_S = 300   # was 30
HOOK_DEAD_CONFIRM_CYCLES = 4  # was 2
```

A "dead hook" must now stay silent 5 min and be re-confirmed across 4 cycles before any restart. Real dead hooks are still caught within ~20 min.

### How to verify
After restarting the app, tail `logs/hotkey.log` for a few hours. `Listener restarted (total: N)` should stay at 0 or single digits, not climb into the hundreds.

### If the lag is still present
Next suspects, in order:
1. Tk `after()` update loop in `whisper_gui.pyw` — check the adaptive refresh-rate code referenced in `readme.md:69-76` is actually engaging when idle.
2. Audio capture thread spinning — check backends/.
3. CUDA/cuDNN context churn — check `temp_online_seg_*.wav` cleanup and model reload patterns.

### Better fix (deferred)
The threshold bump is a heuristic patch. A correct fix would distinguish keyboard-only idle from system idle — e.g., install a low-level mouse hook just to track mouse activity, then compute `keyboard_idle = system_idle adjusted for mouse-only events`. Not worth the complexity unless the threshold approach proves insufficient.

---

## Earlier — Adaptive GUI loop (per readme.md)

The original 30 ms fixed Tk update loop burned ~30% of a CPU core continuously and caused system-wide typing lag on Windows even when the app was idle. Replaced with an adaptive refresh rate that drops to ~0% CPU when idle. See `readme.md:69-76`.
