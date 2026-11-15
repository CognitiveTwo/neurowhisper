#!/usr/bin/env bash
# Double-clickable Gatekeeper quarantine removal helper for ad-hoc signed builds.
set -euo pipefail

APP_NAME="neurowhisper.app"
TARGET_APP="/Applications/${APP_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  neurowhisper — First-Run Gatekeeper Quarantine Helper"
echo "============================================================"
echo ""

if [[ ! -d "${TARGET_APP}" ]]; then
    if [[ -d "${SCRIPT_DIR}/${APP_NAME}" ]]; then
        echo "Copying ${APP_NAME} to /Applications..."
        cp -R "${SCRIPT_DIR}/${APP_NAME}" "/Applications/"
    elif [[ -d "${HOME}/Applications/${APP_NAME}" ]]; then
        TARGET_APP="${HOME}/Applications/${APP_NAME}"
    else
        echo "ERROR: Could not find /Applications/${APP_NAME}."
        echo "Please drag 'neurowhisper.app' onto the 'Applications' folder first,"
        echo "then double-click Unquarantine.command again."
        echo ""
        read -r -p "Press Return to close..." _
        exit 1
    fi
fi

echo "Clearing macOS download quarantine attribute on: ${TARGET_APP}"
/usr/bin/xattr -cr "${TARGET_APP}"

echo ""
echo "Done! Launching neurowhisper..."
echo ""
echo "IMPORTANT — macOS Permissions:"
echo "  1. Allow Microphone access when prompted."
echo "  2. Grant Accessibility permission in:"
echo "     System Settings -> Privacy & Security -> Accessibility"
echo "     (If updating from an older .dmg, toggle neurowhisper OFF and ON once.)"
echo ""

/usr/bin/open "${TARGET_APP}"
exit 0
