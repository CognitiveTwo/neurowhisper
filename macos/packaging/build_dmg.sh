#!/usr/bin/env bash
# Build standalone ad-hoc signed neurowhisper.app and compressed macOS .dmg installer.
# Requires: macOS 13+ (arm64 or x86_64) and `uv` (https://docs.astral.sh/uv/).
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: build_dmg.sh must be executed on macOS (Darwin)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACOS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RES_DIR="${MACOS_DIR}/resources"
ARCH="${TARGET_ARCH:-$(uname -m)}"
PY_VER="${PYTHON_VERSION:-3.12}"
VERSION="${NEUROWHISPER_VERSION:-1.0.0}"

BUILD_ROOT="${NEUROWHISPER_BUILD_ROOT:-/tmp/neurowhisper-macos-build-${ARCH}}"
VENV_DIR="${BUILD_ROOT}/venv"
PYI_WORK_DIR="${BUILD_ROOT}/pyi-work"
PYI_DIST_DIR="${BUILD_ROOT}/pyi-dist"
STAGE_DIR="${BUILD_ROOT}/dmg-stage"
OUT_DIR="${NEUROWHISPER_DIST_DIR:-${MACOS_DIR}/dist}"
OUT_DMG="${OUT_DIR}/neurowhisper-macos-${ARCH}.dmg"
BUNDLE_ID="com.neuroflash.neurowhisper"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is required to provision relocatable Python ${PY_VER} (Tcl/Tk 8.6.14+)." >&2
    echo "Install uv via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "==> [1/6] Provisioning isolated uv Python ${PY_VER} build environment in ${VENV_DIR}"
rm -rf "${PYI_WORK_DIR}" "${PYI_DIST_DIR}" "${STAGE_DIR}"
mkdir -p "${BUILD_ROOT}" "${OUT_DIR}"

uv venv --python "${PY_VER}" --clear "${VENV_DIR}"
uv pip install --python "${VENV_DIR}/bin/python" --upgrade pip "pyinstaller>=6.10.0"
uv pip install --python "${VENV_DIR}/bin/python" -r "${MACOS_DIR}/requirements-macos.txt"

if [[ ! -f "${RES_DIR}/icon.icns" ]]; then
    echo "==> Generating ${RES_DIR}/icon.icns from icon.png"
    "${VENV_DIR}/bin/python" -c "
from PIL import Image
img = Image.open('${RES_DIR}/icon.png').convert('RGBA')
img.save('${RES_DIR}/icon.icns', format='ICNS')
"
fi

echo "==> [2/6] Building PyInstaller onedir bundle (neurowhisper.app)"
"${VENV_DIR}/bin/pyinstaller" \
    --noconfirm \
    --clean \
    --workpath "${PYI_WORK_DIR}" \
    --distpath "${PYI_DIST_DIR}" \
    "${SCRIPT_DIR}/neurowhisper-macos.spec"

APP_BUNDLE="${PYI_DIST_DIR}/neurowhisper.app"
if [[ ! -d "${APP_BUNDLE}" ]]; then
    echo "ERROR: Expected bundle not found at ${APP_BUNDLE}" >&2
    exit 1
fi

echo "==> [3/6] Ad-hoc code signing inner frameworks/dylibs and outer .app with entitlements"
find "${APP_BUNDLE}/Contents/Frameworks" "${APP_BUNDLE}/Contents/Resources" \
    -type f \( -name "*.dylib" -o -name "*.so" \) -print0 2>/dev/null | \
    while IFS= read -r -d '' bin_file; do
        /usr/bin/codesign --force --sign - --timestamp=none "${bin_file}"
    done

/usr/bin/codesign \
    --force \
    --deep \
    --sign - \
    --timestamp=none \
    --identifier "${BUNDLE_ID}" \
    --entitlements "${RES_DIR}/entitlements.plist" \
    "${APP_BUNDLE}"

echo "==> [4/6] Verifying code signature integrity"
/usr/bin/codesign --verify --deep --strict --verbose=2 "${APP_BUNDLE}"

echo "==> [5/6] Staging DMG contents (preserving symlinks & code signatures via ditto)"
mkdir -p "${STAGE_DIR}"
/usr/bin/ditto "${APP_BUNDLE}" "${STAGE_DIR}/neurowhisper.app"
ln -s /Applications "${STAGE_DIR}/Applications"
cp "${SCRIPT_DIR}/Unquarantine.command" "${STAGE_DIR}/Unquarantine.command"
chmod +x "${STAGE_DIR}/Unquarantine.command"

cat > "${STAGE_DIR}/FIRST_RUN_README.txt" <<'EOF'
neurowhisper for macOS — Quick Installation & Permissions Guide
===============================================================

1. INSTALL
   Drag "neurowhisper.app" onto the "Applications" folder shortcut in this window.

2. FIRST LAUNCH (Ad-Hoc Signed Build)
   Because this build is community/ad-hoc signed (no Apple notarization fee),
   macOS Gatekeeper may block the first launch of a browser-downloaded DMG.
   Simply double-click "Unquarantine.command" in this window (or run:
     xattr -cr /Applications/neurowhisper.app
   in Terminal) after dragging neurowhisper.app into Applications.

3. MICROPHONE & ACCESSIBILITY PERMISSIONS
   - Microphone: Click "Allow" when macOS prompts on your first recording.
   - Accessibility (for global hotkeys & live typing into other apps):
     Click "Grant Accessibility" inside neurowhisper (or open
     System Settings -> Privacy & Security -> Accessibility and enable neurowhisper).
   - Updating from a previous DMG: macOS ties Accessibility trust to the exact
     build hash. After replacing neurowhisper.app, toggle neurowhisper OFF and
     back ON once in Accessibility settings, then click "Re-check" in the app.
EOF

echo "==> [6/6] Creating compressed UDZO disk image: ${OUT_DMG}"
rm -f "${OUT_DMG}"
attempt=1
while (( attempt <= 3 )); do
    if /usr/bin/hdiutil create \
        -volname "neurowhisper" \
        -srcfolder "${STAGE_DIR}" \
        -ov \
        -format UDZO \
        "${OUT_DMG}"; then
        break
    fi
    echo "hdiutil attempt ${attempt} busy; retrying in 3s..."
    sleep 3
    (( attempt++ ))
done

if [[ ! -f "${OUT_DMG}" ]]; then
    echo "ERROR: Failed to produce ${OUT_DMG}" >&2
    exit 1
fi

/usr/bin/shasum -a 256 "${OUT_DMG}" > "${OUT_DMG}.sha256"
rm -rf "${OUT_DIR}/neurowhisper.app"
/usr/bin/ditto "${APP_BUNDLE}" "${OUT_DIR}/neurowhisper.app"

echo ""
echo "SUCCESS! Built standalone macOS disk image:"
echo "  DMG:    ${OUT_DMG}"
echo "  SHA256: $(cat "${OUT_DMG}.sha256")"
