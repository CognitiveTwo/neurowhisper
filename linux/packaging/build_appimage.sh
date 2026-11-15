#!/usr/bin/env bash
# Build a standalone, single-file neurowhisper-linux-<arch>.AppImage
# Uses relocatable Python 3.12 (via uv python-build-standalone with bundled Tcl/Tk 8.6+)
# + PyInstaller onedir + SquashFS appimagetool (zero /tmp extraction lag on launch).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINUX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_ROOT="${LINUX_DIR}/build"
DIST_ROOT="${LINUX_DIR}/dist"
VENV_DIR="${NEUROWHISPER_BUILD_VENV:-${BUILD_ROOT}/venv-py312}"
PYTHON_VERSION="${NEUROWHISPER_PYTHON_VERSION:-${PYTHON_VERSION:-3.12}}"

UNAME_ARCH="${TARGET_ARCH:-$(uname -m)}"
case "${UNAME_ARCH}" in
    x86_64|amd64)
        APPIMAGE_ARCH="x86_64"
        ;;
    aarch64|arm64)
        APPIMAGE_ARCH="aarch64"
        ;;
    *)
        echo "Unsupported Linux architecture: ${UNAME_ARCH}" >&2
        exit 1
        ;;
esac

echo "==> Building standalone neurowhisper Linux AppImage (${APPIMAGE_ARCH})"
mkdir -p "${BUILD_ROOT}" "${DIST_ROOT}"

# 1. Provision isolated relocatable Python 3.12 build environment (with bundled Tcl/Tk)
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    if command -v uv >/dev/null 2>&1; then
        echo "==> Creating relocatable Python ${PYTHON_VERSION} venv via uv..."
        uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
    else
        echo "==> uv not found; creating venv with host python3..."
        python3 -m venv "${VENV_DIR}"
    fi
fi

PY_BIN="${VENV_DIR}/bin/python"

echo "==> Installing Linux runtime dependencies & PyInstaller..."
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PY_BIN}" \
        -r "${LINUX_DIR}/requirements-linux.txt" \
        "pyinstaller>=6.6.0"
else
    "${PY_BIN}" -m pip install --upgrade pip wheel
    "${PY_BIN}" -m pip install \
        -r "${LINUX_DIR}/requirements-linux.txt" \
        "pyinstaller>=6.6.0"
fi

# Verify Tkinter is functional in the build Python before freezing
"${PY_BIN}" -c "import tkinter, _tkinter; print('Tcl/Tk version:', tkinter.TkVersion)"

# 2. Run PyInstaller onedir build
echo "==> Running PyInstaller onedir build..."
rm -rf "${BUILD_ROOT}/pyinstaller" "${DIST_ROOT}/neurowhisper"
"${VENV_DIR}/bin/pyinstaller" \
    --noconfirm \
    --clean \
    --workpath "${BUILD_ROOT}/pyinstaller" \
    --distpath "${DIST_ROOT}" \
    "${SCRIPT_DIR}/neurowhisper-linux.spec"

# 3. Assemble FreeDesktop AppDir layout
APPDIR="${BUILD_ROOT}/neurowhisper.AppDir"
echo "==> Assembling AppDir at ${APPDIR}..."
rm -rf "${APPDIR}"
mkdir -p \
    "${APPDIR}/usr/bin" \
    "${APPDIR}/usr/lib" \
    "${APPDIR}/usr/share/applications" \
    "${APPDIR}/usr/share/icons/hicolor/256x256/apps"

cp -a "${DIST_ROOT}/neurowhisper/." "${APPDIR}/usr/bin/"
cp "${LINUX_DIR}/resources/AppRun" "${APPDIR}/AppRun"
chmod +x "${APPDIR}/AppRun" "${APPDIR}/usr/bin/neurowhisper"

cp "${LINUX_DIR}/resources/neurowhisper.desktop" "${APPDIR}/neurowhisper.desktop"
cp "${LINUX_DIR}/resources/neurowhisper.desktop" "${APPDIR}/usr/share/applications/neurowhisper.desktop"
cp "${LINUX_DIR}/resources/neurowhisper.png" "${APPDIR}/neurowhisper.png"
cp "${LINUX_DIR}/resources/neurowhisper.png" "${APPDIR}/.DirIcon"
cp "${LINUX_DIR}/resources/neurowhisper.png" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/neurowhisper.png"

# Copy PortAudio shared library into AppDir/usr/lib if available on build host
for pa_candidate in \
    "/usr/lib/${UNAME_ARCH}-linux-gnu/libportaudio.so.2" \
    "/usr/lib64/libportaudio.so.2" \
    "/usr/lib/libportaudio.so.2"
do
    if [[ -f "${pa_candidate}" ]]; then
        cp -L "${pa_candidate}" "${APPDIR}/usr/lib/libportaudio.so.2"
        break
    fi
done

# Sanity-check frozen --version and --fw-worker hook before SquashFS compression
echo "==> Verifying frozen binary CLI & --fw-worker entrypoint..."
"${APPDIR}/AppRun" --version
printf '{"cmd":"shutdown"}\n' | "${APPDIR}/AppRun" --fw-worker >/dev/null

# 4. Package AppDir into a SquashFS .AppImage via appimagetool
APPIMAGETOOL_BIN="${APPIMAGETOOL:-}"
if [[ -z "${APPIMAGETOOL_BIN}" ]]; then
    if command -v appimagetool >/dev/null 2>&1; then
        APPIMAGETOOL_BIN="$(command -v appimagetool)"
    else
        CACHED_TOOL="${BUILD_ROOT}/appimagetool-${APPIMAGE_ARCH}.AppImage"
        if [[ ! -x "${CACHED_TOOL}" ]]; then
            TOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${APPIMAGE_ARCH}.AppImage"
            echo "==> Downloading appimagetool (${APPIMAGE_ARCH})..."
            curl -fsSL "${TOOL_URL}" -o "${CACHED_TOOL}"
            chmod +x "${CACHED_TOOL}"
        fi
        APPIMAGETOOL_BIN="${CACHED_TOOL}"
    fi
fi

OUTPUT_APPIMAGE="${DIST_ROOT}/neurowhisper-linux-${APPIMAGE_ARCH}.AppImage"
rm -f "${OUTPUT_APPIMAGE}"

echo "==> Compressing SquashFS AppImage -> ${OUTPUT_APPIMAGE}"
ARCH="${APPIMAGE_ARCH}" APPIMAGE_EXTRACT_AND_RUN=1 \
    "${APPIMAGETOOL_BIN}" "${APPDIR}" "${OUTPUT_APPIMAGE}"
chmod +x "${OUTPUT_APPIMAGE}"

(
    cd "${DIST_ROOT}"
    sha256sum "$(basename "${OUTPUT_APPIMAGE}")" > "$(basename "${OUTPUT_APPIMAGE}").sha256"
)

echo "==> Done! Standalone Linux AppImage ready:"
ls -lh "${OUTPUT_APPIMAGE}" "${OUTPUT_APPIMAGE}.sha256"
