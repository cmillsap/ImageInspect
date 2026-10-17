# -*- mode: python ; coding: utf-8 -*-
r"""PyInstaller spec for Image Inspect.

Builds a one-folder distribution, which the Inno Setup script wraps into
the installer. One-folder rather than one-file: a one-file build would
unpack hundreds of MB (gigabytes for the GPU build) to temp on every launch.

The same spec builds both variants; which one depends on the onnxruntime
installed in the venv running it. With onnxruntime-gpu, the CUDA and cuDNN
libraries from the nvidia-* packages are bundled too, keeping their
site-packages layout (_internal\nvidia\...), which is where
onnxruntime.preload_dlls() looks for them next to the onnxruntime package.

packaging\build.ps1 sets up the build venvs and runs this:

    build\venv-cpu\Scripts\python -m PyInstaller packaging\ImageInspect.spec
"""

import glob
import os

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files
import importlib.util

PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))
SITE_PACKAGES = os.path.dirname(os.path.dirname(
    importlib.util.find_spec("onnxruntime").origin))
CUDA = os.path.isdir(os.path.join(SITE_PACKAGES, "nvidia"))
print(f"Building the {'GPU (CUDA)' if CUDA else 'CPU'} variant")

datas = [(os.path.join(SPECPATH, "icon.ico"), "packaging"),
         (os.path.join(PROJECT_ROOT, "README.md"), ".")]
binaries = collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("onnxruntime_genai")
datas += collect_data_files("onnxruntime_genai")

if CUDA:
    # Only the run-time DLLs: the packages' headers and import libraries
    # are for compiling against CUDA. Their licences come along.
    for pattern in ("nvidia/cu13/bin/x86_64/*.dll", "nvidia/cudnn/bin/*.dll"):
        for path in glob.glob(os.path.join(SITE_PACKAGES, pattern)):
            binaries.append((path, os.path.dirname(os.path.relpath(path, SITE_PACKAGES))))
    for path in glob.glob(os.path.join(SITE_PACKAGES, "nvidia_*.dist-info", "licenses", "*")) + \
            glob.glob(os.path.join(SITE_PACKAGES, "nvidia_*.dist-info", "License.txt")):
        package = os.path.basename(path.split(".dist-info")[0])
        datas.append((path, os.path.join("licenses", package)))

# The version Windows shows in the exe's properties, from version.py.
VERSION = {}
exec(open(os.path.join(PROJECT_ROOT, "version.py"), encoding="utf-8").read(), VERSION)
VERSION = VERSION["__version__"]
numbers = tuple(int(n) for n in VERSION.split(".")) + (0,)
VERSION_FILE = os.path.join(workpath, "version_info.txt")
os.makedirs(workpath, exist_ok=True)
with open(VERSION_FILE, "w", encoding="utf-8") as f:
    f.write(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={numbers}, prodvers={numbers}),
  kids=[StringFileInfo([StringTable('040904B0', [
    StringStruct('CompanyName', 'Chris Millsap'),
    StringStruct('FileDescription', 'Image Inspect'),
    StringStruct('FileVersion', '{VERSION}'),
    StringStruct('InternalName', 'ImageInspect'),
    StringStruct('OriginalFilename', 'ImageInspect.exe'),
    StringStruct('ProductName', 'Image Inspect'),
    StringStruct('ProductVersion', '{VERSION}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])])
""")

a = Analysis(
    [os.path.join(PROJECT_ROOT, "app.py")],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Optional imports that would otherwise drag in other Qt bindings, the
    # test stack, and the tokenizers library's training extras.
    excludes=[
        "PyQt5", "PySide2", "PySide6", "tkinter",
        "matplotlib", "pytest", "_pytest", "pandas", "IPython",
        "torch", "tensorflow", "huggingface_hub", "transformers",
    ],
    noarchive=False,
    optimize=0,
)

# TensorRT's provider needs TensorRT itself, which is never installed.
a.binaries = [b for b in a.binaries
              if "onnxruntime_providers_tensorrt" not in b[0]]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ImageInspect",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SPECPATH, "icon.ico"),
    version=VERSION_FILE,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ImageInspect",
)
