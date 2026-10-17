; Inno Setup script for Image Inspect.
;
; Wraps a PyInstaller one-folder build into a single setup executable, in
; one of two variants that install the same app:
;
;   cpu   ImageInspect-<version>-Setup.exe      runs the models on the CPU
;   cuda  ImageInspect-<version>-GPU-Setup.exe  adds NVIDIA CUDA, ~1.6 GB more
;
; Both share an AppId, so running either over an existing install upgrades
; it or switches it to the other variant. packaging\build.ps1 builds the app
; and runs this with /DAppVersion=... /DVariant=cpu|cuda.

#ifndef AppVersion
  #error Pass /DAppVersion=x.y.z (packaging\build.ps1 does)
#endif
#ifndef Variant
  #define Variant "cpu"
#endif

#define AppName "Image Inspect"
#define AppPublisher "Chris Millsap"
#define AppExeName "ImageInspect.exe"
#define SourceDir "..\dist\" + Variant + "\ImageInspect"
#if Variant == "cuda"
  #define SetupSuffix "-GPU-Setup"
#else
  #define SetupSuffix "-Setup"
#endif

[Setup]
; Never reuse this GUID in another product; it is how Windows recognises an
; existing install and offers to upgrade it in place.
AppId={{3B8C1E52-9F4D-4A67-B0C2-7E1D5A93F6C8}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/cmillsap/ImageInspect
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\ImageInspect
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}

; Per-user by default, so the common case installs with no UAC prompt at all.
; A user who wants it for everyone can still choose that on the first page.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline

Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
#if Variant == "cuda"
; Compress in a separate 64-bit process, which has the memory for a solid
; ~2 GB payload.
LZMAUseSeparateProcess=yes
#endif

ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

WizardStyle=modern
SetupIconFile=icon.ico
OutputDir=..\dist
OutputBaseFilename=ImageInspect-{#AppVersion}{#SetupSuffix}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Clear out the previous build's libraries first, so switching from the GPU
; variant to the CPU one does not leave 1.6 GB of CUDA behind.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; \
    Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; \
    Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
var
  Version: TWindowsVersion;
begin
  Result := True;
  GetWindowsVersionEx(Version);
  // PyQt6 and onnxruntime are built against the Windows 10 SDK; older
  // releases fail at DLL load with an error the user cannot act on.
  if Version.Major < 10 then
  begin
    MsgBox('{#AppName} requires Windows 10 or later.', mbCriticalError, MB_OK);
    Result := False;
    Exit;
  end;
#if Variant == "cuda"
  // nvcuda.dll comes with the NVIDIA driver. Without it this variant only
  // runs on the CPU, which the much smaller standard installer also does.
  if (not WizardSilent) and
     (not FileExists(ExpandConstant('{sys}\nvcuda.dll'))) then
    Result := MsgBox('No NVIDIA graphics driver was found, so this GPU ' +
      'version of {#AppName} would run on the processor only.' + #13#10#13#10 +
      'The standard installer (ImageInspect-{#AppVersion}-Setup.exe) does ' +
      'the same and is much smaller. Install the GPU version anyway?',
      mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
#endif
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  // The search index and the downloaded models (up to several GB) live
  // outside the install folder; offer to remove them, keeping them by
  // default since re-indexing a photo library takes a while.
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\ImageInspect');
    if DirExists(DataDir) and (not UninstallSilent) and
       (MsgBox('Also delete the search index and downloaded models in ' +
               DataDir + '?' + #13#10#13#10 +
               'Keep them if you will reinstall: re-indexing your photos ' +
               'and downloading the models again takes a while.',
               mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES) then
    begin
      DelTree(DataDir, True, True, True);
      RegDeleteKeyIncludingSubkeys(HKCU, 'Software\ImageInspect');
    end;
  end;
end;
