; basketball-clip Inno Setup 安装脚本（E-2）
; 编译：由 packaging/build_installer.py 调 ISCC（/D 注入版本与源目录），产物 packaging/dist/
; 手动编译：ISCC.exe installer.iss（在 packaging/ 目录下，需 dist/basketball-clip/ 已由打包段产出）

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\basketball-clip"
#endif

[Setup]
AppId={{9BB9938A-2C05-4D73-9960-29D93EDDFB35}
AppName=basketball-clip
AppVersion={#AppVersion}
AppPublisher=huangli
; 默认装到用户目录，免管理员权限
DefaultDirName={localappdata}\Programs\basketball-clip
DefaultGroupName=basketball-clip
PrivilegesRequired=lowest
; dialog 会在静默安装启动时弹「安装模式」对话框挂起进程，改 commandline（保留 /ALLUSERS /CURRENTUSER 覆盖能力）
PrivilegesRequiredOverridesAllowed=commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=dist
OutputBaseFilename=basketball-clip-setup-{#AppVersion}
SetupIconFile=icon.ico
UninstallDisplayName=basketball-clip
UninstallDisplayIcon={app}\icon.ico
LicenseFile=INSTALLER_LICENSE.txt
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ShowLanguageDialog=no
; 向导首页被跳过的语言选择也不需要：仅简体中文
DisableWelcomePage=no

[Languages]
Name: "chs"; MessagesFile: "lang\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面图标"; GroupDescription: "附加任务:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\basketball-clip"; Filename: "{app}\basketball-clip.exe"; IconFilename: "{app}\icon.ico"
Name: "{group}\卸载 basketball-clip"; Filename: "{uninstallexe}"
Name: "{autodesktop}\basketball-clip"; Filename: "{app}\basketball-clip.exe"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\basketball-clip.exe"; Description: "启动 basketball-clip"; Flags: nowait postinstall skipifsilent unchecked

[Code]
// 卸载收尾：用户数据（work\ output\ photos\ 等运行时目录）询问保留，不静默删
// SuppressibleMsgBox：/VERYSILENT（自带 /SUPPRESSMSGBOXES）下被抑制取 IDNO=保留数据，
// 避免 plain MsgBox 在静默卸载时弹窗无人应答导致卸载进程永久挂起（E-2 审查实测复现）
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDir: String;
  HasData: Boolean;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    AppDir := ExpandConstant('{app}');
    HasData := DirExists(AppDir + '\work') or DirExists(AppDir + '\output') or
               DirExists(AppDir + '\photos');
    if HasData then
    begin
      if SuppressibleMsgBox('卸载已完成。检测到安装目录下还有用户数据（work\、output\、photos\ 等，'
                + '含检测中间产物与集锦成品）。' + #13#10#13#10
                + '是否一并删除这些数据？选择「否」将保留。', mbConfirmation, MB_YESNO, IDNO) = IDYES then
      begin
        DelTree(AppDir + '\work', True, True, True);
        DelTree(AppDir + '\output', True, True, True);
        DelTree(AppDir + '\photos', True, True, True);
      end;
    end;
  end;
end;
