@echo off
setlocal
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
if not exist build mkdir build
cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 pack.cpp /Fo"build\pack.obj" /Fe"build\pack.exe" cabinet.lib
if errorlevel 1 exit /b %errorlevel%
powershell.exe -NoProfile -File "%~dp0make-payload.ps1"
if errorlevel 1 exit /b %errorlevel%
rc /nologo /fo "build\payload.res" "build\payload.rc"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 launcher.cpp "build\payload.res" /Fo"build\launcher.obj" /Fe"..\gpu-trigger-standalone.exe" cabinet.lib bcrypt.lib shell32.lib ole32.lib
exit /b %errorlevel%
