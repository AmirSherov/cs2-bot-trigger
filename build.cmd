@echo off
setlocal
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
if not exist build mkdir build
cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 /external:I"sdk\include" /external:I"sdk\tensorrt" /external:I"sdk\cuda" /external:W0 main.cpp /Fo"build\main.obj" /Fe"gpu-trigger.exe" onnxruntime.lib d3d11.lib d3d12.lib d3dcompiler.lib dxgi.lib windowscodecs.lib ole32.lib user32.lib
exit /b %errorlevel%
