@echo off
title RetailIQ - AI Retail Intelligence Platform V3.0
color 0A

echo.
echo  =====================================================
echo    RetailIQ - AI Retail Intelligence Platform V3.0
echo    Starting development server...
echo  =====================================================
echo.

:: Change to webapp directory
cd /d "d:\AI RETAIL INTELLIGENCE\webapp"

:: Check if node_modules exists
if not exist "node_modules\" (
    echo  [!] node_modules not found. Installing dependencies...
    echo.
    call npm install
    echo.
)

:: Kill any existing process on port 5173
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5173"') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: Start Vite in background and open browser after delay
echo  [OK] Starting Vite dev server on http://localhost:5173
echo  [OK] Opening browser in 3 seconds...
echo.
echo  Press Ctrl+C to stop the server.
echo.

:: Open browser after 3 second delay (background)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5173"

:: Start the dev server (foreground — keeps window open)
call npm run dev
