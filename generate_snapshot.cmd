@echo off
setlocal enabledelayedexpansion

REM Configure output file and remove if it exists
set "output_file=project_snapshot.txt"
if exist "%output_file%" del "%output_file%"

REM Get the script directory (always ends with a backslash)
set "script_dir=%~dp0"

REM Define file names to ignore (case-insensitive)
set "script_name=generate_snapshot.cmd"
set "output_name=project_snapshot.txt"

REM Generate filtered directory structure (excluding ignored files)
echo === PROJECT DIRECTORY STRUCTURE === >> "%output_file%"
tree /F /A | findstr /v /i "%script_name%" | findstr /v /i "%output_name%" >> "%output_file%"
echo. >> "%output_file%"
echo. >> "%output_file%"

REM Count total files (excluding ignored ones)
set file_count=0
for /r %%f in (*) do (
    if /i not "%%~nxf"=="%script_name%" if /i not "%%~nxf"=="%output_name%" (
        set /a file_count+=1
    )
)

REM Initialize processing variables
set /a processed=0
set /a bar_width=50

REM Obtain a carriage return character (CR) without extra output
for /F "delims=" %%a in ('"prompt $H & for %%b in (1) do rem"') do set "CR=%%a"

REM Process files with progress display
echo === FILE CONTENTS === >> "%output_file%"
for /r %%f in (*) do (
    if /i not "%%~nxf"=="%script_name%" if /i not "%%~nxf"=="%output_name%" (
        set /a processed+=1
        set /a progress=processed*100/file_count
        set /a hashes=progress*bar_width/100

        REM Build progress bar (using '#' for completed portion)
        set "bar="
        for /l %%i in (1,1,!hashes!) do (
            set "bar=!bar!#"
        )
        set /a spaces_count=bar_width - hashes
        set "spaces="
        for /l %%i in (1,1,!spaces_count!) do (
            set "spaces=!spaces! "
        )
        
        REM Clear screen and update progress display for smooth updates
        cls
        echo Progress: [!bar!!spaces!] !progress!%%  Processing: !processed!/!file_count! files

        REM Calculate relative path: Remove the script directory from the full path.
        set "rel=%%f"
        set "rel=!rel:%script_dir%=!"

        REM Append file header and content to output file using the relative path
        >> "%output_file%" echo.
        >> "%output_file%" echo ========================================
        >> "%output_file%" echo === File: !rel!
        >> "%output_file%" echo ========================================
        type "%%f" 2>nul >> "%output_file%"
        >> "%output_file%" echo.
        >> "%output_file%" echo.
    )
)

REM Finalize progress bar update with a newline
cls
echo Progress: [##################################################] 100%%  Processing: !processed!/!file_count! files
echo.
echo Project snapshot saved to %output_file%
pause
