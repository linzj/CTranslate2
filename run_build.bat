@echo OFF
SETLOCAL

REM This script builds the CTranslate2 project with only the DirectML backend.

set PATH=C:\Program Files\LLVM\bin;%PATH%
set CC=clang-cl
set CXX=clang-cl
REM Configure the project using CMake.
echo "Configuring project with CMake..."
cmake . -B build -G "Ninja" -DWITH_DIRECTML=ON -DOPENMP_RUNTIME=NONE -DWITH_MKL=OFF -DBUILD_CLI=ON -DBUILD_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

IF %ERRORLEVEL% NEQ 0 (
    echo "CMake configuration failed."
    goto:eof
)

REM Build the project.
echo "Building project..."
cmake --build build --config Release

IF %ERRORLEVEL% NEQ 0 (
    echo "Build failed."
    goto:eof
)

echo "Build successful! The binaries are in the build/cli/Release directory."

ENDLOCAL