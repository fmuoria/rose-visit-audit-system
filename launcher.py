# launcher.py - Easy launcher for ROSE Audit System
"""
Launcher script for ROSE Audit System
Checks Python version & dependencies, then launches the GUI application.
Also generates cross-platform launcher scripts.
"""

import sys
import subprocess
import importlib.util
import os
import stat

# ----------------------------
# 1. Check Python version
# ----------------------------
def check_python_version():
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"Current version: {sys.version}")
        input("Press Enter to exit...")
        sys.exit(1)
    else:
        print(f"✅ Python version {sys.version.split()[0]} is compatible")

# ----------------------------
# 2. Check & install dependencies
# ----------------------------
def check_and_install_dependencies():
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'geopy': 'geopy',
        'sklearn': 'scikit-learn',
        'textstat': 'textstat',
        'openpyxl': 'openpyxl'
    }

    missing_packages = []

    print("🔍 Checking dependencies...")
    for import_name, package_name in required_packages.items():
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                missing_packages.append(package_name)
                print(f"❌ {package_name} not found")
            else:
                print(f"✅ {package_name} is installed")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name} not found")

    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please install manually using: pip install -r requirements.txt")
            input("Press Enter to exit...")
            sys.exit(1)
    else:
        print("✅ All dependencies are already installed!")

# ----------------------------
# 3. Launch GUI
# ----------------------------
def launch_gui():
    gui_path = os.path.join("src", "gui_application.py")

    if not os.path.exists(gui_path):
        print(f"❌ GUI application not found at {gui_path}")
        print("Please ensure you're running this from the project root directory")
        input("Press Enter to exit...")
        sys.exit(1)

    print("🚀 Launching ROSE Audit System GUI...")

    try:
        sys.path.insert(0, "src")
        import gui_application
        gui_application.main()
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

# ----------------------------
# 4. Create cross-platform launchers
# ----------------------------
def create_launcher_files():
    batch_content = '''
@echo off
title ROSE Audit System Launcher
echo.
echo ========================================
echo    ROSE Visit Audit System Launcher
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Starting ROSE Audit System...
python launcher.py
pause >nul
'''

    shell_content = '''#!/bin/bash
echo "========================================"
echo "   ROSE Visit Audit System Launcher"
echo "========================================"
echo

if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed"
        read -p "Press Enter to exit..."
        exit 1
    else
        PYTHON_CMD=python
    fi
else
    PYTHON_CMD=python3
fi

echo "Starting ROSE Audit System..."
$PYTHON_CMD launcher.py
read -p "Press Enter to exit..."
'''

    with open("launcher.bat", "w") as f:
        f.write(batch_content)
    print("✅ Created launcher.bat for Windows")

    with open("launcher.sh", "w") as f:
        f.write(shell_content)
    os.chmod("launcher.sh", os.stat("launcher.sh").st_mode | stat.S_IEXEC)
    print("✅ Created launcher.sh for macOS/Linux")

# ----------------------------
# 5. Main entry point
# ----------------------------
def main():
    print("=" * 60)
    print("🌹 ROSE Women Leaders Visit Audit System")
    print("=" * 60)
    print()

    check_python_version()
    print()

    check_and_install_dependencies()
    print()

    launch_gui()

if __name__ == "__main__":
    # If you just want to run GUI:
    main()

    # If you also want to regenerate .bat/.sh launchers automatically:
    # create_launcher_files()
