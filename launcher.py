# launcher.py - Easy launcher for ROSE Audit System
"""
Simple launcher script for ROSE Audit System
This script handles dependencies and launches the GUI application
"""

import sys
import subprocess
import importlib.util
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"Current version: {sys.version}")
        print("Please install Python 3.7+ from https://python.org")
        input("Press Enter to exit...")
        sys.exit(1)
    else:
        print(f"✅ Python version {sys.version.split()[0]} is compatible")

def check_and_install_dependencies():
    """Check and install required dependencies"""
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
            # Install missing packages
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

def launch_gui():
    """Launch the GUI application"""
    gui_path = os.path.join("src", "gui_application.py")
    
    if not os.path.exists(gui_path):
        print(f"❌ GUI application not found at {gui_path}")
        print("Please ensure you're running this from the project root directory")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("🚀 Launching ROSE Audit System GUI...")
    
    try:
        # Add src directory to Python path
        sys.path.insert(0, "src")
        
        # Import and run the GUI
        import gui_application
        gui_application.main()
        
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        print("Please check the error message above and try again")
        input("Press Enter to exit...")
        sys.exit(1)

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🌹 ROSE Women Leaders Visit Audit System")
    print("=" * 60)
    print()
    
    # Check Python version
    check_python_version()
    print()
    
    # Check and install dependencies
    check_and_install_dependencies()
    print()
    
    # Launch GUI
    launch_gui()

if __name__ == "__main__":
    main()

# =============================================================================
# Windows Batch File Launcher (launcher.bat)
# Save this content in a file named 'launcher.bat'
# =============================================================================

batch_content = '''
@echo off
title ROSE Audit System Launcher
echo.
echo ========================================
echo    ROSE Visit Audit System Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    echo.
    pause
    exit /b 1
)

echo Starting ROSE Audit System...
echo.

REM Run the launcher
python launcher.py

echo.
echo Press any key to exit...
pause >nul
'''

# =============================================================================
# macOS/Linux Shell Script Launcher (launcher.sh)  
# Save this content in a file named 'launcher.sh'
# =============================================================================

shell_content = '''#!/bin/bash

echo "========================================"
echo "   ROSE Visit Audit System Launcher"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed"
        echo "Please install Python from https://python.org"
        echo
        read -p "Press Enter to exit..."
        exit 1
    else
        PYTHON_CMD=python
    fi
else
    PYTHON_CMD=python3
fi

echo "Starting ROSE Audit System..."
echo

# Run the launcher
$PYTHON_CMD launcher.py

echo
echo "Press Enter to exit..."
read
'''

if __name__ == "__main__":
    # This section creates the launcher files when run directly
    print("Creating launcher files...")
    
    # Create batch file for Windows
    with open("launcher.bat", "w") as f:
        f.write(batch_content)
    print("✅ Created launcher.bat for Windows")
    
    # Create shell script for macOS/Linux  
    with open("launcher.sh", "w") as f:
        f.write(shell_content)
    
    # Make shell script executable
    import stat
    os.chmod("launcher.sh", os.stat("launcher.sh").st_mode | stat.S_IEXEC)
    print("✅ Created launcher.sh for macOS/Linux")
    
    print("\nLauncher files created successfully!")
    print("Users can now run:")
    print("- Windows: Double-click launcher.bat")
    print("- macOS/Linux: ./launcher.sh")
