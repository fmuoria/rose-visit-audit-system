# launcher.py - Easy launcher for ROSE Audit System
"""
Launcher script for ROSE Audit System
- Checks Python version and dependencies
- Runs the VisitAuditSystem audit
- Launches the GUI application
"""

import sys
import subprocess
import importlib.util
import os
import pandas as pd

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

def run_audit():
    """Run the VisitAuditSystem audit logic before GUI"""
    try:
        sys.path.insert(0, "src")
        from audit_system import VisitAuditSystem

        # Load sample dataset (replace with your real visits file if needed)
        data_path = os.path.join("data", "visits.xlsx")
        if os.path.exists(data_path):
            df = pd.read_excel(data_path)
            audit = VisitAuditSystem(df)
            results = audit.audit_location_similarity()

            print("\n📊 AUDIT RESULTS")
            if not results:
                print("✅ No suspicious locations detected")
            else:
                for trainer_result in results:
                    print(f"\nTrainer: {trainer_result['trainer']}")
                    print(f"- Total visits: {trainer_result['total_visits']}")
                    print(f"- Flagged pairs: {trainer_result['flagged_pairs']}")
                    print(f"- Out of country: {trainer_result['flagged_out_of_country']}")
        else:
            print("⚠️ No dataset found at data/visits.xlsx, skipping audit run")
    
    except Exception as e:
        print(f"❌ Audit run failed: {e}")

def launch_gui():
    """Launch the GUI application"""
    gui_path = os.path.join("src", "gui_application.py")
    
    if not os.path.exists(gui_path):
        print(f"❌ GUI application not found at {gui_path}")
        print("Please ensure you're running this from the project root directory")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("\n🚀 Launching ROSE Audit System GUI...")
    
    try:
        sys.path.insert(0, "src")
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
    
    check_python_version()
    print()
    
    check_and_install_dependencies()
    print()
    
    run_audit()   # 🔹 Run your audit first
    print()
    
    launch_gui()  # 🔹 Then launch the GUI

if __name__ == "__main__":
    main()
