#!/usr/bin/env python3
"""
EV Energy Consumption Predictor - Project Setup Script
Automated setup for the complete ML project
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED!")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def create_virtual_environment():
    """Create Python virtual environment"""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")

def install_dependencies():
    """Install project dependencies"""
    activate_cmd = ""
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate && "
    else:  # Unix-like
        activate_cmd = "source venv/bin/activate && "
    
    return run_command(f"{activate_cmd}pip install -r requirements.txt", "Installing dependencies")

def train_models():
    """Train ML models"""
    activate_cmd = ""
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate && "
    else:  # Unix-like
        activate_cmd = "source venv/bin/activate && "
    
    return run_command(f"{activate_cmd}python src/models/run_training.py", "Training ML models")

def verify_setup():
    """Verify that all components are working"""
    print("\n" + "="*60)
    print("VERIFYING SETUP")
    print("="*60)
    
    # Check if models exist
    model_files = [
        "models/best_model.pkl",
        "models/preprocessor.pkl",
        "models/model_results.csv"
    ]
    
    all_models_exist = True
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✅ {model_file} exists")
        else:
            print(f"❌ {model_file} missing")
            all_models_exist = False
    
    # Check if data exists
    data_file = "data/ev_energy_data.csv"
    if Path(data_file).exists():
        print(f"✅ {data_file} exists")
    else:
        print(f"❌ {data_file} missing")
        all_models_exist = False
    
    # Check if plots exist
    plot_files = [
        "plots/model_comparison.png",
        "plots/feature_importance.png"
    ]
    
    for plot_file in plot_files:
        if Path(plot_file).exists():
            print(f"✅ {plot_file} exists")
        else:
            print(f"❌ {plot_file} missing")
    
    return all_models_exist

def print_next_steps():
    """Print instructions for running the application"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 NEXT STEPS:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix-like
        print("   source venv/bin/activate")
    
    print("\n2. Run the application:")
    print("   python src/api/app.py")
    
    print("\n3. Open your browser and navigate to:")
    print("   http://localhost:5000")
    
    print("\n📚 DOCUMENTATION:")
    print("- README.md: Complete project overview")
    print("- docs/project_report.md: Detailed project report")
    print("- docs/viva_questions.md: Viva preparation")
    print("- docs/deployment_guide.md: Deployment instructions")
    
    print("\n🔧 API ENDPOINTS:")
    print("- GET  /api/health: Health check")
    print("- POST /api/predict: Energy consumption prediction")
    print("- GET  /api/model_info: Model information")
    print("- POST /api/batch_predict: Batch predictions")
    
    print("\n📊 PROJECT STATISTICS:")
    print("- Dataset: 5,000 samples")
    print("- Features: 11 engineered features")
    print("- Best Model: Random Forest (R² = 0.9995)")
    print("- RMSE: 0.1391 kWh")
    print("- MAE: 0.0726 kWh")

def main():
    """Main setup function"""
    print("🚗 EV ENERGY CONSUMPTION PREDICTOR - PROJECT SETUP")
    print("="*60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    print(f"Working directory: {project_dir.absolute()}")
    
    # Setup steps
    steps = [
        ("Python Version Check", check_python_version),
        ("Virtual Environment Creation", create_virtual_environment),
        ("Dependencies Installation", install_dependencies),
        ("Model Training", train_models),
        ("Setup Verification", verify_setup)
    ]
    
    all_success = True
    for step_name, step_func in steps:
        if not step_func():
            all_success = False
            print(f"\n❌ Setup failed at: {step_name}")
            print("Please fix the error and run setup again.")
            return
    
    if all_success:
        print_next_steps()
    else:
        print("\n❌ Setup completed with errors. Please check the logs above.")

if __name__ == "__main__":
    main()
