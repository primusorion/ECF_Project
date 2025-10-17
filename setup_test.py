"""
Quick Setup and Test Script
Verifies installation and runs a quick test
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False


def check_dependencies():
    """Check if dependencies are installed"""
    print("\nChecking dependencies...")
    
    required = [
        'numpy',
        'pandas',
        'sklearn',
        'yaml',
        'plotly'
    ]
    
    missing = []
    
    for package in required:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"✓ {package} - OK")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def check_directory_structure():
    """Check if required directories exist"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        'data/synthetic',
        'data/raw',
        'data/processed',
        'models/saved_models',
        'logs',
        'config'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} - OK")
        else:
            print(f"✗ {dir_path} - MISSING (will be created)")
            os.makedirs(dir_path, exist_ok=True)
    
    print("\n✓ Directory structure ready")
    return True


def run_quick_test():
    """Run a quick functionality test"""
    print("\n" + "="*60)
    print("Running Quick Test...")
    print("="*60)
    
    try:
        # Test data generation
        print("\n1. Testing data generation...")
        sys.path.append('src')
        from src.data_generation.data_generator import EquipmentDataGenerator
        
        generator = EquipmentDataGenerator(seed=42)
        test_data = generator.generate_normal_operation('Motor', n_samples=10)
        print(f"   ✓ Generated {len(test_data)} test samples")
        
        # Test preprocessing
        print("\n2. Testing preprocessing...")
        from src.preprocessing.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(test_data)
        print(f"   ✓ Preprocessed {len(clean_data)} samples")
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_next_steps():
    """Print next steps for user"""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
Your Predictive Maintenance System is ready!

Quick Start Commands:

1. Run the complete pipeline:
   python main.py --mode full

2. Generate data only (10,000 samples):
   python main.py --mode generate --samples 10000

3. Train models:
   python main.py --mode train

4. Launch monitoring dashboard:
   python main.py --mode monitor

5. Run interactive demo:
   python notebooks/demo_script.py

Documentation:
- README.md - Full project documentation
- GETTING_STARTED.md - Detailed setup guide
- PROJECT_OVERVIEW.md - System architecture

Need help? Check the logs:
- logs/predictive_maintenance.log
""")
    print("="*60)


def main():
    """Main setup verification"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     PREDICTIVE MAINTENANCE SYSTEM - SETUP VERIFICATION           ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    all_ok = True
    
    # Run checks
    if not check_python_version():
        all_ok = False
    
    if not check_dependencies():
        all_ok = False
        print("\n⚠ Please install dependencies first:")
        print("   pip install -r requirements.txt")
        return
    
    if not check_directory_structure():
        all_ok = False
    
    if all_ok:
        print("\n✓ Setup verification complete!")
        
        # Run quick test
        run_test = input("\nRun quick functionality test? (y/n): ").lower().strip()
        
        if run_test == 'y':
            if run_quick_test():
                print_next_steps()
            else:
                print("\n⚠ Some tests failed. Check error messages above.")
        else:
            print_next_steps()
    else:
        print("\n⚠ Setup incomplete. Please resolve issues above.")


if __name__ == '__main__':
    main()
