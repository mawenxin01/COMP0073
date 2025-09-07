#!/usr/bin/env python3
"""
Medical Triage Assistant - Test Runner
======================================

Comprehensive test runner with different test categories and reporting options.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests  
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --parallel         # Run tests in parallel
    python run_tests.py --verbose          # Verbose output
    python run_tests.py --fast             # Skip slow tests
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    if description:
        print(f"🚀 {description}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description or 'Command'} completed successfully")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description or 'Command'} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Medical Triage Assistant tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                           # Run all tests
  python run_tests.py --unit --verbose          # Run unit tests with verbose output
  python run_tests.py --model-integration       # Run only model integration tests
  python run_tests.py --coverage --html         # Run with HTML coverage report
  python run_tests.py --parallel --fast         # Fast parallel execution
  python run_tests.py --integration --slow      # Include slow integration tests
        """
    )
    
    # Test selection options
    parser.add_argument('--unit', action='store_true', 
                       help='Run only unit tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--api', action='store_true',
                       help='Run only API tests')
    parser.add_argument('--model', action='store_true',
                       help='Run only model tests')
    parser.add_argument('--preprocessing', action='store_true',
                       help='Run only preprocessing tests')
    parser.add_argument('--feature-extraction', action='store_true',
                       help='Run only feature extraction tests')
    parser.add_argument('--model-integration', action='store_true',
                       help='Run only model integration tests')
    
    # Execution options
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output')
    parser.add_argument('--fast', action='store_true',
                       help='Skip slow tests')
    parser.add_argument('--slow', action='store_true',
                       help='Include slow tests')
    
    # Coverage options
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--html', action='store_true',
                       help='Generate HTML coverage report')
    parser.add_argument('--xml', action='store_true',
                       help='Generate XML coverage report')
    
    # Output options
    parser.add_argument('--junit', action='store_true',
                       help='Generate JUnit XML report')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark tests')
    
    # Specific test options
    parser.add_argument('--test-file', type=str,
                       help='Run specific test file')
    parser.add_argument('--test-function', type=str,
                       help='Run specific test function')
    parser.add_argument('--keywords', '-k', type=str,
                       help='Run tests matching keywords')
    
    # Environment options
    parser.add_argument('--install-deps', action='store_true',
                       help='Install test dependencies first')
    parser.add_argument('--check-env', action='store_true',
                       help='Check test environment')
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(project_root)
    
    # Install dependencies if requested
    if args.install_deps:
        print("📦 Installing test dependencies...")
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'tests/requirements.txt']
        if not run_command(cmd, "Installing test dependencies"):
            return 1
    
    # Check environment if requested
    if args.check_env:
        check_test_environment()
    
    # Build pytest command
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add test directory
    cmd.append('tests/')
    
    # Test selection
    markers = []
    if args.unit:
        markers.append('unit')
    if args.integration:
        markers.append('integration')
    if args.api:
        markers.append('api')
    if args.model:
        markers.append('model')
    if args.preprocessing:
        markers.append('preprocessing')
    if args.feature_extraction:
        markers.append('feature_extraction')
    if args.model_integration:
        # Run tests specifically from test_model_integration.py
        cmd = [sys.executable, '-m', 'pytest', 'tests/integration/test_model_integration.py']
        # Skip the normal marker processing for model_integration
        model_integration_mode = True
    else:
        model_integration_mode = False
    
    if markers and not model_integration_mode:
        cmd.extend(['-m', ' or '.join(markers)])
    
    # Speed options
    if args.fast:
        cmd.extend(['-m', 'not slow'])
    elif args.slow:
        cmd.extend(['-m', 'slow'])
    
    # Verbosity
    if args.verbose:
        cmd.append('-vv')
    elif args.quiet:
        cmd.append('-q')
    else:
        cmd.append('-v')
    
    # Parallel execution
    if args.parallel:
        cmd.extend(['-n', 'auto'])
    
    # Coverage options
    if args.coverage or args.html or args.xml:
        cmd.extend(['--cov=medical_triage_assistant'])
        cmd.append('--cov-report=term-missing')
        
        if args.html:
            cmd.append('--cov-report=html:tests/coverage_html')
        if args.xml:
            cmd.append('--cov-report=xml:tests/coverage.xml')
    
    # Output formats
    if args.junit:
        cmd.append('--junit-xml=tests/junit.xml')
    
    if args.benchmark:
        cmd.append('--benchmark-only')
    
    # Specific test selection
    if args.test_file:
        cmd.append(f'tests/{args.test_file}')
    
    if args.test_function:
        cmd.extend(['-k', args.test_function])
    
    if args.keywords:
        cmd.extend(['-k', args.keywords])
    
    # Additional pytest options
    cmd.extend([
        '--tb=short',
        '--disable-warnings'
    ])
    
    # Only use strict-markers if not in model_integration mode
    if not model_integration_mode:
        cmd.append('--strict-markers')
    
    # Run the tests
    success = run_command(cmd, "Running tests")
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
        if args.coverage or args.html:
            print("📊 Coverage report generated")
            if args.html:
                print("   HTML report: tests/coverage_html/index.html")
    else:
        print("❌ Some tests failed")
        print("💡 Check the output above for details")
    print(f"{'='*60}")
    
    return 0 if success else 1


def check_test_environment():
    """Check if test environment is properly set up"""
    print("\n🔍 Checking test environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("⚠️  Python 3.8+ recommended for testing")
    
    # Check required packages
    required_packages = [
        'pytest', 'pandas', 'numpy', 'scikit-learn', 'flask'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    # Check test directories
    test_dirs = ['tests/unit', 'tests/integration', 'tests/fixtures']
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"✅ {test_dir} - exists")
        else:
            print(f"❌ {test_dir} - missing")
    
    print("🔍 Environment check completed\n")


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
