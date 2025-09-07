#!/usr/bin/env python3
"""
Install dependencies for hybrid retrieval system
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages for hybrid retrieval"""
    
    print("🔧 Installing dependencies for hybrid retrieval system...")
    
    packages = [
        "rank-bm25",
        "nltk",
        "sentence-transformers",  # Ensure latest version
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"   Successful: {success_count}/{len(packages)}")
    
    if success_count == len(packages):
        print("✅ All dependencies installed successfully!")
        
        # Download NLTK data
        try:
            import nltk
            print("📥 Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            print("✅ NLTK data downloaded successfully!")
        except Exception as e:
            print(f"⚠️ NLTK data download failed: {e}")
    else:
        print("❌ Some dependencies failed to install")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
