#!/usr/bin/env python3
"""
Quick Example: Extract Greenspace from Yokohama Satellite Imagery
"""

import os
import sys

def main():
    """Run a simple greenspace extraction example"""
    
    print("=" * 60)
    print("Greenspace Extraction Demo - Yokohama, Japan")
    print("Using GSI Seamless Photo Tiles + U-Net")
    print("=" * 60)
    
    # Check if running the simple or full version
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        print("\nRunning full version with TorchGeo...")
        os.system("python greenspace_extraction.py")
    else:
        print("\nRunning simplified version...")
        print("(Use --full flag for complete TorchGeo integration)")
        os.system("python greenspace_simple.py")

if __name__ == "__main__":
    main()
