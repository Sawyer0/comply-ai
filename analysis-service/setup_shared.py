#!/usr/bin/env python3
"""
Setup script to configure shared components integration for analysis service.

This script ensures the shared directory is properly configured and accessible
to the analysis service.
"""

import sys
import os
from pathlib import Path


def setup_shared_integration():
    """Setup shared components integration."""

    # Get the shared directory path
    current_dir = Path(__file__).parent
    shared_dir = current_dir.parent / "shared"

    print(f"Setting up shared integration for analysis service...")
    print(f"Current directory: {current_dir}")
    print(f"Shared directory: {shared_dir}")

    # Check if shared directory exists
    if not shared_dir.exists():
        print(f"ERROR: Shared directory not found at {shared_dir}")
        print("Please ensure the shared directory exists at the correct location.")
        return False

    # Check if shared directory has the expected structure
    expected_dirs = ["interfaces", "utils", "exceptions", "models", "database"]
    missing_dirs = []

    for expected_dir in expected_dirs:
        if not (shared_dir / expected_dir).exists():
            missing_dirs.append(expected_dir)

    if missing_dirs:
        print(f"WARNING: Missing expected directories in shared: {missing_dirs}")

    # Add shared directory to Python path in a .pth file
    site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
    if site_packages.exists():
        pth_file = site_packages / "analysis-shared.pth"
        try:
            with open(pth_file, "w") as f:
                f.write(str(shared_dir.absolute()) + "\n")
            print(f"Created .pth file: {pth_file}")
        except Exception as e:
            print(f"WARNING: Could not create .pth file: {e}")

    # Create a symlink to shared directory (alternative approach)
    analysis_src = current_dir / "src" / "analysis"
    shared_link = analysis_src / "shared_lib"

    try:
        if shared_link.exists():
            shared_link.unlink()

        # Create symlink (Windows requires admin privileges, so we'll copy instead)
        if os.name == "nt":  # Windows
            import shutil

            if shared_link.exists():
                shutil.rmtree(shared_link)
            shutil.copytree(shared_dir, shared_link)
            print(f"Copied shared directory to: {shared_link}")
        else:  # Unix-like systems
            shared_link.symlink_to(shared_dir.absolute())
            print(f"Created symlink: {shared_link} -> {shared_dir}")

    except Exception as e:
        print(f"WARNING: Could not create shared link: {e}")

    print("Analysis service shared integration setup complete!")
    return True


if __name__ == "__main__":
    success = setup_shared_integration()
    sys.exit(0 if success else 1)
