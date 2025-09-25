#!/usr/bin/env python3
"""
Mapper Service CLI Entry Point

This script provides the main entry point for the Mapper Service CLI.
It can be used directly or installed as a console script.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from mapper.cli import cli

if __name__ == "__main__":
    cli()
