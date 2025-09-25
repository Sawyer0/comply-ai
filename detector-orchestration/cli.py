#!/usr/bin/env python3
"""CLI entry point for orchestration service.

This script provides a convenient entry point for the orchestration CLI.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestration.cli.main import main

if __name__ == "__main__":
    asyncio.run(main())
