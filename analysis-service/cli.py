#!/usr/bin/env python3
"""
Analysis Service CLI entry point.

Usage:
    python cli.py --help
    python cli.py tenant create test-tenant --name "Test Tenant"
    python cli.py plugin list
    python cli.py analyze run --content "test content" --tenant-id test-tenant
    python cli.py service health
"""

from src.analysis.cli import cli

if __name__ == "__main__":
    cli()
