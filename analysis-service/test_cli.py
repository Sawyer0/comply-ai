#!/usr/bin/env python3
"""
Simple test script to verify CLI commands work.
"""

import subprocess
import sys


def test_cli_help():
    """Test that CLI help works."""
    try:
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print("✓ CLI help command works")
            print(
                result.stdout[:200] + "..."
                if len(result.stdout) > 200
                else result.stdout
            )
            return True
        else:
            print("✗ CLI help command failed")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False


def test_cli_commands():
    """Test that CLI command groups work."""
    commands = ["tenant", "plugin", "analyze", "service"]

    for command in commands:
        try:
            result = subprocess.run(
                [sys.executable, "cli.py", command, "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                print(f"✓ CLI {command} command group works")
            else:
                print(f"✗ CLI {command} command group failed")
                print(result.stderr)
                return False

        except Exception as e:
            print(f"✗ CLI {command} test failed: {e}")
            return False

    return True


if __name__ == "__main__":
    print("Testing Analysis Service CLI...")
    print("-" * 40)

    success = True
    success &= test_cli_help()
    success &= test_cli_commands()

    print("-" * 40)
    if success:
        print("✓ All CLI tests passed!")
    else:
        print("✗ Some CLI tests failed!")
        sys.exit(1)
