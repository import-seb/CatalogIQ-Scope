"""Freeze/preflight the final assessment; test execution requires explicit authorization."""
from catalogiq.cli import protected_test_main

if __name__ == "__main__":
    raise SystemExit(protected_test_main())
