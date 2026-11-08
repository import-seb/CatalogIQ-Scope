"""Run the candidate relationship rules in warn mode; nothing here is approved."""
from catalogiq.cli import relationship_conflicts_main

if __name__ == "__main__":
    raise SystemExit(relationship_conflicts_main())
