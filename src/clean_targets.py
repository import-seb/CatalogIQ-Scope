"""Compatibility entry point; use python -m catalogiq.cleaning for new work."""
from .catalogiq.cleaning import (
    CHANGE_COLUMNS, FLAG_COLUMNS, PROVENANCE, REASONS, TARGET_COLUMNS,
    CleaningResult, check_single_parent_counts, clean_training, iqr_bounds,
    load_source, main, run, sha256, write_csv,
)

if __name__ == "__main__":
    main()
