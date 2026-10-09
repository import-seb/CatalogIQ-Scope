"""Run: python -m scripts.split_data --input <cleaned CSV> --output-dir <new directory>."""
from catalogiq.cli import split_data_main as main


if __name__ == "__main__":
    raise SystemExit(main())
