"""Compatibility entry point; implementation lives in the catalogiq package."""
import argparse
import json
from catalogiq.feature_decisions import *  # noqa: F403


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', choices=['training', 'target'], required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.dataset), indent=2))
