#!/usr/bin/env python3
"""Build lossless memory-mapped session storage from processed events."""

from __future__ import annotations

import argparse
import json

from ktbench.data.session_store import build_session_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_root")
    parser.add_argument("output_root")
    args = parser.parse_args()
    metadata = build_session_store(args.processed_root, args.output_root)
    print(json.dumps(metadata.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
