from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as: python scripts/build_index.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policyhub.indexing import build_or_load_index
from policyhub.llm import configure_llm
from policyhub.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/refresh LvZhengTong index")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear the Chroma collection and rebuild embeddings from RAW_DOCS_DIR",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_llm(settings)

    _ = build_or_load_index(settings, rebuild=bool(args.rebuild))
    print("OK: index built/loaded.")


if __name__ == "__main__":
    main()
