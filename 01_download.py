"""Download the two PRISM files at the pinned revision and verify their checksums.

    python -I 01_download.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from src.data import sha256  # noqa: E402


def main():
    config.RAW.mkdir(parents=True, exist_ok=True)
    for name, digest in config.FILES.items():
        path = config.RAW / name
        if not path.exists() or sha256(path) != digest:
            url = f"https://huggingface.co/datasets/{config.HF_DATASET}/resolve/{config.HF_REVISION}/{name}"
            print("downloading", url)
            urllib.request.urlretrieve(url, path)
        ok = sha256(path) == digest
        print(f"{name}: {'ok' if ok else 'CHECKSUM MISMATCH'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
