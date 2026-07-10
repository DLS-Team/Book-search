from __future__ import annotations

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="en", help="Dataset split/language, default: en")
    parser.add_argument("--max-books", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=Path("data/raw/project_gutenberg_books.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Streaming manu/project_gutenberg...")
    print(f"Split: {args.split}")
    print(f"Max books: {args.max_books}")
    print(f"Output: {args.output}")

    ds = load_dataset("manu/project_gutenberg", split=args.split, streaming=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, total=args.max_books):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.max_books:
                break

    print(f"Saved books: {count}")


if __name__ == "__main__":
    main()