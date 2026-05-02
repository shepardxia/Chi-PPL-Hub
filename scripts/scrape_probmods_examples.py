"""
Scrape probmods/webppl example .wppl files into structured JSONL.

These are standalone WebPPL programs with no metadata —
just raw code. Source files are the lossless layer.

Usage:
    python scripts/scrape_probmods_examples.py
"""

import json
from pathlib import Path

SOURCES_DIR = Path("data/sources/webppl/examples")
OUTPUT_FILE = Path("data/raw/probmods_examples.jsonl")


def scrape_example(filepath: Path) -> dict:
    """Parse a single .wppl example file."""
    code = filepath.read_text(encoding="utf-8").strip()

    # Extract leading comments as description (if any)
    comment_lines = []
    for line in code.split("\n"):
        if line.strip().startswith("//"):
            comment_lines.append(line.strip().lstrip("/").strip())
        elif line.strip() == "":
            if comment_lines:
                comment_lines.append("")
        else:
            break

    return {
        "id": f"probmods-examples/{filepath.stem}",
        "source": "probmods-examples",
        "source_file": str(filepath),
        "frontmatter": {},
        "title": filepath.stem,
        "language_version": "",
        "category": "",
        "tags": [],
        "sections": [
            {"type": "code", "content": code},
        ],
        "leading_comments": "\n".join(comment_lines).strip() if comment_lines else "",
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not SOURCES_DIR.exists():
        print(f"ERROR: {SOURCES_DIR} not found. Clone webppl first:")
        print("  git clone https://github.com/probmods/webppl.git data/sources/webppl")
        return

    wppl_files = sorted(SOURCES_DIR.glob("*.wppl"))
    print(f"Found {len(wppl_files)} .wppl example files")

    records = []
    for filepath in wppl_files:
        record = scrape_example(filepath)
        records.append(record)

    with open(OUTPUT_FILE, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nResults:")
    print(f"  Examples extracted: {len(records)}")
    print(f"  Output: {OUTPUT_FILE}")

    for r in records:
        code_len = len(r["sections"][0]["content"].split("\n"))
        print(f"  {r['title']}: {code_len} lines")


if __name__ == "__main__":
    main()
