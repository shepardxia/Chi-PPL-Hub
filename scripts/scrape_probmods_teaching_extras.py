"""
Scrape probmods2 teaching_extras into structured JSONL.

teaching_extras/ holds advanced-topic standalone demos (Bayesian Spelke,
Causal Abstraction, Extended Tug of War, RSA variants, etc). Structurally
they're the same as chapters: markdown with ~~~~ code fences and a YAML
frontmatter. Several files are pure-prose write-ups with no WebPPL — those
are skipped.

Usage:
    python scripts/scrape_probmods_teaching_extras.py
"""

import json
from pathlib import Path

from scrape_probmods_chapters import parse_frontmatter, parse_sections

SOURCES_DIR = Path("data/sources/probmods2/teaching_extras")
OUTPUT_FILE = Path("data/raw/probmods_teaching_extras.jsonl")


def scrape_file(filepath: Path) -> dict:
    text = filepath.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)
    sections = parse_sections(body)
    code_sections = [s for s in sections if s["type"] == "code"]
    return {
        "id": f"probmods2-teaching-extras/{filepath.stem}",
        "source": "probmods2",
        "source_file": str(filepath),
        "frontmatter": frontmatter,
        "title": frontmatter.get("title", filepath.stem),
        "language_version": "",
        "category": "teaching_extra",
        "tags": [],
        "sections": sections,
        "code_block_count": len(code_sections),
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCES_DIR.exists():
        print(f"ERROR: {SOURCES_DIR} not found.")
        return

    files = sorted(SOURCES_DIR.glob("*.md"))
    print(f"Found {len(files)} teaching_extras files")

    records = []
    skipped = []
    for filepath in files:
        record = scrape_file(filepath)
        if record["code_block_count"] > 0:
            records.append(record)
        else:
            skipped.append(filepath.name)

    with open(OUTPUT_FILE, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    total = sum(r["code_block_count"] for r in records)
    print(f"\nResults:")
    print(f"  With code:    {len(records)}")
    print(f"  Skipped (no code): {skipped}")
    print(f"  Total code blocks: {total}")
    print(f"  Output: {OUTPUT_FILE}")
    for r in records:
        print(f"  {r['title']}: {r['code_block_count']} code blocks")


if __name__ == "__main__":
    main()
