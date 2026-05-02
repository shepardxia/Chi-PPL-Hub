"""
Scrape probmods2 tutorial chapters into structured JSONL.

Chapters are markdown with ~~~~-delimited WebPPL code blocks
interleaved with tutorial prose. Preserves full document structure.

Usage:
    python scripts/scrape_probmods_chapters.py
"""

import json
import re
from pathlib import Path

SOURCES_DIR = Path("data/sources/probmods2/chapters")
OUTPUT_FILE = Path("data/raw/probmods_chapters.jsonl")


def parse_frontmatter(text: str):
    """Extract YAML frontmatter and remaining body."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text

    yaml_str, body = match.group(1), match.group(2)

    frontmatter = {}
    for line in yaml_str.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            value = value.strip().strip("'\"")
            frontmatter[key.strip()] = value

    return frontmatter, body


def parse_sections(body: str) -> list[dict]:
    """Parse markdown body into ordered prose/code sections.

    probmods2 uses ~~~~ as code fence delimiters for WebPPL blocks.
    Also handles ``` fences and HTML blocks.
    """
    sections = []
    current_type = "prose"
    current_lines = []

    def flush():
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({"type": current_type, "content": content})

    in_fenced = False

    for line in body.split("\n"):
        # Detect code fence boundaries (~~~~ or ```)
        if not in_fenced:
            if re.match(r"^(````|~~~~|```|~~~)", line.strip()):
                flush()
                current_lines = []
                current_type = "code"
                in_fenced = True
                continue
        else:
            if re.match(r"^(````|~~~~|```|~~~)", line.strip()):
                flush()
                current_lines = []
                current_type = "prose"
                in_fenced = False
                continue
            current_lines.append(line)
            continue

        # Everything else is prose
        if current_type != "prose":
            flush()
            current_lines = []
            current_type = "prose"
        current_lines.append(line)

    flush()
    return sections


def scrape_chapter(filepath: Path) -> dict:
    """Parse a single probmods2 chapter."""
    text = filepath.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)
    sections = parse_sections(body)

    code_sections = [s for s in sections if s["type"] == "code"]

    return {
        "id": f"probmods2/{filepath.stem}",
        "source": "probmods2",
        "source_file": str(filepath),
        "frontmatter": frontmatter,
        "title": frontmatter.get("title", filepath.stem),
        "language_version": "",
        "category": "tutorial",
        "tags": [],
        "sections": sections,
        "code_block_count": len(code_sections),
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not SOURCES_DIR.exists():
        print(f"ERROR: {SOURCES_DIR} not found. Clone probmods2 first:")
        print("  git clone https://github.com/probmods/probmods2.git data/sources/probmods2")
        return

    chapter_files = sorted(SOURCES_DIR.glob("*.md"))
    print(f"Found {len(chapter_files)} chapter files")

    records = []
    for filepath in chapter_files:
        record = scrape_chapter(filepath)
        # Only include chapters that have at least one code block
        if record["code_block_count"] > 0:
            records.append(record)
        else:
            print(f"  Skipping {filepath.name} (no code blocks)")

    with open(OUTPUT_FILE, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    total_code_blocks = sum(r["code_block_count"] for r in records)
    print(f"\nResults:")
    print(f"  Chapters with code: {len(records)}")
    print(f"  Total code blocks:  {total_code_blocks}")
    print(f"  Output: {OUTPUT_FILE}")

    for r in records:
        print(f"  {r['title']}: {r['code_block_count']} code blocks")


if __name__ == "__main__":
    main()
