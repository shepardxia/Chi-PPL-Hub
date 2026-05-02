"""
Scrape probmods2 exercises + solutions into structured JSONL.

Each exercise file in `exercises/X.md` is paired with the solution
file `solutions/X.md` when present. Both are parsed with the same
fence/section logic used for chapter scraping; pairing is an
annotation, not a structural merge. Records mirror the chapter
schema with added `solution_file` and `solution_sections` fields.

Usage:
    python scripts/scrape_probmods_exercises.py
"""

import json
import re
from pathlib import Path
from typing import Optional

EXERCISES_DIR = Path("data/sources/probmods2/exercises")
SOLUTIONS_DIR = Path("data/sources/probmods2/solutions")
OUTPUT_FILE = Path("data/raw/probmods_exercises.jsonl")


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

    Handles ~~~~, ~~~, ```` and ``` fences (probmods mixes them).
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

        if current_type != "prose":
            flush()
            current_lines = []
            current_type = "prose"
        current_lines.append(line)

    flush()
    return sections


def scrape_exercise(ex_path: Path, sol_path: Optional[Path]) -> dict:
    """Parse an exercise file and pair with its solution if present."""
    ex_text = ex_path.read_text(encoding="utf-8")
    ex_frontmatter, ex_body = parse_frontmatter(ex_text)
    ex_sections = parse_sections(ex_body)

    record = {
        "id": f"probmods2-exercises/{ex_path.stem}",
        "source": "probmods2-exercises",
        "source_file": str(ex_path),
        "frontmatter": ex_frontmatter,
        "title": ex_frontmatter.get("title", ex_path.stem),
        "language_version": "",
        "category": "exercise",
        "tags": [],
        "sections": ex_sections,
        "code_block_count": sum(1 for s in ex_sections if s["type"] == "code"),
        "solution_file": None,
        "solution_sections": None,
        "solution_frontmatter": None,
        "solution_code_block_count": 0,
    }

    if sol_path is not None and sol_path.exists():
        sol_text = sol_path.read_text(encoding="utf-8")
        sol_frontmatter, sol_body = parse_frontmatter(sol_text)
        sol_sections = parse_sections(sol_body)
        record["solution_file"] = str(sol_path)
        record["solution_sections"] = sol_sections
        record["solution_frontmatter"] = sol_frontmatter
        record["solution_code_block_count"] = sum(
            1 for s in sol_sections if s["type"] == "code"
        )

    return record


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not EXERCISES_DIR.exists():
        print(f"ERROR: {EXERCISES_DIR} not found.")
        return

    exercise_files = sorted(EXERCISES_DIR.glob("*.md"))
    print(f"Found {len(exercise_files)} exercise files")

    records = []
    unpaired = []
    for ex_path in exercise_files:
        sol_path = SOLUTIONS_DIR / ex_path.name
        record = scrape_exercise(ex_path, sol_path if sol_path.exists() else None)
        records.append(record)
        if record["solution_file"] is None:
            unpaired.append(ex_path.name)

    with open(OUTPUT_FILE, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    total_ex_blocks = sum(r["code_block_count"] for r in records)
    total_sol_blocks = sum(r["solution_code_block_count"] for r in records)
    print(f"\nResults:")
    print(f"  Exercise files: {len(records)}")
    print(f"  Exercise code blocks: {total_ex_blocks}")
    print(f"  Solution code blocks: {total_sol_blocks}")
    print(f"  Unpaired (no solution): {len(unpaired)}")
    for u in unpaired:
        print(f"    {u}")
    print(f"  Output: {OUTPUT_FILE}")

    for r in records:
        paired = "paired" if r["solution_file"] else "UNPAIRED"
        print(
            f"  {r['title']}: "
            f"{r['code_block_count']} ex / "
            f"{r['solution_code_block_count']} sol ({paired})"
        )


if __name__ == "__main__":
    main()
