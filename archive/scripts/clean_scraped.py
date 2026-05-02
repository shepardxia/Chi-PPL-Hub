"""
Clean scraped JSONL data before building the eval dataset.

Pipeline:
  data/raw/*.jsonl (lossless) --> clean_scraped.py --> data/cleaned/*.jsonl

Transformations applied:
1. Re-parse from source files with HTML comments stripped
   (author TODOs, draft content, and old Church code that's not rendered
   to readers but survives in source).
2. Detect which runtime dep packages each record uses, based on
   symbols that appear in code blocks.

Block-level classification (category / dataset-fit) is a separate
downstream stage, not part of cleaning.

Usage:
    python scripts/clean_scraped.py
"""

import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")


# ---------------------------------------------------------------------------
# Shared parsing (same logic as scrapers, so cleaning is self-contained)
# ---------------------------------------------------------------------------

def parse_frontmatter(text):
    """Extract YAML frontmatter and remaining body from markdown."""
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


_WEBPPL_FENCE_LANGS = {"", "js", "javascript", "webppl", "wppl"}


def _looks_like_json_output(content):
    """Heuristic: content is JSON-style data output, not WebPPL code.

    Detects the common probmods2 pattern of a ` ``` ` fence wrapping an
    example return value dump (e.g., `{"mus":[...]}`). Must start with `{`
    or `[` and have no WebPPL-like keywords.
    """
    stripped = content.strip()
    if not stripped:
        return False
    if stripped[0] not in "{[":
        return False
    return not re.search(r"\b(var|function|return|Infer|flip|sample|condition|factor|mem|repeat|map|filter)\b", stripped)


def parse_sections_fenced(body):
    """Parse markdown body into ordered prose/code sections.

    Fence semantics in probmods2:
      - `~~~~` (or `~~~`) is the book's WebPPL convention — always code.
      - ` ```lang ` is code only when lang ∈ {js, javascript, webppl, wppl, ""}.
        Triple-backtick with an output tag like `json` or `text` (or the
        common case of a bare ` ``` ` wrapping an example output) is prose,
        since it's used in the book for result samples / data dumps, not
        for runnable code.
      - Closing fence must match the opening marker exactly.
    """
    sections = []
    current_type = "prose"
    current_lines = []

    def flush():
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                # Demote backtick-fenced blocks that look like JSON/data
                # output back to prose (probmods2 convention for result dumps).
                final_type = current_type
                if final_type == "code" and _looks_like_json_output(content):
                    final_type = "prose"
                sections.append({"type": final_type, "content": content})

    fence_open = re.compile(r"^(~~~~|~~~|````|```)(\w*)")

    in_fenced = False
    open_marker = None

    for line in body.split("\n"):
        stripped = line.lstrip()
        if not in_fenced:
            m = fence_open.match(stripped)
            if m:
                marker, lang = m.group(1), m.group(2).lower()
                flush()
                current_lines = []
                in_fenced = True
                open_marker = marker
                # Tilde fences are always code in probmods2. Backtick fences
                # are code only when the language tag is js/webppl or empty
                # followed by recognized WebPPL patterns.
                if marker.startswith("~"):
                    current_type = "code"
                elif lang in _WEBPPL_FENCE_LANGS:
                    current_type = "code"
                else:
                    current_type = "prose"
                continue
        else:
            if stripped.startswith(open_marker) and stripped.rstrip() == open_marker:
                flush()
                current_lines = []
                current_type = "prose"
                in_fenced = False
                open_marker = None
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


def parse_sections_indented(body):
    """Parse markdown body with indented code blocks (ForestDB style).

    Also handles fenced code blocks as fallback.
    """
    sections = []
    current_type = None
    current_lines = []

    def flush():
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({"type": current_type, "content": content})

    in_fenced = False
    fence_marker = None

    for line in body.split("\n"):
        if not in_fenced:
            fence_match = re.match(r"^(````|~~~~|```|~~~)", line)
            if fence_match:
                flush()
                current_lines = []
                current_type = "code"
                in_fenced = True
                fence_marker = fence_match.group(1)
                continue
        else:
            if line.strip().startswith(fence_marker):
                flush()
                current_lines = []
                current_type = None
                in_fenced = False
                fence_marker = None
                continue
            current_lines.append(line)
            continue

        is_code_line = line.startswith("\t") or line.startswith("    ")
        is_blank = line.strip() == ""

        if is_code_line:
            if line.startswith("\t"):
                stripped = line[1:]
            else:
                stripped = line[4:]
            if current_type != "code":
                flush()
                current_lines = []
                current_type = "code"
            current_lines.append(stripped)
        elif is_blank:
            current_lines.append("")
        else:
            if current_type != "prose":
                flush()
                current_lines = []
                current_type = "prose"
            current_lines.append(line)

    flush()
    return sections


# ---------------------------------------------------------------------------
# Transformation 1: Strip HTML comments (operates on raw text)
# ---------------------------------------------------------------------------

def strip_html_comments(text):
    """Remove HTML comment blocks (<!-- ... -->) from raw markdown text.

    These contain author TODOs, old Church code, and draft content
    invisible to readers in the rendered book.
    """
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Transformation 2: Detect dependencies
# ---------------------------------------------------------------------------

# Map from symbols found in code to the dep package that provides them.
# These are loadable via `webppl --require <dep-path>`.
SYMBOL_TO_DEP = {
    "towData": "probmods-towdata",
    "towMeans": "probmods-towdata",
    "towConfigurations": "probmods-towdata",
    "MCMC_Callbacks": "probmods-deps",
    "euclideanDistance": "probmods-deps",
    "timeit": "probmods-deps",
    # Physics: box2d-backed sandbox (eval/deps/probmods-physics)
    "physics.run": "probmods-physics",
    "physics.animate": "probmods-physics",
    "worldWidth": "probmods-physics",
    "worldHeight": "probmods-physics",
    # Draw: headless canvas shim (eval/deps/probmods-draw).
    # Keyed on "Draw(" to avoid matching uniformDraw.
    "Draw(": "probmods-draw",
}


def detect_deps(sections):
    """Detect which dep packages a record needs based on symbols in code blocks.

    Returns a set of dep package names.
    """
    deps = set()
    for section in sections:
        if section["type"] == "code":
            for symbol, dep in SYMBOL_TO_DEP.items():
                if symbol in section["content"]:
                    deps.add(dep)
    return sorted(deps)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Source-specific parse strategy
PARSE_STRATEGY = {
    "forestdb": parse_sections_indented,
    "probmods2": parse_sections_fenced,
    "probmods2-exercises": parse_sections_fenced,
    "probmods2-teaching-extras": parse_sections_fenced,
    "probmods-examples": None,  # single code block, no re-parsing needed
}


def _reparse_file(path, parse_fn):
    text = Path(path).read_text(encoding="utf-8")
    _, body = parse_frontmatter(text)
    body = strip_html_comments(body)
    return parse_fn(body)


def clean_record(record):
    """Apply all transformations to a single record.

    Re-reads the source file(s) with HTML comments stripped and applies
    the source's parse strategy. Exercise records also re-parse their
    paired solution file into `solution_sections`.
    """
    source = record.get("source", "")
    parse_fn = PARSE_STRATEGY.get(source)
    source_file = record.get("source_file", "")

    if parse_fn and source_file and Path(source_file).exists():
        sections = _reparse_file(source_file, parse_fn)
    else:
        sections = record.get("sections", [])

    record = dict(record)
    record["sections"] = sections
    record["code_block_count"] = sum(1 for s in sections if s["type"] == "code")

    # Exercise records: also re-parse the solution file.
    solution_file = record.get("solution_file")
    if parse_fn and solution_file and Path(solution_file).exists():
        record["solution_sections"] = _reparse_file(solution_file, parse_fn)
        record["solution_code_block_count"] = sum(
            1 for s in record["solution_sections"] if s["type"] == "code"
        )

    # Deps look at BOTH exercise-side and solution-side code, since the
    # runtime env must satisfy whichever code we'll actually run.
    combined = list(sections) + list(record.get("solution_sections") or [])
    record["deps"] = detect_deps(combined)
    return record


def main():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    for jsonl_file in sorted(RAW_DIR.glob("*.jsonl")):
        print(f"Cleaning {jsonl_file.name}...")

        records = []
        with open(jsonl_file) as f:
            for line in f:
                records.append(json.loads(line))

        original_code_blocks = sum(
            sum(1 for s in r["sections"] if s["type"] == "code")
            for r in records
        )

        cleaned = [clean_record(r) for r in records]
        cleaned = [r for r in cleaned if r["code_block_count"] > 0]

        cleaned_code_blocks = sum(r["code_block_count"] for r in cleaned)
        removed = original_code_blocks - cleaned_code_blocks

        out_path = CLEANED_DIR / jsonl_file.name
        with open(out_path, "w") as f:
            for record in cleaned:
                f.write(json.dumps(record) + "\n")

        print(f"  Records: {len(records)} -> {len(cleaned)}")
        print(f"  Code blocks: {original_code_blocks} -> {cleaned_code_blocks} (-{removed})")
        print(f"  Output: {out_path}")
        print()


if __name__ == "__main__":
    main()
