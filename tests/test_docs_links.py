"""Every local markdown link resolves -- files, directories and in-page
anchors.

Exists because broken doc links were found by hand in two separate
audits (a link to a nonexistent "Section 5.4", and five notebooks
described in a README that had never been written). Re-deriving that
check each time is wasted work; as a test it runs on every pytest.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
HEADING = re.compile(r"^#+\s+(.*)$", re.M)
MARKDOWN_FILES = [*sorted((REPO / "docs").glob("*.md")), REPO / "README.md"]


def _slug(heading: str) -> str:
    """GitHub's heading-to-anchor slug: lowercase, drop formatting and
    punctuation, spaces to hyphens."""
    s = re.sub(r"[`*_]", "", heading.strip().lower())
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s+", "-", s).strip("-")


def _local_links(text: str):
    for m in LINK.finditer(text):
        target = m.group(2).strip()
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        yield text[: m.start()].count("\n") + 1, target


@pytest.mark.parametrize("md", MARKDOWN_FILES, ids=lambda p: p.name)
def test_local_markdown_links_resolve(md):
    text = md.read_text(encoding="utf-8")
    own_anchors = {_slug(h) for h in HEADING.findall(text)}
    broken = []
    for line, target in _local_links(text):
        if target.startswith("#"):
            if _slug(target[1:]) not in own_anchors:
                broken.append(f"{md.name}:{line} anchor not found: {target}")
            continue
        path, _, frag = target.partition("#")
        resolved = (md.parent / path).resolve()
        if not resolved.exists():
            broken.append(f"{md.name}:{line} missing path: {target}")
        elif frag and resolved.suffix == ".md":
            other = {_slug(h) for h in HEADING.findall(resolved.read_text(encoding="utf-8"))}
            if _slug(frag) not in other:
                broken.append(f"{md.name}:{line} anchor not in {path}: #{frag}")
    assert not broken, "\n".join(broken)


def test_the_check_actually_sees_links():
    # Guard against the parametrized test passing vacuously because the
    # link regex or the file glob silently stopped matching anything.
    assert len(MARKDOWN_FILES) > 5
    total = sum(len(list(_local_links(md.read_text(encoding="utf-8")))) for md in MARKDOWN_FILES)
    assert total > 15, total
