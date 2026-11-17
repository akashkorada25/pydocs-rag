"""Turn the Python documentation's reStructuredText files into clean, citable chunks.

Each chunk keeps the page it came from, its section heading and a link to that section on
docs.python.org, so answers can cite their sources precisely.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

DOCS_BASE = "https://docs.python.org/3"

HEADING_CHARS = set("=-~*^#\"'`+:.")

# Directives whose whole block is noise for question answering
DROP_DIRECTIVES = {
    "index", "highlight", "sectionauthor", "moduleauthor", "testsetup", "testcleanup", "toctree",
    "only", "raw", "include", "currentmodule", "module", "contents", "tabularcolumns", "cssclass",
    "rst-class", "csv-table", "image", "figure", "productionlist", "literalinclude", "impl-detail",
}
# Directives whose content we keep, with a readable label in front
LABELLED_DIRECTIVES = {
    "note": "Note:", "warning": "Warning:", "seealso": "See also:", "tip": "Tip:", "important": "Important:",
    "caution": "Caution:", "versionadded": "New in version", "versionchanged": "Changed in version",
    "deprecated": "Deprecated since version", "attention": "Attention:", "hint": "Hint:", "danger": "Danger:",
}

ROLE_RE = re.compile(r":[\w:+.-]+:`([^`]*?)`")
LINK_RE = re.compile(r"`([^`<]+?)\s*<[^>]+>`__?\\?")
LITERAL_RE = re.compile(r"``(.+?)``")
FOOTNOTE_RE = re.compile(r"\s*\[(?:#\w*|\d+|\*)\]_")
LABEL_RE = re.compile(r"^\.\. _([\w.-]+):\s*$")
DIRECTIVE_RE = re.compile(r"^(\s*)\.\. ([\w-]+)::\s*(.*)$")


@dataclass
class Chunk:
    id: str
    doc: str          # e.g. "tutorial/controlflow"
    doc_title: str    # e.g. "More Control Flow Tools"
    section: str      # innermost heading, e.g. "Default Argument Values"
    anchor: str       # e.g. "tut-defaultargs" ("" if the section has no label)
    url: str
    text: str

    @property
    def source(self) -> str:
        return f"{self.doc}#{self.section}"

    @property
    def search_text(self) -> str:
        """What retrievers index: the page and section titles give each chunk its context."""
        return f"{self.doc_title} - {self.section}\n{self.text}"

    def to_dict(self):
        return asdict(self)


def slugify(title: str) -> str:
    """The id Sphinx gives a section heading (docutils make_id), used as the URL anchor."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return re.sub(r"^[0-9-]+", "", slug)


def clean_inline(text: str) -> str:
    def role(m):
        inner = m.group(1)
        inner = re.sub(r"\s*<[^>]*>$", "", inner) if "<" in inner and inner.rstrip().endswith(">") else inner
        return inner.lstrip("!~")

    text = ROLE_RE.sub(role, text)
    text = LINK_RE.sub(r"\1", text)
    text = LITERAL_RE.sub(r"`\1`", text)
    text = FOOTNOTE_RE.sub("", text)
    text = text.replace("\\ ", "")
    return text


def _is_underline(line: str, title: str) -> bool:
    s = line.rstrip()
    return (len(s) >= 3 and len(set(s)) == 1 and s[0] in HEADING_CHARS
            and len(s) >= len(title.strip()) - 2 and not line.startswith(" "))


def parse_sections(text: str) -> list[dict]:
    """Split a reST document into sections: [{title, level, anchor, lines}]."""
    lines = text.splitlines()
    styles: list[tuple[str, bool]] = []  # heading style order defines levels
    sections = [{"title": "", "level": 0, "anchor": "", "lines": []}]
    pending_label = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        m = LABEL_RE.match(line)
        if m:
            pending_label = m.group(1)
            i += 1
            continue
        # Heading with overline: ===\nTitle\n===
        if (_is_underline(line, "") and i + 2 < len(lines) and lines[i + 1].strip()
                and lines[i + 2].rstrip() == line.rstrip() and not lines[i + 1].startswith("  ")):
            style = (line[0], True)
            title = lines[i + 1].strip()
            i += 3
        elif line.strip() and not line.startswith((" ", "\t", "..")) and nxt and _is_underline(nxt, line):
            style = (nxt[0], False)
            title = line.strip()
            i += 2
        else:
            sections[-1]["lines"].append(line)
            if line.strip():
                pending_label = pending_label if line.startswith("..") else ""
            i += 1
            continue
        if style not in styles:
            styles.append(style)
        sections.append({"title": clean_inline(title), "level": styles.index(style) + 1,
                         "anchor": pending_label, "lines": []})
        pending_label = ""
    return sections


def clean_body(lines: list[str]) -> str:
    out: list[str] = []
    skip_indent: int | None = None
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if skip_indent is not None:
            if not line.strip() or indent > skip_indent:
                continue
            skip_indent = None
        if LABEL_RE.match(line.strip()):
            continue
        m = DIRECTIVE_RE.match(line)
        if m:
            name, arg = m.group(2), m.group(3).strip()
            if name in DROP_DIRECTIVES:
                skip_indent = indent
                continue
            if name in LABELLED_DIRECTIVES:
                out.append(" " * indent + f"{LABELLED_DIRECTIVES[name]} {clean_inline(arg)}".rstrip())
            # code-block, doctest, function, class, etc.: keep the body only
            continue
        if line.strip().startswith(".. ") or line.strip() == "..":
            # reST comment: skip it and its indented block
            skip_indent = indent
            continue
        if line.rstrip().endswith("::"):
            stripped = line.rstrip()[:-2].rstrip()
            line = (stripped + ":") if stripped else ""
        out.append(line.rstrip())
    text = clean_inline("\n".join(out))  # whole text, so roles and links that wrap lines are cleaned
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    """Pack paragraphs into chunks of up to max_words, carrying a short overlap between chunks."""
    paragraphs = [p.strip("\n") for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current, count = [], [], 0
    for para in paragraphs:
        n = len(para.split())
        if current and count + n > max_words:
            chunks.append("\n\n".join(current))
            tail = current[-1]
            current, count = ([tail], len(tail.split())) if len(tail.split()) <= overlap_words else ([], 0)
        if n > max_words * 1.5:  # very long paragraph: split by words
            words = para.split()
            step = max_words - overlap_words
            for start in range(0, len(words), step):
                piece = " ".join(words[start:start + max_words])
                chunks.append("\n\n".join(current + [piece]) if current else piece)
                current, count = [], 0
            continue
        current.append(para)
        count += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def ingest_file(path: Path, root: Path, max_words: int = 220) -> list[Chunk]:
    doc = path.relative_to(root).with_suffix("").as_posix()
    sections = parse_sections(path.read_text(encoding="utf-8"))
    titled = [s for s in sections if s["title"]]
    doc_title = titled[0]["title"] if titled else doc
    chunks: list[Chunk] = []
    for sec in sections:
        body = clean_body(sec["lines"])
        if len(body.split()) < 8:  # headings with no real content
            continue
        section = sec["title"] or doc_title
        anchor = sec["anchor"] or (slugify(sec["title"]) if sec["level"] > 1 else "")
        url = f"{DOCS_BASE}/{doc}.html" + (f"#{anchor}" if anchor else "")
        for piece in chunk_text(body, max_words=max_words):
            chunks.append(Chunk(f"{doc}#{len(chunks)}", doc, doc_title, section, anchor, url, piece))
    return chunks


def ingest(raw_dir: str | Path, out_path: str | Path, max_words: int = 220) -> list[Chunk]:
    raw_dir, out_path = Path(raw_dir), Path(out_path)
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.rglob("*.rst")):
        if path.name == "index.rst":
            continue
        chunks.extend(ingest_file(path, raw_dir, max_words))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return chunks


def load_chunks(path: str | Path) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
