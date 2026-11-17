import re

from ragassist.app import DEFAULT_CHUNKS
from ragassist.ingest import DOCS_BASE, chunk_text, clean_body, clean_inline, load_chunks, parse_sections, slugify

SAMPLE = """\
.. _tut-top:

*****************
More Control Flow
*****************

Intro text about :keyword:`while` loops and more words here to keep it.

.. _tut-defaults:

Default Argument Values
=======================

The default value is evaluated only once. See :func:`len <len>` and
`the docs <https://example.com/x>`_ for more.

.. index::
   single: hidden entry

.. note::
   Important note text.

Example::

   def f(a, L=[]):
       return L

Keyword-Only Arguments
----------------------

Use a bare ``*`` in the parameter list.
"""


def test_slugify_matches_sphinx_ids():
    assert slugify("Keyword-Only Arguments") == "keyword-only-arguments"
    assert slugify("Why does -22 // 10 return -3?") == "why-does-22-10-return-3"
    assert slugify("3. Numbers") == "numbers"


def test_clean_inline_roles_links_literals():
    assert clean_inline(":func:`len`") == "len"
    assert clean_inline(":ref:`Packages <tut-packages>`") == "Packages"
    assert clean_inline(":keyword:`!if`") == "if"
    assert clean_inline("`docs\n<https://x.org>`_\\. more") == "docs. more"
    assert clean_inline("use ``x = 1``") == "use `x = 1`"


def test_parse_sections_titles_levels_and_labels():
    secs = [s for s in parse_sections(SAMPLE) if s["title"]]
    assert [s["title"] for s in secs] == ["More Control Flow", "Default Argument Values", "Keyword-Only Arguments"]
    assert [s["level"] for s in secs] == [1, 2, 3]
    assert secs[0]["anchor"] == "tut-top" and secs[1]["anchor"] == "tut-defaults"
    assert secs[2]["anchor"] == ""


def test_clean_body_drops_index_keeps_note_and_code():
    body = clean_body(parse_sections(SAMPLE)[2]["lines"])
    assert "hidden entry" not in body and ".. index" not in body
    assert "Note:" in body and "Important note text." in body
    assert "Example:" in body and "def f(a, L=[]):" in body
    assert "<https://" not in body and "len" in body


def test_chunk_text_respects_size_and_overlap():
    paras = [" ".join(f"w{i}_{j}" for j in range(60)) for i in range(10)]
    chunks = chunk_text("\n\n".join(paras), max_words=130, overlap_words=60)
    assert all(len(c.split()) <= 130 for c in chunks)
    # the last paragraph of one chunk starts the next one
    assert chunks[1].split("\n\n")[0] == chunks[0].split("\n\n")[-1]
    assert len(chunk_text("one two three", max_words=10)) == 1


def test_shipped_corpus_is_clean():
    chunks = load_chunks(DEFAULT_CHUNKS)
    assert len(chunks) > 500
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.url.startswith(DOCS_BASE) for c in chunks)
    leftovers = [c.id for c in chunks if re.search(r"(?<!\w):\w[\w-]*:`|<https?://[^>]*>`_", c.text)]
    assert not leftovers, leftovers[:5]
    assert {c.doc.split("/")[0] for c in chunks} == {"tutorial", "faq", "howto"}
