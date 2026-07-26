from rag.chunk import chunk_text, Chunk
from rag.config import Settings

S = Settings(chunk_chars=200, chunk_overlap_chars=40)


def test_splits_on_markdown_headings():
    text = "# A\n\nalpha content\n\n# B\n\nbeta content\n"
    chunks = chunk_text(text, S)
    paths = [c.heading_path for c in chunks]
    assert "A" in paths and "B" in paths


def test_heading_path_is_hierarchical():
    text = "# Top\n\nintro\n\n## Inner\n\ndetail here\n"
    chunks = chunk_text(text, S)
    assert any(c.heading_path == "Top > Inner" for c in chunks)


def test_long_section_is_split_with_overlap():
    body = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(f"# H\n\n{body}\n", S)
    assert len(chunks) > 1
    assert all(len(c.text) <= S.chunk_chars * 1.5 for c in chunks)
    # overlap means the tail of one chunk reappears at the head of the next
    assert chunks[0].text[-20:] in chunks[1].text


def test_ordinals_are_sequential_from_zero():
    text = "# A\n\n" + " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, S)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_plain_text_without_headings_still_chunks():
    chunks = chunk_text("just prose, no headings at all", S)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ""


def test_empty_text_produces_no_chunks():
    assert chunk_text("   \n\n  ", S) == []


def test_oversized_paragraph_does_not_double_apply_overlap():
    # One unbroken paragraph (no blank lines) forces word-level wrapping.
    # Each wrapped piece already carries its own overlap; re-packing those
    # pieces at the paragraph level would stack a second overlap on top and
    # inflate every chunk after the first past the budget.
    body = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(f"# H\n\n{body}\n", S)
    assert len(chunks) > 1
    assert all(
        len(c.text) <= S.chunk_chars + S.chunk_overlap_chars for c in chunks
    )


def test_single_token_longer_than_budget_is_split():
    # A long run with no whitespace (URL, hash, base64 blob) has no word
    # boundary to wrap on, so it must be hard-split instead of emitted whole.
    token = "x" * (S.chunk_chars * 3)
    chunks = chunk_text(f"# H\n\n{token}\n", S)
    assert len(chunks) > 1
    assert all(
        len(c.text) <= S.chunk_chars + S.chunk_overlap_chars for c in chunks
    )


def test_fenced_code_comments_are_not_parsed_as_headings():
    # Shell comments inside a fenced code block look exactly like markdown
    # headings to a line-by-line scanner with no fence state. This is the
    # dominant shape for personal technical notes full of shell snippets, not
    # an edge case.
    text = (
        "# Real Heading\n"
        "\n"
        "intro text before the fence\n"
        "\n"
        "```bash\n"
        "# this is a shell comment, not a heading\n"
        "echo hi\n"
        "## another comment that looks like an H2\n"
        "```\n"
        "\n"
        "more text after the fence\n"
    )
    chunks = chunk_text(text, S)
    paths = {c.heading_path for c in chunks}
    assert paths == {"Real Heading"}, (
        f"fenced '#' comments must not become heading paths, got {paths}"
    )
    # the fenced content itself must still be present verbatim in some chunk
    assert any("echo hi" in c.text for c in chunks)
    assert any("more text after the fence" in c.text for c in chunks)
