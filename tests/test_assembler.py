from pdf2md.assembler import MarkdownDocumentAssembler


def test_frontmatter_contains_title_and_pages():
    assembler = MarkdownDocumentAssembler(ingestor=None)
    md = assembler.assemble(
        {"title": "Doc", "page_count": 2, "author": "", "source_file": "x.pdf", "creation_date": ""},
        [["# Heading", "Body text"], []],
    )
    assert md.startswith("---")
    assert "title: Doc" in md
    assert "page_count: 2" in md
    assert "<!-- page: 0 -->" in md
    assert "<!-- page: 1 -->" not in md  # empty page skipped


def test_table_parser_splits_multiple_tables():
    from pdf2md.parsers.table import TableParser
    text = "<table><tr><td>1</td></tr></table><table><tr><td>2</td></tr></table>"
    result = TableParser._split_tables(text)
    assert len(result) == 2
