"""Convert the Markdown report to a Word document (Pandoc from pypandoc_binary).

    python -m Lab2.tools.build_report                    # Lab2/REPORT.md -> Lab2/REPORT.docx
    python -m Lab2.tools.build_report --out some.docx

Layout follows the lab 1 report: A4, margins 2 / 2 / 2.5 / 1.5 cm, Times New Roman 14 pt,
1.5 line spacing, 1.25 cm first-line indent, bold headings (Word styles Heading 1-3),
Consolas for code, page numbers from the second page.

A line `!code <file> <name> [<name> ...]` in the Markdown is replaced with the current
source of those functions ("Class.method" for methods, "@first..last" for the lines
between two line prefixes), so the listings always match the program.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import textwrap
import zipfile
from pathlib import Path

import pypandoc

from tools.code_screenshots import _fragment

_ROOT = Path(__file__).resolve().parent.parent.parent
_LAB = _ROOT / "Lab2"

_FONT = "Times New Roman"
_MONO = "Consolas"
_INDENT = 709           # 1.25 cm in twentieths of a point

_FONTS = f'<w:rFonts w:ascii="{_FONT}" w:hAnsi="{_FONT}" w:cs="{_FONT}" w:eastAsia="{_FONT}"/>'
_HEADING_RPR = f'<w:rPr>{_FONTS}<w:b/><w:bCs/><w:color w:val="000000"/><w:sz w:val="%d"/><w:szCs w:val="%d"/></w:rPr>'

_DOC_DEFAULTS = f"""<w:docDefaults>
    <w:rPrDefault><w:rPr>{_FONTS}<w:sz w:val="28"/><w:szCs w:val="28"/>
      <w:lang w:val="uk-UA" w:eastAsia="uk-UA" w:bidi="ar-SA"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:before="0" w:after="0" w:line="360" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>"""


def _paragraph(style_id: str, name: str, ppr: str, rpr: str = "", based: str = "Normal",
               nxt: str = "", custom: bool = True) -> str:
    return (f'<w:style w:type="paragraph" {"w:customStyle=\"1\" " if custom else ""}w:styleId="{style_id}">'
            f'<w:name w:val="{name}"/><w:basedOn w:val="{based}"/>'
            + (f'<w:next w:val="{nxt}"/>' if nxt else "") + "<w:qFormat/>"
            + f"<w:pPr>{ppr}</w:pPr>" + (f"<w:rPr>{rpr}</w:rPr>" if rpr else "") + "</w:style>")


def _heading(level: int, size: int, before: int, after: int) -> str:
    return (f'<w:style w:type="paragraph" w:styleId="Heading{level}"><w:name w:val="heading {level}"/>'
            f'<w:basedOn w:val="Normal"/><w:next w:val="BodyText"/><w:uiPriority w:val="9"/><w:qFormat/>'
            f'<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="{before}" w:after="{after}"/>'
            f'<w:jc w:val="left"/><w:outlineLvl w:val="{level - 1}"/></w:pPr>'
            + _HEADING_RPR % (size, size) + "</w:style>")


_BORDER = '<w:%s w:val="single" w:sz="4" w:space="0" w:color="000000"/>'

STYLES = {
    "BodyText": _paragraph("BodyText", "Body Text", f'<w:ind w:firstLine="{_INDENT}"/><w:jc w:val="both"/>',
                           custom=False),
    "FirstParagraph": _paragraph("FirstParagraph", "First Paragraph", "", based="BodyText", nxt="BodyText"),
    # list items: the body text without the first-line indent
    "Compact": _paragraph("Compact", "Compact", '<w:ind w:firstLine="0"/>', based="BodyText"),
    # keepNext on every cell keeps a table on one page
    "TableText": _paragraph("TableText", "Table Text",
                            '<w:keepNext/><w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/>'
                            '<w:ind w:firstLine="0"/><w:jc w:val="left"/>',
                            '<w:sz w:val="24"/><w:szCs w:val="24"/>'),
    "Heading1": _heading(1, 32, 360, 180),
    "Heading2": _heading(2, 28, 280, 120),
    "Heading3": _heading(3, 28, 240, 120),
    "Caption": _paragraph("Caption", "caption",
                          '<w:spacing w:before="120" w:after="240"/><w:ind w:firstLine="0"/><w:jc w:val="center"/>',
                          custom=False),
    "ImageCaption": _paragraph("ImageCaption", "Image Caption", "", based="Caption"),
    "TableCaption": _paragraph("TableCaption", "Table Caption",
                               '<w:keepNext/><w:spacing w:before="240" w:after="120"/>', based="Caption"),
    "Figure": _paragraph("Figure", "Figure",
                         '<w:spacing w:before="180" w:after="0" w:line="240" w:lineRule="auto"/>'
                         '<w:ind w:firstLine="0"/><w:jc w:val="center"/>'),
    "CaptionedFigure": _paragraph("CaptionedFigure", "Captioned Figure", "<w:keepNext/>", based="Figure"),
    "SourceCode": _paragraph(
        "SourceCode", "Source Code",
        '<w:keepLines/><w:pBdr>' + "".join(f'<w:{s} w:val="single" w:sz="4" w:space="4" w:color="BFBFBF"/>'
                             for s in ("top", "left", "bottom", "right")) + '</w:pBdr>'
        '<w:shd w:val="clear" w:color="auto" w:fill="F7F7F7"/>'
        '<w:spacing w:before="120" w:after="180" w:line="240" w:lineRule="auto"/>'
        '<w:ind w:left="113" w:right="113" w:firstLine="0"/><w:jc w:val="left"/>',
        f'<w:rFonts w:ascii="{_MONO}" w:hAnsi="{_MONO}" w:cs="{_MONO}"/><w:sz w:val="18"/><w:szCs w:val="18"/>'),
    "VerbatimChar": (f'<w:style w:type="character" w:customStyle="1" w:styleId="VerbatimChar">'
                     f'<w:name w:val="Verbatim Char"/><w:rPr>'
                     f'<w:rFonts w:ascii="{_MONO}" w:hAnsi="{_MONO}" w:cs="{_MONO}"/></w:rPr></w:style>'),
    "Table": ('<w:style w:type="table" w:default="1" w:styleId="Table"><w:name w:val="Table"/>'
              '<w:basedOn w:val="TableNormal"/><w:semiHidden/><w:unhideWhenUsed/><w:qFormat/>'
              '<w:tblPr><w:jc w:val="center"/><w:tblInd w:w="0" w:type="dxa"/><w:tblBorders>'
              + "".join(_BORDER % s for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
              + '</w:tblBorders><w:tblCellMar><w:top w:w="28" w:type="dxa"/><w:left w:w="85" w:type="dxa"/>'
              '<w:bottom w:w="28" w:type="dxa"/><w:right w:w="85" w:type="dxa"/></w:tblCellMar></w:tblPr>'
              '<w:tblStylePr w:type="firstRow"><w:rPr><w:b/><w:bCs/></w:rPr>'
              '<w:tcPr><w:vAlign w:val="center"/></w:tcPr></w:tblStylePr></w:style>'),
    # title page
    "TitleCenter": _paragraph("TitleCenter", "Title Center", '<w:ind w:firstLine="0"/><w:jc w:val="center"/>'),
    "TitleMain": _paragraph("TitleMain", "Title Main", '<w:spacing w:before="3500"/>', based="TitleCenter"),
    "TitleAuthor": _paragraph("TitleAuthor", "Title Author",
                              '<w:spacing w:before="1700"/><w:ind w:left="4820" w:firstLine="0"/><w:jc w:val="left"/>'),
    "TitleCity": _paragraph("TitleCity", "Title City", '<w:spacing w:before="3300"/>',
                            "<w:b/><w:bCs/><w:i/><w:iCs/>", based="TitleCenter"),
}

_SECTION = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter1"/>'
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1134" w:right="850" w:bottom="1134" w:left="1417" w:header="709" '
            'w:footer="454" w:gutter="0"/><w:titlePg/></w:sectPr>')

_FOOTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           '<w:p><w:pPr><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>'
           '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
           '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>2</w:t></w:r>'
           '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')


_TABLE_GAP = ('<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="200" w:lineRule="exact"/>'
              '<w:ind w:firstLine="0"/></w:pPr></w:p>')


def _pandoc(*args: str) -> None:
    subprocess.run([pypandoc.get_pandoc_path(), *args], check=True, cwd=str(_LAB))


def _rewrite(docx: Path, edits: dict, extra: dict | None = None) -> None:
    """Rewrite members of a .docx: edits[name](text) -> text; extra adds new members."""
    with zipfile.ZipFile(docx) as z:
        members = [(info, z.read(info.filename)) for info in z.infolist()]
    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as z:
        for info, data in members:
            if info.filename in edits:
                data = edits[info.filename](data.decode("utf-8")).encode("utf-8")
            z.writestr(info, data)
        for name, text in (extra or {}).items():
            z.writestr(name, text)


def _styles(xml: str) -> str:
    xml = re.sub(r"<w:docDefaults>.*?</w:docDefaults>", lambda m: _DOC_DEFAULTS, xml, flags=re.S)
    for style_id, block in STYLES.items():
        pattern = r'<w:style [^>]*w:styleId="%s"[^>]*>.*?</w:style>' % style_id
        if re.search(pattern, xml, flags=re.S):
            xml = re.sub(pattern, lambda m: block, xml, count=1, flags=re.S)
        else:
            xml = xml.replace("</w:styles>", block + "</w:styles>")
    return xml


def make_reference(path: Path) -> None:
    """Pandoc's default reference document with the report's styles, page set-up and footer."""
    _pandoc("-o", str(path), "--print-default-data-file", "reference.docx")
    _rewrite(path, {
        "word/styles.xml": _styles,
        "word/document.xml": lambda x: re.sub(r"<w:sectPr>.*?</w:sectPr>", lambda m: _SECTION, x, flags=re.S),
        "word/_rels/document.xml.rels": lambda x: x.replace(
            "</Relationships>",
            '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" '
            'Id="rIdFooter1" Target="footer1.xml"/></Relationships>'),
        "[Content_Types].xml": lambda x: x.replace(
            "</Types>",
            '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.wordprocessingml.footer+xml"/></Types>'),
    }, extra={"word/footer1.xml": _FOOTER})


def expand_code(text: str) -> str:
    """Replace `!code file name...` lines with the source of those fragments."""
    def block(match: re.Match) -> str:
        path, *names = match.group(1).split()
        parts = []
        for name in names:
            what = tuple(name[1:].split("..")) if name.startswith("@") else name
            _, lines = _fragment(path, what)
            parts.append(textwrap.dedent("\n".join(lines)))
        return "```python\n" + "\n\n".join(parts) + "\n```"
    return re.sub(r"^!code (.+)$", block, text, flags=re.M)


def _document(xml: str) -> str:
    # table cells: compact 12 pt text instead of the list style
    # and full text width, with a small gap before the text that follows
    def table(match: re.Match) -> str:
        return (match.group(0)
                .replace('w:val="Compact"', 'w:val="TableText"')
                .replace('<w:tblW w:type="auto" w:w="0" />', '<w:tblW w:type="pct" w:w="5000" />')
                + _TABLE_GAP)
    xml = re.sub(r"<w:tbl>.*?</w:tbl>", table, xml, flags=re.S)
    # Pandoc marks text inside formulas as both "normal" and "plain"; the schema allows one
    xml = re.sub(r"(<m:nor\s*/>)<m:sty [^>]*/>", r"\1", xml)
    # the first heading starts the page after the title page
    return re.sub(r'(<w:pStyle w:val="Heading1"\s*/>)', r"\1<w:pageBreakBefore/>", xml, count=1)


def build(src: Path, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reference = Path(tmp) / "reference.docx"
        make_reference(reference)
        expanded = Path(tmp) / "report.md"
        expanded.write_text(expand_code(src.read_text(encoding="utf-8")), encoding="utf-8")
        _pandoc(str(expanded), "--from", "markdown", "--to", "docx", "-o", str(out),
                "--reference-doc", str(reference), "--resource-path", str(src.parent))
    _rewrite(out, {
        "word/document.xml": _document,
        # Pandoc lists every image separately; a default for the extension is more portable
        "[Content_Types].xml": lambda x: x.replace(
            "<Override ", '<Default Extension="png" ContentType="image/png"/><Override ', 1),
    })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert the Markdown report to .docx.")
    ap.add_argument("--src", default=str(_LAB / "REPORT.md"))
    ap.add_argument("--out", default=str(_LAB / "REPORT.docx"))
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    build(Path(args.src).resolve(), out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
