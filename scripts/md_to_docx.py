"""
Converts docs/ARCHITECTURE.md into a high-quality Microsoft Word (.docx) document
using pure Python standard libraries (zipfile, xml.sax.saxutils).
"""
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def create_docx_from_markdown(md_path: Path, docx_path: Path):
    text = md_path.read_text(encoding="utf-8")

    # XML components
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
        <w:color w:val="262626"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="160" w:line="276" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:pPr>
      <w:spacing w:before="360" w:after="180"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri Light" w:hAnsi="Calibri Light"/>
      <w:b/>
      <w:sz w:val="36"/>
      <w:szCs w:val="36"/>
      <w:color w:val="0F4C81"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:pPr>
      <w:spacing w:before="280" w:after="140"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri Light" w:hAnsi="Calibri Light"/>
      <w:b/>
      <w:sz w:val="28"/>
      <w:szCs w:val="28"/>
      <w:color w:val="1E6B7B"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:pPr>
      <w:spacing w:before="200" w:after="100"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
      <w:color w:val="333333"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="CodeBlock">
    <w:name w:val="Code Block"/>
    <w:pPr>
      <w:spacing w:before="100" w:after="100" w:line="240" w:lineRule="auto"/>
      <w:shd w:val="clear" w:color="auto" w:fill="F4F6F8"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>
      <w:sz w:val="19"/>
      <w:szCs w:val="19"/>
      <w:color w:val="1F2937"/>
    </w:rPr>
  </w:style>
</w:styles>"""

    # Parse markdown lines into OpenXML document elements
    body_elements = []
    lines = text.split("\n")
    i = 0
    n = len(lines)

    def format_runs(md_inline: str) -> str:
        """Parse bold, inline code, and normal text into w:r elements."""
        runs_xml = []
        # Tokenize markdown formatting
        tokens = re.split(r'(\*\*.*?\*\*|\`.*?\`|\*.*?\*)', md_inline)
        for tok in tokens:
            if not tok:
                continue
            if tok.startswith("**") and tok.endswith("**"):
                inner = escape(tok[2:-2])
                runs_xml.append(f'<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{inner}</w:t></w:r>')
            elif tok.startswith("`") and tok.endswith("`"):
                inner = escape(tok[1:-1])
                runs_xml.append(f'<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:shd w:val="clear" w:color="auto" w:fill="F0F2F5"/><w:color w:val="C7254E"/></w:rPr><w:t xml:space="preserve">{inner}</w:t></w:r>')
            elif tok.startswith("*") and tok.endswith("*"):
                inner = escape(tok[1:-1])
                runs_xml.append(f'<w:r><w:rPr><w:i/></w:rPr><w:t xml:space="preserve">{inner}</w:t></w:r>')
            else:
                inner = escape(tok)
                runs_xml.append(f'<w:r><w:t xml:space="preserve">{inner}</w:t></w:r>')
        return "".join(runs_xml)

    in_code_block = False
    code_lines = []

    while i < n:
        line = lines[i]

        # Check for code blocks
        if line.strip().startswith("```"):
            if in_code_block:
                # End of code block
                in_code_block = False
                for cl in code_lines:
                    safe_c = escape(cl)
                    body_elements.append(
                        f'<w:p><w:pPr><w:pStyle w:val="CodeBlock"/></w:pPr>'
                        f'<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="19"/></w:rPr>'
                        f'<w:t xml:space="preserve">{safe_c}</w:t></w:r></w:p>'
                    )
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        stripped = line.strip()

        # Blank line
        if not stripped:
            i += 1
            continue

        # Horizontal Rule
        if stripped in ("---", "***", "___"):
            body_elements.append(
                '<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="12" w:space="8" w:color="D1D5DB"/></w:pBdr></w:pPr></w:p>'
            )
            i += 1
            continue

        # Headings
        if stripped.startswith("# ") and not stripped.startswith("## "):
            h_text = stripped[2:].strip()
            body_elements.append(f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>{format_runs(h_text)}</w:p>')
            i += 1
            continue
        elif stripped.startswith("## ") and not stripped.startswith("### "):
            h_text = stripped[3:].strip()
            body_elements.append(f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr>{format_runs(h_text)}</w:p>')
            i += 1
            continue
        elif stripped.startswith("### "):
            h_text = stripped[4:].strip()
            body_elements.append(f'<w:p><w:pPr><w:pStyle w:val="Heading3"/></w:pPr>{format_runs(h_text)}</w:p>')
            i += 1
            continue

        # Markdown Table Detection
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            if len(table_lines) >= 2:
                # Build Word Table
                rows_xml = []
                is_header = True
                for r_idx, t_line in enumerate(table_lines):
                    # Check if separator line (|---|---|)
                    if re.match(r"^\|[\s\-:]+(\|[\s\-:]+)+\|$", t_line):
                        is_header = False
                        continue

                    cells = [c.strip() for c in t_line.strip("|").split("|")]
                    cells_xml = []
                    for c_text in cells:
                        shd = '<w:shd w:val="clear" w:color="auto" w:fill="0F4C81"/>' if (r_idx == 0) else (
                            '<w:shd w:val="clear" w:color="auto" w:fill="F8FAFC"/>' if (r_idx % 2 == 1) else ''
                        )
                        text_fmt = format_runs(c_text)
                        if r_idx == 0:
                            # Bold white text for table header
                            text_fmt = f'<w:r><w:rPr><w:b/><w:color w:val="FFFFFF"/></w:rPr><w:t xml:space="preserve">{escape(c_text)}</w:t></w:r>'

                        cell_xml = f"""<w:tc>
  <w:tcPr>
    {shd}
    <w:tcMar>
      <w:top w:w="120" w:type="dxa"/>
      <w:bottom w:w="120" w:type="dxa"/>
      <w:left w:w="160" w:type="dxa"/>
      <w:right w:w="160" w:type="dxa"/>
    </w:tcMar>
  </w:tcPr>
  <w:p><w:pPr><w:spacing w:before="40" w:after="40"/></w:pPr>{text_fmt}</w:p>
</w:tc>"""
                        cells_xml.append(cell_xml)

                    row_xml = f"""<w:tr>
  <w:trPr>
    {'<w:tblHeader/>' if (r_idx == 0) else ''}
  </w:trPr>
  {''.join(cells_xml)}
</w:tr>"""
                    rows_xml.append(row_xml)

                tbl_xml = f"""<w:tbl>
  <w:tblPr>
    <w:tblStyle w:val="TableGrid"/>
    <w:tblW w:w="5000" w:type="pct"/>
    <w:tblBorders>
      <w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
      <w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
      <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
      <w:insideV w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
    </w:tblBorders>
  </w:tblPr>
  {''.join(rows_xml)}
</w:tbl>"""
                body_elements.append(tbl_xml)
            continue

        # Bullet lists
        if stripped.startswith("* ") or stripped.startswith("- "):
            bullet_text = stripped[2:].strip()
            body_elements.append(
                f'<w:p><w:pPr><w:pStyle w:val="Normal"/><w:ind w:left="480" w:hanging="240"/><w:spacing w:after="80"/></w:pPr>'
                f'<w:r><w:rPr><w:rFonts w:ascii="Symbol" w:hAnsi="Symbol"/></w:rPr><w:t> </w:t></w:r>'
                f'{format_runs(bullet_text)}</w:p>'
            )
            i += 1
            continue

        # Blockquote / callout
        if stripped.startswith(">"):
            bq_text = stripped.lstrip("> ").strip()
            body_elements.append(
                f'<w:p><w:pPr><w:pBdr><w:left w:val="single" w:sz="24" w:space="12" w:color="0F4C81"/></w:pBdr>'
                f'<w:ind w:left="360"/><w:shd w:val="clear" w:color="auto" w:fill="F0F7FF"/>'
                f'<w:spacing w:before="120" w:after="120"/></w:pPr>'
                f'{format_runs(bq_text)}</w:p>'
            )
            i += 1
            continue

        # Standard paragraph
        body_elements.append(f'<w:p>{format_runs(stripped)}</w:p>')
        i += 1

    # Document XML wrapping with standard 1-inch margins
    doc_body = "".join(body_elements)
    doc_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {doc_body}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    # Assemble .docx ZIP package
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/styles.xml", styles_xml)
        zf.writestr("word/document.xml", doc_xml)

    print(f"Successfully generated DOCX at: {docx_path} ({docx_path.stat().st_size} bytes)")


if __name__ == "__main__":
    src_md = Path("docs/ARCHITECTURE.md")
    dst_docx = Path("docs/ARCHITECTURE.docx")
    create_docx_from_markdown(src_md, dst_docx)
