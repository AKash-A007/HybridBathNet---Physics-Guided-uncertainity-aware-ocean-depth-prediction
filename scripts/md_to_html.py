"""
Generates a standalone styled HTML version of docs/ARCHITECTURE.md with live Mermaid diagram rendering.
"""
import re
from pathlib import Path


def generate_html_from_md(md_path: Path, html_path: Path):
    md_content = md_path.read_text(encoding="utf-8")

    # Simple markdown to HTML conversion
    html_body = md_content

    # Extract Mermaid code block
    mermaid_match = re.search(r"```mermaid\s+(.*?)\s+```", html_body, re.DOTALL)
    mermaid_code = mermaid_match.group(1) if mermaid_match else ""

    # Replace mermaid block with a div
    html_body = re.sub(
        r"```mermaid\s+.*?\s+```",
        r'<div class="mermaid-container"><h3>Pipeline Architecture Flowchart</h3><pre class="mermaid">\n' + mermaid_code + '\n</pre></div>',
        html_body,
        flags=re.DOTALL
    )

    # Convert code blocks
    html_body = re.sub(
        r"```([a-zA-Z0-9_-]*)\n(.*?)\n```",
        r'<pre><code class="language-\1">\2</code></pre>',
        html_body,
        flags=re.DOTALL
    )

    # Headings
    html_body = re.sub(r"^### (.*)$", r"<h3>\1</h3>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^## (.*)$", r"<h2>\1</h2>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^# (.*)$", r"<h1>\1</h1>", html_body, flags=re.MULTILINE)

    # Tables
    def table_replace(match):
        lines = match.group(0).strip().split("\n")
        rows = []
        is_header = True
        for r_idx, l in enumerate(lines):
            if re.match(r"^\|[\s\-:]+(\|[\s\-:]+)+\|$", l):
                is_header = False
                continue
            cells = [c.strip() for c in l.strip("|").split("|")]
            tag = "th" if is_header else "td"
            row_html = "".join([f"<{tag}>{c}</{tag}>" for c in cells])
            rows.append(f"<tr>{row_html}</tr>")
        return f"<table><tbody>{''.join(rows)}</tbody></table>"

    html_body = re.sub(r"(\|.*\|\n)+", table_replace, html_body)

    # Inline formatting
    html_body = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html_body)
    html_body = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html_body)
    html_body = re.sub(r"`(.*?)`", r"<code>\1</code>", html_body)
    html_body = re.sub(r"^---$", r"<hr/>", html_body, flags=re.MULTILINE)

    # Wrap in complete standalone HTML page
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HybridBathNet Architecture Specification</title>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
  </script>
  <style>
    :root {{
      --primary: #0f4c81;
      --primary-light: #e8f1f8;
      --accent: #0284c7;
      --text: #1f2937;
      --bg: #ffffff;
      --card-bg: #f8fafc;
      --border: #e2e8f0;
      --code-bg: #f1f5f9;
    }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.65;
      color: var(--text);
      background: var(--bg);
      max-width: 960px;
      margin: 0 auto;
      padding: 40px 24px 80px;
    }}
    h1 {{
      color: var(--primary);
      font-size: 2.25rem;
      border-bottom: 2px solid var(--primary-light);
      padding-bottom: 12px;
      margin-top: 0;
    }}
    h2 {{
      color: var(--primary);
      font-size: 1.5rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      margin-top: 36px;
    }}
    h3 {{
      color: #334155;
      font-size: 1.15rem;
      margin-top: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 10px 14px;
      border: 1px solid var(--border);
      text-align: left;
    }}
    th {{
      background: var(--primary);
      color: #ffffff;
      font-weight: 600;
    }}
    tr:nth-child(even) {{
      background: var(--card-bg);
    }}
    pre {{
      background: var(--code-bg);
      padding: 16px;
      border-radius: 8px;
      overflow-x: auto;
      font-family: 'JetBrains Mono', Consolas, monospace;
      font-size: 0.88rem;
      border: 1px solid var(--border);
    }}
    code {{
      font-family: 'JetBrains Mono', Consolas, monospace;
      font-size: 0.88em;
      background: var(--code-bg);
      padding: 2px 6px;
      border-radius: 4px;
      color: #b91c1c;
    }}
    pre code {{
      background: transparent;
      padding: 0;
      color: inherit;
    }}
    hr {{
      border: 0;
      height: 1px;
      background: var(--border);
      margin: 32px 0;
    }}
    .mermaid-container {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin: 24px 0;
      text-align: center;
    }}
    @media print {{
      body {{ max-width: 100%; padding: 0; }}
      .mermaid-container {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
{html_body}
</body>
</html>"""

    html_path.write_text(full_html, encoding="utf-8")
    print(f"Generated HTML doc at: {html_path}")


if __name__ == "__main__":
    generate_html_from_md(Path("docs/ARCHITECTURE.md"), Path("docs/ARCHITECTURE.html"))
