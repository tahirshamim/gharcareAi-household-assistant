"""Checks the web page without a browser: syntax, no unsafe DOM APIs, and that every API path the page calls exists in the backend."""
import re
import shutil
import subprocess
import unittest
from pathlib import Path

API = Path(__file__).parent.parent / "src" / "gharcare" / "api"
JS = (API / "static" / "app.js").read_text(encoding="utf-8")
HTML = (API / "static" / "index.html").read_text(encoding="utf-8")


def _split_top_level(text, sep):
    parts, depth, quote, buf = [], 0, None, ""
    for ch in text:
        if quote:
            buf += ch
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote, buf = ch, buf + ch
        elif ch in "([{":
            depth, buf = depth + 1, buf + ch
        elif ch in ")]}":
            depth, buf = depth - 1, buf + ch
        elif ch == sep and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    return parts + [buf]


def _first_argument(source, start):
    depth, quote, i = 0, None, start
    while i < len(source):
        ch = source[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif ch == "," and depth == 0:
            break
        i += 1
    return source[start:i]


def frontend_paths():
    paths = set()
    for call in re.finditer(r"\b(?:api|post|downloadFile)\(", JS):
        argument = _first_argument(JS, call.end()).strip()
        if not argument.startswith(('"', "'")):
            continue
        path = ""
        for part in _split_top_level(argument, "+"):
            part = part.strip()
            if part[:1] in "\"'":
                literal = part[1:-1]
                path += literal.split("?")[0]
                if "?" in literal:
                    break
            elif path.endswith("/"):
                path += "{}"                       # a path parameter such as an id
        paths.add(path)
    return paths


def backend_paths():
    source = (API / "app.py").read_text(encoding="utf-8")
    found = {re.sub(r"\{[^}]+\}", "{}", p) for p in re.findall(r'@api\.(?:get|post|put|delete)\(\s*"([^"]+)"', source)}
    found |= {p[len("/api"):] for p in re.findall(r'@app\.get\(\s*"(/api/[^"]+)"', source) if not p.startswith("/api/docs")}
    return found


class FrontendTests(unittest.TestCase):
    def test_javascript_syntax(self):
        if not shutil.which("node"):
            self.skipTest("node not installed")
        result = subprocess.run(["node", "--check", str(API / "static" / "app.js")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_unsafe_dom_apis(self):
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
            self.assertNotIn(forbidden, JS, f"{forbidden} could let document text run as script")

    def test_html_is_compatible_with_a_strict_content_security_policy(self):
        self.assertNotRegex(HTML, r"<script(?![^>]*\bsrc=)")                       # no inline scripts
        self.assertNotRegex(HTML, r"\son[a-z]+\s*=")                               # no inline event handlers
        self.assertNotIn("http://", HTML.replace("http://www.w3.org", ""))        # nothing loaded from other sites
        self.assertNotRegex(HTML, r"https?://")
        self.assertNotRegex(HTML, r"\sstyle\s*=")                                 # inline style attributes are blocked
        self.assertNotRegex(JS, r"\bstyle\s*:\s*[\"']")                          # ...including via h(tag, {style: ...})
        self.assertNotIn(".setAttribute(\"style\"", JS)

    def test_every_api_path_the_page_calls_exists_in_the_backend(self):
        used, available = frontend_paths(), backend_paths()
        self.assertGreaterEqual(len(used), 15)                                      # the parser really found the calls
        self.assertEqual(sorted(p for p in used if p not in available), [], "the page calls API paths the backend does not define")
        for expected in ("/ask", "/documents/{}/extract", "/drafts/{}/confirm", "/maintenance/{}/complete", "/report", "/extraction/accuracy"):
            self.assertIn(expected, used)

    def test_parser_helpers(self):
        self.assertEqual(_split_top_level('"/a/" + x.id + "/b"', "+"), ['"/a/" ', " x.id ", ' "/b"'])
        self.assertEqual(_first_argument('foo("/x/" + id, {a: 1})', 4), '"/x/" + id')


if __name__ == "__main__":
    unittest.main()
