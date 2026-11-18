"""Runs only when pypdf is installed. Builds a tiny real PDF in memory (no fixture files)."""
import importlib.util
import unittest

from gharcare.errors import IngestionError
from gharcare.ingestion import PdfTextExtractor


def make_pdf(pages):
    def esc(t):
        return t.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    n = len(pages)
    font_id = 3 + 2 * n
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{' '.join(f'{3 + 2 * i} 0 R' for i in range(n))}] /Count {n} >>".encode()]
    for i, text in enumerate(pages):
        objs.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
                     f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>").encode())
        stream = f"BT /F1 12 Tf 72 720 Td ({esc(text)}) Tj ET".encode() if text else b""
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


@unittest.skipUnless(importlib.util.find_spec("pypdf"), "pypdf not installed (pip install pypdf)")
class PdfExtractionTests(unittest.TestCase):
    def test_extracts_text_per_page_with_correct_numbers(self):
        pages = PdfTextExtractor().extract(make_pdf(["Error code E4 defrost sensor", "Second page text"]))
        self.assertEqual([p.page_number for p in pages], [1, 2])
        self.assertIn("E4", pages[0].text)
        self.assertIn("Second page", pages[1].text)

    def test_page_without_text_layer_yields_empty_text_not_an_error(self):
        pages = PdfTextExtractor().extract(make_pdf(["", "Has text"]))
        self.assertEqual(pages[0].text.strip(), "")
        self.assertIn("Has text", pages[1].text)

    def test_garbage_bytes_become_a_clean_ingestion_error(self):
        with self.assertRaises(IngestionError):
            PdfTextExtractor().extract(b"this is not a pdf")


if __name__ == "__main__":
    unittest.main()
