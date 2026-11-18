import unittest

from gharcare.errors import DomainValidationError
from gharcare.ingestion import Chunker, PageText


def chunk(pages, **kw):
    return Chunker(**kw).chunk_pages("u1", "doc1", [PageText(i, t) for i, t in enumerate(pages, 1)])


class ChunkerTests(unittest.TestCase):
    def test_chunks_never_cross_pages_and_keep_exact_page_numbers(self):
        chunks = chunk(["Alpha paragraph.\n\nBeta paragraph.", "Gamma paragraph."])
        self.assertEqual([c.page_number for c in chunks], [1, 2])
        self.assertIn("Alpha", chunks[0].text)
        self.assertIn("Beta", chunks[0].text)
        self.assertNotIn("Gamma", chunks[0].text)

    def test_heading_becomes_section_and_is_prepended(self):
        chunks = chunk(["DEFROSTING\n\nUnplug the fridge first."])
        self.assertEqual(chunks[0].section, "DEFROSTING")
        self.assertEqual(chunks[0].text, "DEFROSTING\nUnplug the fridge first.")

    def test_section_persists_across_pages_until_a_new_heading(self):
        chunks = chunk(["SAFETY\n\nPage one text.", "Page two text.", "INSTALLATION\n\nPage three text."])
        self.assertEqual([c.section for c in chunks], ["SAFETY", "SAFETY", "INSTALLATION"])

    def test_chunks_do_not_cross_headings(self):
        chunks = chunk(["ONE\n\nfirst body\n\nTWO\n\nsecond body"])
        self.assertEqual([c.section for c in chunks], ["ONE", "TWO"])
        self.assertNotIn("second", chunks[0].text)

    def test_numbered_steps_inside_a_list_are_not_split_into_headings(self):
        page = "Follow these steps:\n1. Switch off the fridge\n2. Remove the food\n3. Wait six hours"
        chunks = chunk([page])
        self.assertEqual(len(chunks), 1)
        self.assertIn("2. Remove the food", chunks[0].text)

    def test_standalone_numbered_line_is_a_heading(self):
        chunks = chunk(["3. Troubleshooting\n\nCheck the plug."])
        self.assertEqual(chunks[0].section, "3. Troubleshooting")

    def test_error_code_table_rows_are_not_mistaken_for_headings(self):
        chunks = chunk(["E4 DEFROST SENSOR FAULT CALL SERVICE\nE1 TEMPERATURE SENSOR FAULT"])
        self.assertIsNone(chunks[0].section)
        self.assertIn("E4", chunks[0].text)

    def test_urdu_script_lines_are_not_misread_as_all_caps_headings(self):
        chunks = chunk(["فریج میں برف جم جائے\n\nپہلے پلگ نکال دیں۔"])
        self.assertIsNone(chunks[0].section)
        self.assertIn("پلگ", chunks[0].text)

    def test_caps_heading_inside_a_hard_wrapped_block(self):
        chunks = chunk(["Some text before\nWARRANTY TERMS\nSome text after"])
        self.assertEqual([c.section for c in chunks], [None, "WARRANTY TERMS"])
        self.assertEqual(chunks[1].text, "WARRANTY TERMS\nSome text after")

    def test_bare_page_numbers_are_dropped_but_other_text_is_kept(self):
        chunks = chunk(["Real content here.\n\n12\n\nPage 3 of 40\n\nMore content."])
        joined = " ".join(c.text for c in chunks)
        self.assertNotIn("Page 3", joined)
        self.assertNotRegex(joined, r"(?m)^12$")
        self.assertIn("Real content", joined)
        self.assertIn("More content", joined)

    def test_oversized_block_is_split_without_losing_words(self):
        words = [f"word{i}" for i in range(400)]
        chunks = chunk([" ".join(words)], max_chars=200)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(c.text) <= 200 for c in chunks))
        self.assertEqual(" ".join(c.text for c in chunks).split(), words)

    def test_oversized_line_list_splits_on_line_boundaries(self):
        lines = [f"E{i} - fault description number {i}" for i in range(40)]
        chunks = chunk(["\n".join(lines)], max_chars=300)
        rebuilt = [ln for c in chunks for ln in c.text.split("\n")]
        self.assertEqual(rebuilt, lines)   # no line was cut in half

    def test_single_giant_token_is_hard_split(self):
        chunks = chunk(["x" * 1000], max_chars=200)
        self.assertTrue(all(len(c.text) <= 200 for c in chunks))
        self.assertEqual(sum(len(c.text) for c in chunks), 1000)

    def test_ids_are_deterministic_and_sequential(self):
        chunks = chunk(["A para.\n\nB para.", "C para."], max_chars=100)
        self.assertEqual([c.id for c in chunks], [f"doc1-{i}" for i in range(len(chunks))])
        self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))

    def test_empty_input_gives_no_chunks(self):
        self.assertEqual(chunk(["", "  \n\n  "]), [])

    def test_windows_line_endings(self):
        chunks = chunk(["DEFROST\r\n\r\nUnplug first.\r\nWait six hours."])
        self.assertEqual(chunks[0].section, "DEFROST")
        self.assertIn("Unplug first.\nWait six hours.", chunks[0].text)

    def test_running_footer_codes_and_page_numbers_are_removed(self):
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        pages = [f"INTRO\n\nThe {w} drawer slides out easily.\n\n512960\n{i}" for i, w in enumerate(words, 1)]
        chunks = chunk(pages)
        self.assertEqual(len(chunks), 6)
        self.assertTrue(all("512960" not in c.text for c in chunks))
        self.assertTrue(all(f"The {w} drawer" in c.text for c, w in zip(chunks, words)))      # real content untouched
        self.assertTrue(any("512960" in c.text for c in chunk(pages[:3])))                    # short documents are left alone

    def test_wrapped_all_caps_headings_become_one_section(self):
        chunks = chunk(["BEFORE USING THE APPLIANCE FOR\nTHE FIRST TIME\nLet the appliance rest upright for two hours."])
        self.assertEqual(chunks[0].section, "BEFORE USING THE APPLIANCE FOR THE FIRST TIME")
        self.assertIn("rest upright", chunks[0].text)

    def test_stray_diagram_fragments_are_dropped_but_short_sentences_survive(self):
        chunks = chunk(["EN", "min 50 mm", "Door open for too long."])
        self.assertEqual([c.text for c in chunks], ["Door open for too long."])
        self.assertEqual(chunk(["E7 - Door open"])[0].text, "E7 - Door open")

    def test_invalid_max_chars(self):
        with self.assertRaises(DomainValidationError):
            Chunker(max_chars=10)


if __name__ == "__main__":
    unittest.main()
