import math
import unittest

from gharcare.retrieval import BM25Index, HashingEmbedder, fuse_rankings, tokenize


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


class TokenizerTests(unittest.TestCase):
    def test_hyphenated_codes_are_kept_whole_and_split(self):
        self.assertEqual(tokenize("Model HRF-306"), ["model", "hrf-306", "hrf", "306"])

    def test_error_codes_survive(self):
        self.assertIn("e4", tokenize("Error code E4!"))

    def test_roman_urdu_and_urdu_script(self):
        self.assertEqual(tokenize("Fridge mein barf"), ["fridge", "mein", "barf"])
        self.assertEqual(tokenize("فریج میں برف"), ["فریج", "میں", "برف"])

    def test_case_insensitive_and_punctuation_free(self):
        self.assertEqual(tokenize("E4, e4; (E4)"), ["e4", "e4", "e4"])


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index([
            ("a", "E4 defrost sensor fault. Call service."),
            ("b", "The refrigerator does not cool. Check the plug and the dial."),
            ("c", "Keep the door closed. Service the unit yearly."),
        ])

    def test_exact_error_code_finds_the_right_chunk_first(self):
        hits = self.index.search("what is E4", k=3)
        self.assertEqual(hits[0][0], "a")
        self.assertEqual(len(hits), 1)          # only chunks sharing a term are returned

    def test_no_overlap_returns_nothing(self):
        self.assertEqual(self.index.search("washing machine drum", k=5), [])

    def test_rare_terms_outweigh_common_terms(self):
        hits = dict(self.index.search("service sensor", k=3))
        self.assertGreater(hits["a"], hits["c"])   # 'sensor' is rarer than 'service'

    def test_a_rare_term_beats_many_repetitions_of_a_common_term(self):
        index = BM25Index([("repeats_common", "the the the unit"), ("common_a", "the dial"),
                           ("common_b", "the door"), ("has_rare", "sensor fault")])
        hits = index.search("the sensor", k=4)
        self.assertEqual(hits[0][0], "has_rare")      # without idf the repeated 'the' would win

    def test_k_limits_results_and_order_is_best_first(self):
        hits = self.index.search("the service plug dial", k=2)
        self.assertEqual(len(hits), 2)
        self.assertGreaterEqual(hits[0][1], hits[1][1])

    def test_empty_index_and_empty_query(self):
        self.assertEqual(BM25Index([]).search("x", 3), [])
        self.assertEqual(self.index.search("", 3), [])


class HashingEmbedderTests(unittest.TestCase):
    def setUp(self):
        self.emb = HashingEmbedder(dimension=256)

    def test_deterministic_across_instances(self):
        self.assertEqual(self.emb.embed_query("ice in freezer"), HashingEmbedder(256).embed_query("ice in freezer"))

    def test_unit_length_and_dimension(self):
        v = self.emb.embed_query("defrost the freezer")
        self.assertEqual(len(v), 256)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0)

    def test_shared_words_score_higher_than_unrelated_text(self):
        q = self.emb.embed_query("ice build-up in the freezer")
        related = self.emb.embed_documents(["Ice build-up in the freezer compartment"])[0]
        unrelated = self.emb.embed_documents(["Adjust the front feet until level"])[0]
        self.assertGreater(cosine(q, related), cosine(q, unrelated) + 0.3)

    def test_empty_text_gives_zero_vector_not_a_crash(self):
        self.assertEqual(sum(self.emb.embed_query("   ")), 0.0)

    def test_it_is_lexical_not_semantic(self):
        # Documented limitation: synonyms are NOT matched. The real model must beat this baseline.
        q = self.emb.embed_query("frost")
        d = self.emb.embed_documents(["ice builds up in the freezer"])[0]
        self.assertAlmostEqual(cosine(q, d), 0.0, places=6)


class ReserveSlotsTests(unittest.TestCase):
    def test_the_best_hit_of_each_retriever_always_gets_a_place(self):
        from gharcare.retrieval.hybrid import reserve_slots
        ordered = ["a", "b", "c", "d", "e", "f"]
        self.assertEqual(reserve_slots(ordered, ["f", "x"], ["a", "b"], k=3, reserve=1), ["a", "b", "f"])   # exact keyword hit 'f' rescued
        self.assertEqual(reserve_slots(ordered, ["a"], ["b"], k=3, reserve=1), ["a", "b", "c"])              # already present: unchanged
        self.assertEqual(reserve_slots(ordered, ["f", "e"], ["d", "c"], k=2, reserve=2), ["a", "f"])  # at most k//2 reserved: fused best kept
        self.assertEqual(reserve_slots(ordered, [], [], k=3, reserve=2), ["a", "b", "c"])


class FusionTests(unittest.TestCase):
    def test_item_found_by_both_rankers_beats_items_found_by_one(self):
        scores = fuse_rankings([["a", "b", "c"], ["b", "d"]])
        ordered = sorted(scores, key=lambda x: -scores[x])
        self.assertEqual(ordered[0], "b")

    def test_known_values(self):
        scores = fuse_rankings([["a"], ["a"]], rrf_k=60)
        self.assertAlmostEqual(scores["a"], 2 / 61)

    def test_weights(self):
        scores = fuse_rankings([["a"], ["b"]], weights=[2.0, 1.0])
        self.assertGreater(scores["a"], scores["b"])


if __name__ == "__main__":
    unittest.main()
