import unittest
from math import log

from lm import BigramLanguageModel, kLM_ORDER, \
    kUNK_CUTOFF, kNEG_INF, kSTART, kEND, lg


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.lm = BigramLanguageModel(kUNK_CUTOFF, jm_lambda=0.6, \
                                      kn_discount = 0.1,
                                      kn_concentration = 1.0,
                                      add_k=0.1)

    def test_vocab(self):
        self.lm.train_seen("a", 300)
        self.lm.train_seen("b")
        self.lm.train_seen("c")
        self.lm.train_seen('d')
        self.lm.finalize()

        # Infrequent words should look the same
        self.assertEqual(self.lm.vocab_lookup("b"),
                         self.lm.vocab_lookup("c"))

        # Infrequent words should look the same as never seen words
        self.assertEqual(self.lm.vocab_lookup("b"),
                         self.lm.vocab_lookup("d"),
                         "")

        # The frequent word should be different from the infrequent word
        self.assertNotEqual(self.lm.vocab_lookup("a"),
                            self.lm.vocab_lookup("b"))

    def test_censor(self):
        self.lm.train_seen("a", 300)
        self.lm.train_seen("b")
        self.lm.train_seen("c")
        self.lm.train_seen('d')
        self.lm.finalize()

        censored_a = list(self.lm.tokenize_and_censor("a b d"))
        censored_b = list(self.lm.tokenize_and_censor("d b a"))
        censored_c = list(self.lm.tokenize_and_censor("a b d"))
        censored_d = list(self.lm.tokenize_and_censor("b d a"))

        self.assertEqual(censored_a, censored_c)
        self.assertEqual(censored_b, censored_d)

        # Should add start and end tag
        print("#".join(self.lm.tokenize("a b d")))
        print(censored_a)
        self.assertEqual(len(censored_a), 5)
        self.assertEqual(censored_a[0], censored_b[0])
        self.assertEqual(censored_a[-1], censored_b[-1])
        self.assertEqual(censored_a[1], censored_b[3])
        self.assertEqual(censored_a[2], censored_b[1])

    def test_lm(self):
        self.lm.train_seen("a", 4)
        self.lm.train_seen('b',4)
        self.lm.train_seen('c',4)
        self.lm.finalize()
        self.lm.add_train("a a b")
        # Test MLE
        word_start = self.lm.vocab_lookup(kSTART)
        word_end = self.lm.vocab_lookup(kEND)
        word_a = self.lm.vocab_lookup("a")
        word_b = self.lm.vocab_lookup("b")
        word_c = self.lm.vocab_lookup("c")
        self.assertAlmostEqual(self.lm.mle(word_start, word_b), log(0.25))
        self.assertAlmostEqual(self.lm.mle(word_start, word_a), log(0.25))
        self.assertAlmostEqual(self.lm.mle(word_a, word_a), log(0.5))
        self.assertAlmostEqual(self.lm.mle(word_a, word_b), log(0.5))
        self.assertAlmostEqual(self.lm.mle(word_a, word_c), log(0.5))

        # Test Add one
        self.assertAlmostEqual(self.lm.laplace(word_start, word_b),
                               log(4.25))
        self.assertAlmostEqual(self.lm.laplace(word_start, word_a),
                               log(4.25))
        self.assertAlmostEqual(self.lm.laplace(word_a, word_a),
                               log(5.25))
        self.assertAlmostEqual(self.lm.laplace(word_a, word_b),
                               log(5.25))
        self.assertAlmostEqual(self.lm.laplace(word_a, word_c),
                               log(5.25))

        # Test Add k 
        self.assertAlmostEqual(self.lm.add_k(word_start, word_b),
                               0.35)
        self.assertAlmostEqual(self.lm.add_k(word_start, word_a),
                               0.35)
        self.assertAlmostEqual(self.lm.add_k(word_a, word_a),
                               0.6)
        self.assertAlmostEqual(self.lm.add_k(word_a, word_b),
                               0.6)
        self.assertAlmostEqual(self.lm.add_k(word_a, word_c),
                               0.6)
        # Test Kneser-Ney
        k=-0.691654876777717
        self.assertAlmostEqual(self.lm.kneser_ney(word_start, word_a),
                               k)
        self.assertAlmostEqual(self.lm.kneser_ney(word_start, word_b),
                               k)

        # Test Jelinek Mercer
        self.assertAlmostEqual(self.lm.jelinek_mercer(word_start, word_a),
                               log(0.15))

if __name__ == '__main__':
    unittest.main()