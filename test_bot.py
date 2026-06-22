import unittest

from bot import split_message


class SplitMessageTests(unittest.TestCase):
    def test_short_message_is_unchanged(self) -> None:
        self.assertEqual(split_message("hello"), ["hello"])

    def test_long_message_respects_limit(self) -> None:
        chunks = split_message("word " * 100, limit=40)
        self.assertTrue(all(len(chunk) <= 40 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), ("word " * 100).split())

    def test_empty_message_has_fallback(self) -> None:
        self.assertEqual(split_message(""), ["I couldn't produce a response."])


if __name__ == "__main__":
    unittest.main()
