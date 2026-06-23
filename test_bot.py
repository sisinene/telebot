import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from bot import (
    build_memory_context,
    clear_memory,
    count_memory,
    get_recent_memory,
    get_relevant_memory,
    init_memory_db,
    save_memory,
    split_message,
)


class SplitMessageTests(unittest.TestCase):
    def test_short_message_is_unchanged(self) -> None:
        self.assertEqual(split_message("hello"), ["hello"])

    def test_long_message_respects_limit(self) -> None:
        chunks = split_message("word " * 100, limit=40)
        self.assertTrue(all(len(chunk) <= 40 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), ("word " * 100).split())

    def test_empty_message_has_fallback(self) -> None:
        self.assertEqual(split_message(""), ["I couldn't produce a response."])


class MemoryTests(unittest.TestCase):
    def test_memory_is_saved_and_loaded_from_sqlite(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            init_memory_db(db_path)

            save_memory(100, 200, "user", "my favorite color is purple", db_path)
            save_memory(100, 200, "assistant", "I will remember purple.", db_path)

            self.assertEqual(count_memory(100, 200, db_path), 2)
            self.assertEqual(
                get_recent_memory(100, 200, db_path=db_path),
                [
                    {"id": 1, "role": "user", "content": "my favorite color is purple"},
                    {"id": 2, "role": "assistant", "content": "I will remember purple."},
                ],
            )

    def test_relevant_memory_finds_older_matching_messages(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            init_memory_db(db_path)

            save_memory(100, 200, "user", "I live in Almaty and like chess", db_path)
            save_memory(100, 200, "assistant", "Got it.", db_path)
            save_memory(100, 200, "user", "What city did I mention?", db_path)

            relevant = get_relevant_memory(
                100,
                200,
                "tell me about Almaty",
                excluded_ids={3},
                db_path=db_path,
            )

            self.assertEqual(relevant[0]["content"], "I live in Almaty and like chess")

    def test_clear_memory_only_deletes_that_chat_user(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            init_memory_db(db_path)

            save_memory(100, 200, "user", "delete me", db_path)
            save_memory(100, 201, "user", "keep me", db_path)

            self.assertEqual(clear_memory(100, 200, db_path), 1)
            self.assertEqual(count_memory(100, 200, db_path), 0)
            self.assertEqual(count_memory(100, 201, db_path), 1)

    def test_context_includes_relevant_memory_block(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            init_memory_db(db_path)

            save_memory(100, 200, "user", "My cat's name is Pixel.", db_path)
            save_memory(100, 200, "assistant", "Pixel is a great name.", db_path)
            for index in range(35):
                save_memory(100, 200, "user", f"filler message {index}", db_path)
            save_memory(100, 200, "user", "What is my cat named?", db_path)

            context = build_memory_context(100, 200, "cat named Pixel", db_path)

            self.assertEqual(context[0]["role"], "system")
            self.assertIn("Pixel", context[0]["content"])
            self.assertEqual(context[-1]["content"], "What is my cat named?")


if __name__ == "__main__":
    unittest.main()
