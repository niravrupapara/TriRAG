import os
import tempfile
import unittest


class EngineHelperTests(unittest.TestCase):
    def test_calculate_file_hash_changes_when_file_content_changes(self):
        os.environ.setdefault("MISTRAL_API_KEY", "test-key")

        from src.engine import calculate_file_hash

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            tmp.write("first")
            path = tmp.name

        try:
            first_hash = calculate_file_hash(path)

            with open(path, "w", encoding="utf-8") as f:
                f.write("second")

            second_hash = calculate_file_hash(path)

            self.assertNotEqual(first_hash, second_hash)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
