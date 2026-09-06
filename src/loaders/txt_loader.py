from .base import BaseLoader


class TXTLoader(BaseLoader):

    def load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()