import os


def get(key: str) -> str:
    return os.getenv(key)
