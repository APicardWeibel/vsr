import random
import string


def random_name(k: int = 30):
    """Generate a random name for a temporary file/folder"""
    return "".join(random.choices(string.ascii_lowercase, k=k))
