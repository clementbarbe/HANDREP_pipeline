"""
Pretty console output — no dependency on logging.
"""


def header(title, width=60):
    """Print a prominent header block."""
    print()
    print("═" * width)
    print(f"  {title}")
    print("═" * width)


def divider(char="─", width=60):
    print(char * width)


def step(number, total, message):
    """Print a numbered step header."""
    print(f"\n[{number}/{total}] {message}")


def info(message):
    print(f"      → {message}")


def success(message):
    print(f"      ✓ {message}")


def warning(message):
    print(f"      ⚠ {message}")


def error(message):
    print(f"      ✗ {message}")