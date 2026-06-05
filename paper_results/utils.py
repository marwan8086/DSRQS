# =============================================================================
# Safe Print Utils for Windows Console Compatibility
# =============================================================================
import sys


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
        print(safe_text)
