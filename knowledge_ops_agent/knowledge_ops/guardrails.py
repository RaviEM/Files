import re

from .config import settings
INJECTION PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"you are now", r"system prompt", r"developer mode", r"jailbreak",
]
PII PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b(?:\d[-]*?){13,16}\b",
]

def validate(query: str) -> tuple[bool, str | None]:
    q = query.strip()
    if not q:
        return False, "Empty query."
    if len(q) > settings.max_query_chars:
        return False, f"Query exceeds {settings.max_query_chars} characters."
    for pat in INJECTION_PATTERNS:
        if re.search(pat, q, re.I):
            return False, "Query looks like a prompt-injection attempt."
    for pat in PII_PATTERNS:
        if re.search(pat, q):
            return False, "Query appears to contain sensitive identifiers."
    return True, None