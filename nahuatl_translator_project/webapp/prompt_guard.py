"""Prompt injection detection and input sanitization.

Scans user-supplied text for patterns that attempt to override system prompts,
extract internal instructions, or hijack the model's behavior. Returns a
risk assessment that the calling code can use to reject or flag the input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class ScanResult:
    """Result of scanning user input for prompt injection."""
    is_suspicious: bool
    risk_level: str  # "none", "low", "high"
    matched_patterns: List[str]

    @property
    def should_block(self) -> bool:
        return self.risk_level == "high"


# --- High-risk patterns: direct prompt override attempts ---
# These are strong signals of injection — block the request.
_HIGH_RISK_PATTERNS = [
    # System/role impersonation
    (r"(?i)\bignore\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions?|prompts?|rules?|guidelines?)", "ignore-previous-instructions"),
    (r"(?i)\bdisregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)", "disregard-instructions"),
    (r"(?i)\bforget\s+(all\s+)?(previous|above|prior|your)\s+(instructions?|prompts?|rules?)", "forget-instructions"),
    (r"(?i)\byou\s+are\s+now\s+(a|an|the)\b", "role-override"),
    (r"(?i)\bact\s+as\s+(a|an|if)\b", "role-override"),
    (r"(?i)\bpretend\s+(you\s+are|to\s+be|you're)\b", "role-override"),
    (r"(?i)\bswitch\s+to\s+(a\s+)?new\s+(role|mode|persona)\b", "role-switch"),

    # System prompt extraction
    (r"(?i)\b(show|reveal|repeat|print|output|display)\s+\w*\s*(your|the)\s+(system\s+)?(prompt|instructions?|rules?|guidelines?)\b", "prompt-extraction"),
    (r"(?i)\btell\s+me\s+(your|the)\s+(system\s+)?(prompt|instructions?|rules?|guidelines?)\b", "prompt-extraction"),
    (r"(?i)\bwhat\s+(are|were)\s+your\s+(system\s+)?(instructions?|prompts?|rules?)\b", "prompt-extraction"),

    # Direct prompt injection markers
    (r"(?i)\[system\]", "system-tag-injection"),
    (r"(?i)<\s*system\s*>", "system-tag-injection"),
    (r"(?i)\bsystem\s*:\s*you\s+(are|should|must|will)\b", "system-role-injection"),
    (r"(?i)\bnew\s+instructions?\s*:", "new-instructions"),
    (r"(?i)\boverride\s+(mode|instructions?|rules?)\b", "override-attempt"),

    # Output manipulation
    (r"(?i)\bdo\s+not\s+translate\b.*\binstead\b", "output-hijack"),
    (r"(?i)\binstead\s+of\s+translat(ing|e|ion)\b", "output-hijack"),
]

# --- Low-risk patterns: suspicious but could be legitimate ---
# These are flagged but not blocked — could appear in real Nahuatl/academic text.
_LOW_RISK_PATTERNS = [
    (r"(?i)\bignore\b.*\brules?\b", "ignore-rules-mention"),
    (r"(?i)\bprompt\b.*\binjection\b", "prompt-injection-mention"),
    (r"(?i)\bjailbreak\b", "jailbreak-mention"),
]


def scan_input(text: str) -> ScanResult:
    """Scan user input text for prompt injection patterns.

    Returns a ScanResult with risk assessment.
    """
    if not text or not text.strip():
        return ScanResult(is_suspicious=False, risk_level="none", matched_patterns=[])

    matched: List[str] = []
    risk = "none"

    # Check high-risk patterns first
    for pattern, label in _HIGH_RISK_PATTERNS:
        if re.search(pattern, text):
            matched.append(label)
            risk = "high"

    # Only check low-risk if no high-risk matches
    if risk == "none":
        for pattern, label in _LOW_RISK_PATTERNS:
            if re.search(pattern, text):
                matched.append(label)
                risk = "low"

    return ScanResult(
        is_suspicious=len(matched) > 0,
        risk_level=risk,
        matched_patterns=matched,
    )


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Basic input sanitization — length limit and null byte removal.

    Does NOT strip legitimate content. Only removes characters that
    should never appear in translation/transcription input.
    """
    # Remove null bytes
    text = text.replace("\x00", "")
    # Length cap
    if len(text) > max_length:
        text = text[:max_length]
    return text
