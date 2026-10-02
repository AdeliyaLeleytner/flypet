"""Conservative evaluation-only extraction of a response's stated direction.

This module is never used to generate or alter a live reply. Temporal change is
not accepted as an answer about current absolute dominance, or vice versa.
"""

import re


def stated_direction(text, mode="current"):
    s = " ".join(text.lower().replace("–", "-").split())
    temporal = bool(re.search(r"\b(change|changed|shift|shifted|reference|earlier|previous)\b", s))
    if mode == "current" and temporal:
        # An explicit conclusion about the present can be scored separately.
        match = re.search(r"\bso (?:the )?current (.+)", s)
        if not match:
            return None
        s = match.group(1)
    elif mode == "comparison" and "current" in s and not temporal:
        return None
    if mode == "comparison" and re.search(r"\bno (?:resolved )?change\b", s):
        return 0
    if re.search(r"\b(balanced|neither)\b", s):
        return 0
    if re.search(
        r"\bno (?:clear |resolved )?(?:approach or avoidance output|mbon output spikes|output signal|resolved output)\b",
        s,
    ):
        return 0
    if (
        re.search(r"\bno (?:relative |resolved |clear )?(?:strength|tendency)\b", s)
        or "no tendency is stronger" in s
    ):
        return 0
    if re.search(r"\b(unclear|unresolved|not determined|cannot tell|not stronger)\b", s):
        return None
    found = []
    found.extend(
        re.findall(
            r"\b(?:toward|towards|favor of|favors|favours|dominated by|closer to)\s+(approach|avoidance)\b",
            s,
        )
    )
    found.extend(
        re.findall(
            r"\b(approach|avoidance)(?:[- ]related)?(?: (?:tendency|outputs?|response|activity))? (?:is |are |appears |seems )?(?:stronger|dominant|dominates|prevails)\b",
            s,
        )
    )
    values = {1 if x == "approach" else -1 for x in found}
    if len(values) == 1:
        return values.pop()
    if len(values) > 1:
        return None
    # A one-word categorical answer is also a complete answer to these tasks.
    if s.strip(" .!") == "approach":
        return 1
    if s.strip(" .!") == "avoidance":
        return -1
    return None
