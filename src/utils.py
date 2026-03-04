import os
import re

def get_group_id(filename):
    """
    Extracts the patient/sample ID from the filename.
    Files with the same Group ID belong to the same biological sample.
    """
    fname = os.path.basename(filename)

    # 1. remove spot / replicate suffix (_0_F9_1 etc.)
    # Note: Using the user's specific logic + robustness
    # The user suggested: fname = fname.split("_")[0]
    # But files like "2_0_C6_1.txt" -> "2".
    parts = fname.split("_")
    if len(parts) > 1:
        fname = parts[0]

    # 2. remove parentheses content (instrument / volume info)
    fname = re.sub(r"\(.*?\)", "", fname)

    # 3. normalize separators
    fname = fname.replace("+", " ").replace("-", " ")

    # collapse whitespace
    fname = re.sub(r"\s+", " ", fname).strip()

    # -------------------------
    # RULE A: leading pure numeric ID (e.g. 30141, 21)
    # -------------------------
    m = re.match(r"^(\d{2,6})\b", fname, re.IGNORECASE)
    if m:
        return m.group(1)

    # -------------------------
    # RULE B: date-based sample (YYYYMMDD Sample X)
    # -------------------------
    m = re.match(r"^(\d{8}\s+TB\s+Sample\s+\d+)", fname, re.IGNORECASE)
    if m:
        return m.group(1)

    # -------------------------
    # RULE C: reject %-only or instrument-only names
    # -------------------------
    if re.fullmatch(r"[\d.%]+", fname):
        return f"UNKNOWN_{fname}"

    # -------------------------
    # RULE D: fallback – first meaningful phrase
    # -------------------------
    tokens = fname.split(" ")
    if not tokens:
        return "UNKNOWN"
    return " ".join(tokens[:4])
