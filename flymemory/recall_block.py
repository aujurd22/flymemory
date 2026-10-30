"""Budget-aware recall packing (dsh-flymemory adaptation).

The measured k-sweep (LongMemEval oracle, 2026-10-30) is monotone in how much
history reaches the reader: top-5 40.0%, top-10 44.2%, top-20 49.2%, top-50 56.7%.
So a fixed entry count is the wrong knob -- a *character budget* is: fill the
prompt with as many full entries as fit, best first, and stop.

Pure here so it can be tested without torch, the model, or the server.
"""


def pack(entries, budget_chars=6000, per_entry_chars=600):
    """Lay out entries best-first until the budget is spent.

    `entries` is an iterable of (label, text): `label` already carries the id and
    metadata line. Returns (block, stats) where stats holds how many entries were
    packed, how many characters were used, and how many were left out.
    """
    budget_chars = max(200, int(budget_chars))
    per_entry_chars = max(80, int(per_entry_chars))
    blocks, used, taken = [], 0, 0
    total = 0
    for label, text in entries:
        total += 1
        body = (text or "").strip()
        if len(body) > per_entry_chars:
            body = body[: per_entry_chars - 1] + "…"
        piece = f"{label}\n{body}"
        if used + len(piece) + 2 > budget_chars:
            continue
        blocks.append(piece)
        used += len(piece) + 2
        taken += 1
    header = f"[memory block: {taken} entries, {used} chars]"
    return "\n\n".join([header] + blocks), {"taken": taken, "used": used,
                                            "skipped": total - taken}


def top_sim(line):
    """Pull the similarity out of a `[#12 | 0.83 | ...]` line, or None."""
    import re
    match = re.search(r"\[#(\d+)\s*\|\s*([0-9.]+)", line or "")
    return float(match.group(2)) if match else None
