"""
Small formatting helpers shared by the analysis scripts.

`effort_table` renders a median-effort table in either orientation:

  orient="horizontal"  frameworks across the columns, grouped under each metric,
                       one row per item -- for suites with many items (Q&A).
  orient="vertical"    frameworks down the rows within a per-item block, metrics
                       across the columns -- for suites with few items (operate-
                       and-profile, new-features).

In both orientations the best (lowest, since lower = more efficient) framework
per item x metric is bolded.
"""

RULE = {"l": ":--", "r": "--:", "c": ":-:"}
SEP = "│"


def format_k(n):
    """Compact human-readable number: 5328 -> '5.3K', 1754121 -> '1.8M', 42 -> '42'."""
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


def markdown_table(headers, rows, aligns=None):
    """
    Render a GitHub-flavored markdown table from a single header row.

    headers : list[str]
    rows    : list[list[str]]   already-formatted cells
    aligns  : list['l'|'r'|'c'] | None  default first column left, rest right.
    """
    if aligns is None:
        aligns = ["l"] + ["r"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join(RULE[a] for a in aligns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def effort_table(
    items,
    frameworks,
    metrics,
    raw_of,
    format_cell,
    orient="horizontal",
    item_header="#",
    lower_is_better=True,
    group_sep=True,
):
    """
    Render an effort table (see module docstring for orientations).

    items      : list[(item_key, item_label)]     rows / row-blocks
    frameworks : list[(fw_key, fw_label)]          column / row order
    metrics    : list[(metric_key, metric_label)]  metric order
    raw_of(item_key, fw_key, metric_key) -> float | None   raw median (None = missing)
    format_cell(metric_key, value) -> str             formats a present value
    item_header: left-column title ('#' for Q&A, 'Task' for operate-and-profile)
    group_sep  : horizontal only -- divider column between metric groups.
    """
    fw_keys = [k for k, _ in frameworks]
    fw_labels = [l for _, l in frameworks]
    m_keys = [k for k, _ in metrics]
    m_labels = [l for _, l in metrics]

    def winners(item_key, metric_key):
        """Frameworks tied for best (lowest, or highest) on this item x metric."""
        vals = [(fw, raw_of(item_key, fw, metric_key)) for fw in fw_keys]
        present = [v for _, v in vals if v is not None]
        if len(present) < 2:
            return set()
        best = min(present) if lower_is_better else max(present)
        return {fw for fw, v in vals if v is not None and v == best}

    def cell(item_key, fw, metric_key, win):
        """Formatted value for one framework, bolded when a winner; '–' if missing."""
        v = raw_of(item_key, fw, metric_key)
        if v is None:
            return "–"
        s = format_cell(metric_key, v)
        return f"**{s}**" if fw in win else s

    if orient == "horizontal":
        span = len(fw_keys)
        top, sub, aligns = [item_header], [""], ["l"]
        for gi, m_label in enumerate(m_labels):
            if gi and group_sep:
                top.append(SEP)
                sub.append(SEP)
                aligns.append("c")
            group = [""] * span
            group[span // 2] = m_label
            top.extend(group)
            sub.extend(fw_labels)
            aligns.extend(["r"] * span)
        lines = [
            "| " + " | ".join(top) + " |",
            "|" + "|".join(RULE[a] for a in aligns) + "|",
            "| " + " | ".join(sub) + " |",
        ]
        for item_key, item_label in items:
            row = [item_label]
            for gi, metric_key in enumerate(m_keys):
                if gi and group_sep:
                    row.append(SEP)
                win = winners(item_key, metric_key)
                row.extend(cell(item_key, fw, metric_key, win) for fw in fw_keys)
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    headers = [item_header, "Framework", SEP] + m_labels
    aligns = ["l", "r", "c"] + ["r"] * len(m_keys)
    rows = []
    for item_key, item_label in items:
        win_by_metric = {mk: winners(item_key, mk) for mk in m_keys}
        for i, fw in enumerate(fw_keys):
            label = item_label if i == 0 else ""
            row = [label, fw_labels[i], SEP]
            row.extend(cell(item_key, fw, mk, win_by_metric[mk]) for mk in m_keys)
            rows.append(row)
    return markdown_table(headers, rows, aligns)


def ordinal(n):
    """Ordinal label for a 1-based index: 1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11 -> '11th'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def attempt_records_table(
    items,
    frameworks,
    metrics,
    attempts_of,
    format_cell,
    item_header="Task",
):
    """
    Render a per-attempt records table (paper appendix Table 11 style): one row per
    (item, framework, attempt) carrying the raw per-attempt value rather than a median.
    Leading cells are blanked as they repeat down each item and framework block.

    items      : list[(item_key, item_label)]      row-block order
    frameworks : list[(fw_key, fw_label)]          framework order within a block
    metrics    : list[(metric_key, metric_label)]  metric column order
    attempts_of(item_key, fw_key) -> list[dict metric_key -> value]   (attempt order)
    format_cell(metric_key, value) -> str          formats one raw value
    """
    m_keys = [k for k, _ in metrics]
    m_labels = [l for _, l in metrics]
    headers = [item_header, "Framework", "Attempt", SEP] + m_labels
    aligns = ["l", "r", "r", "c"] + ["r"] * len(m_keys)
    rows = []
    for item_key, item_label in items:
        item_shown = False
        for fw_key, fw_label in frameworks:
            attempts = attempts_of(item_key, fw_key)
            for index, attempt in enumerate(attempts):
                row = [
                    "" if item_shown else item_label,
                    fw_label if index == 0 else "",
                    ordinal(index + 1),
                    SEP,
                ]
                row.extend(format_cell(mk, attempt[mk]) for mk in m_keys)
                rows.append(row)
                item_shown = True
    return markdown_table(headers, rows, aligns)
