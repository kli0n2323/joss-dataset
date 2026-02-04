#!/usr/bin/env python3
# Copyright (c) 2026.
# SPDX-License-Identifier: MIT

"""
Plot number of opened/closed JOSS issues per year.

Inputs:
- data/derived/joss_submissions.json (from normalize step)

Outputs:
- data/plots/open_issues_per_year.png
- data/plots/closed_issues_per_year.png
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

LOGGER: logging.Logger = logging.getLogger(__name__)


def _unix_to_year(ts: int) -> int:
    """
    Convert unix seconds to UTC year.

    Args:
        ts: UNIX timestamp in seconds.

    Returns:
        The UTC year corresponding to the timestamp.

    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return int(dt.year)


def _load_submissions(path: Path) -> list[dict[str, Any]]:
    """
    Load normalized submissions JSON list.

    Args:
        path: Path to a JSON file containing a top-level list of submissions.

    Returns:
        A list of submission objects (dicts).

    Raises:
        RuntimeError: If the JSON is not a top-level list.

    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = "Expected top-level JSON list of submissions"
        raise RuntimeError(msg)

    return [item for item in data if isinstance(item, dict)]


def _count_years(
    submissions: list[dict[str, Any]],
    key: str,
    *,
    skip_zero: bool,
) -> Counter[int]:
    """
    Count occurrences per UTC year for a given UNIX timestamp key.

    Args:
        submissions: Normalized submissions list.
        key: Field name containing UNIX timestamp seconds (e.g., "Opened", "Closed").
        skip_zero: Whether to skip timestamps equal to 0.

    Returns:
        A Counter mapping year -> count.

    """
    counts: Counter[int] = Counter()

    for sub in submissions:
        ts = sub.get(key)
        if not isinstance(ts, int):
            continue
        if skip_zero and ts == 0:
            continue

        year = _unix_to_year(ts)
        counts[year] += 1

    return counts


def _plot_counts(
    counts: Counter[int],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """
    Plot counts per year and save to PNG.

    Args:
        counts: Counter mapping year -> count.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        out_path: Output PNG path.

    Raises:
        RuntimeError: If there is no data to plot.

    """
    if not counts:
        msg = f"No data to plot for: {title}"
        raise RuntimeError(msg)

    years = sorted(counts.keys())
    values = [counts[y] for y in years]

    fig, ax = plt.subplots()
    ax.bar(years, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args.

    Returns:
        Parsed CLI namespace.

    """
    parser = argparse.ArgumentParser(
        description="Plot opened/closed issues per year (matplotlib)."
    )
    parser.add_argument(
        "--in-file",
        default="data/derived/joss_submissions.json",
        help="Input normalized submissions JSON (from transform step)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/plots",
        help="Directory to write PNG plots",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG/INFO/WARNING/ERROR)",
    )
    return parser.parse_args()


def main() -> int:
    """
    Run the plotter.

    Returns:
        Process exit code.

    Raises:
        RuntimeError: If the input file is missing.

    """
    args = parse_args()

    level_name = str(args.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    in_file = Path(str(args.in_file))
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_file.exists():
        msg = (
            f"Input file not found: {in_file}\n"
            "Run the normalization step first to generate "
            "data/derived/joss_submissions.json"
        )
        raise RuntimeError(msg)

    submissions = _load_submissions(in_file)
    LOGGER.info("Loaded %s submissions from %s", len(submissions), in_file)

    opened_counts = _count_years(submissions, "Opened", skip_zero=False)
    closed_counts = _count_years(submissions, "Closed", skip_zero=True)

    LOGGER.info(
        "Opened years: %s-%s (total=%s)",
        min(opened_counts.keys()),
        max(opened_counts.keys()),
        sum(opened_counts.values()),
    )

    if closed_counts:
        LOGGER.info(
            "Closed years: %s-%s (total=%s)",
            min(closed_counts.keys()),
            max(closed_counts.keys()),
            sum(closed_counts.values()),
        )
    else:
        LOGGER.warning("No closed issues found (Closed==0 everywhere?)")

    open_plot = out_dir / "open_issues_per_year.png"
    closed_plot = out_dir / "closed_issues_per_year.png"

    _plot_counts(
        opened_counts,
        title="Number of issues opened per year (editorialbot)",
        xlabel="Year",
        ylabel="Opened issues",
        out_path=open_plot,
    )
    LOGGER.info("Wrote %s", open_plot)

    if closed_counts:
        _plot_counts(
            closed_counts,
            title="Number of issues closed per year (editorialbot)",
            xlabel="Year",
            ylabel="Closed issues",
            out_path=closed_plot,
        )
        LOGGER.info("Wrote %s", closed_plot)
    else:
        LOGGER.info("Skipping closed-issues plot because there is no data.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
