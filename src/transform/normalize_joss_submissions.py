#!/usr/bin/env python3
# Copyright (c) 2026.
# SPDX-License-Identifier: MIT

"""Normalize raw GitHub issue JSON files into a single JOSS submissions JSON file."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.transform.joss_submission import from_github_issue_payload

LOGGER: logging.Logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file from disk.

    Returns:
        The decoded JSON object.

    """
    return json.loads(path.read_text(encoding="utf-8"))


def _labels_from_issue(issue: dict[str, Any]) -> list[str]:
    """
    Extract label names from a GitHub issue payload.

    Returns:
        A list of label names.

    """
    labels_obj = issue.get("labels", [])
    if not isinstance(labels_obj, list):
        return []

    names: list[str] = []
    for item in labels_obj:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        Parsed CLI args.

    """
    parser = argparse.ArgumentParser(
        description="Normalize JOSS issue bodies into one JSON file."
    )
    parser.add_argument(
        "--in-dir",
        default="data/raw/openjournals_joss-reviews/issues",
        help="Directory containing issue_<N>.json files from ingest step",
    )
    parser.add_argument(
        "--out-file",
        default="data/derived/joss_submissions.json",
        help="Output JSON file (list of normalized submissions)",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)"
    )
    return parser.parse_args()


def _normalize_dir(in_dir: Path) -> tuple[list[dict[str, Any]], int, int]:
    """
    Normalize all issues in `in_dir` into submission records.

    Returns:
        A tuple of (submissions, skipped_count, failed_count).

    """
    issue_files = sorted(in_dir.glob("issue_*.json"))
    LOGGER.info("Found %s issue files in %s", len(issue_files), in_dir)

    submissions: list[dict[str, Any]] = []
    skipped = 0
    failed = 0

    for path in issue_files:
        raw = _load_json(path)

        issue = raw.get("issue")
        if not isinstance(issue, dict):
            skipped += 1
            continue

        body = issue.get("body")
        if not isinstance(body, str) or "<!--author-handle-->" not in body:
            skipped += 1
            continue

        number = issue.get("number")
        created_at = issue.get("created_at")
        closed_at = issue.get("closed_at")

        if not isinstance(number, int) or not isinstance(created_at, str):
            skipped += 1
            continue

        labels = _labels_from_issue(issue)

        try:
            submission = from_github_issue_payload(
                issue_number=number,
                body=body,
                created_at=created_at,
                closed_at=closed_at if isinstance(closed_at, str) else None,
                labels=labels,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            LOGGER.warning("Failed to parse %s: %s", path.name, exc)
            continue

        submissions.append(submission.model_dump(by_alias=True))

    return submissions, skipped, failed


def main() -> int:
    """
    Run the normalization routine.

    Raises:
        RuntimeError: If the input directory does not exist.

    Returns:
        Process exit code (0 for success).

    """
    args = parse_args()
    level_name = str(args.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    in_dir = Path(str(args.in_dir))
    if not in_dir.exists():
        msg = f"Input dir does not exist: {in_dir}"
        raise RuntimeError(msg)

    out_file = Path(str(args.out_file))
    out_file.parent.mkdir(parents=True, exist_ok=True)

    submissions, skipped, failed = _normalize_dir(in_dir)

    out_file.write_text(
        json.dumps(submissions, indent=2, sort_keys=True), encoding="utf-8"
    )
    LOGGER.info(
        "Wrote %s normalized submissions to %s (skipped=%s failed=%s).",
        len(submissions),
        out_file,
        skipped,
        failed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
