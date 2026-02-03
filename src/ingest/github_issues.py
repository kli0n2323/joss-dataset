# Copyright (c) 2026.
# SPDX-License-Identifier: MIT

"""
Collect issues opened by `editorialbot` from `openjournals/joss-reviews`.

This script queries the GitHub REST API for issues (open + closed) from the
`openjournals/joss-reviews` repository, filters to those opened by the
`editorialbot` account, and writes each issue to an individual JSON file for
downstream analysis.

Authentication:
- Provide a GitHub Personal Access Token via the `GITHUB_TOKEN` env var.

Pagination:
- Uses `per_page=100` by default.

Output:
- data/raw/openjournals_joss-reviews/issues/issue_<N>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import requests

API_BASE: str = "https://api.github.com"
GITHUB_API_VERSION: str = "2022-11-28"

LOGGER: logging.Logger = logging.getLogger(__name__)

EMPTY: str = ""
HTTP_OK: int = 200
HTTP_FORBIDDEN: int = 403
PER_PAGE_REQUIRED: int = 100

JsonObject = dict[str, object]
JsonList = list[JsonObject]


@dataclass(frozen=True)
class RepoTarget:
    """A GitHub repository identifier."""

    owner: str
    repo: str

    def full_name(self) -> str:
        """
        Return the repository in 'owner/repo' form.

        Returns:
            The repository in 'owner/repo' form.

        """
        return f"{self.owner}/{self.repo}"


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the ingestion script."""

    token: str
    target: RepoTarget
    bot: str
    out_dir: Path
    state: str
    per_page: int
    overwrite: bool
    max_pages: int | None


def utc_now_iso() -> str:
    """
    Return the current UTC time in ISO-8601 format (seconds precision).

    Returns:
        The current UTC time in ISO-8601 format (seconds precision).

    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_token() -> str:
    """
    Read `GITHUB_TOKEN` from the environment.

    Returns:
        The GitHub token read from the `GITHUB_TOKEN` environment variable.

    Raises:
        RuntimeError: If `GITHUB_TOKEN` is missing/empty.

    """
    token: str = os.environ.get("GITHUB_TOKEN", EMPTY).strip()
    if not token:
        msg = (
            "Missing GITHUB_TOKEN environment variable.\n"
            "Set it before running, e.g.:\n"
            "  export GITHUB_TOKEN='ghp_...'\n"
            "or (PowerShell):\n"
            '  setx GITHUB_TOKEN "ghp_..."'
        )
        raise RuntimeError(msg)
    return token


def build_headers(token: str) -> dict[str, str]:
    """
    Build GitHub REST API headers for authenticated requests.

    Args:
        token: A GitHub personal access token.

    Returns:
        A dictionary of headers for GitHub REST API requests.

    """
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "joss-dataset-ingest",
    }


def rate_limit_status(headers: requests.structures.CaseInsensitiveDict[str]) -> str:
    """
    Format GitHub rate limit headers for logging.

    Args:
        headers: Response headers from GitHub.

    Returns:
        A human-readable rate limit status string.

    """
    remaining: str | None = headers.get("X-RateLimit-Remaining")
    limit: str | None = headers.get("X-RateLimit-Limit")
    reset: str | None = headers.get("X-RateLimit-Reset")

    if remaining is None or limit is None:
        return "rate-limit: unknown"

    if reset is None:
        return f"rate-limit: {remaining}/{limit}"

    try:
        reset_dt = datetime.fromtimestamp(int(reset), tz=timezone.utc)
    except ValueError:
        return f"rate-limit: {remaining}/{limit} (reset parse error)"

    return f"rate-limit: {remaining}/{limit} (resets {reset_dt.isoformat()})"


def sleep_until_reset(resp: requests.Response) -> None:
    """
    Sleep until GitHub's rate limit reset if the response indicates limiting.

    Args:
        resp: A GitHub API response.

    """
    remaining: str | None = resp.headers.get("X-RateLimit-Remaining")
    reset: str | None = resp.headers.get("X-RateLimit-Reset")

    if resp.status_code != HTTP_FORBIDDEN or remaining != "0" or reset is None:
        return

    reset_ts = int(reset)
    now_ts = int(time.time())
    sleep_for = max(0, reset_ts - now_ts) + 5

    reset_dt = datetime.fromtimestamp(reset_ts, tz=timezone.utc)
    LOGGER.warning("Rate limited. Sleeping %ss until %s.", sleep_for, reset_dt.isoformat())
    time.sleep(sleep_for)


def fetch_issues_page(
    session: requests.Session,
    target: RepoTarget,
    *,
    page: int,
    per_page: int,
    state: str,
) -> JsonList:
    """
    Fetch a single page of issues/PRs from GitHub.

    Note: GitHub's /issues endpoint can include pull requests. We keep the raw
    objects and filter later to avoid discarding data prematurely.

    Args:
        session: A configured requests session.
        target: Repository identifier.
        page: The page number to fetch (1-indexed).
        per_page: Items per page (GitHub max is 100).
        state: Issue state filter ("open", "closed", or "all").

    Returns:
        A list of issue/PR JSON objects.

    Raises:
        RuntimeError: If the GitHub API returns a non-200 response or an
            unexpected JSON payload type.

    """
    url: str = f"{API_BASE}/repos/{target.owner}/{target.repo}/issues"
    params: dict[str, object] = {
        "state": state,
        "per_page": per_page,
        "page": page,
        "sort": "created",
        "direction": "desc",
    }

    resp = session.get(url, params=params, timeout=30)
    if resp.status_code == HTTP_FORBIDDEN:
        sleep_until_reset(resp)
        resp = session.get(url, params=params, timeout=30)

    if resp.status_code != HTTP_OK:
        msg = (
            f"GitHub API error {resp.status_code} for {resp.url}\n"
            f"Response (first 500 chars): {resp.text[:500]}"
        )
        raise RuntimeError(msg)

    LOGGER.info("Fetched page %s (%s). %s", page, target.full_name(), rate_limit_status(resp.headers))

    data = resp.json()
    if not isinstance(data, list):
        err_msg = "Unexpected response type: expected list"
        raise RuntimeError(err_msg)

    return cast(JsonList, data)


def opened_by_login(issue: JsonObject, login: str) -> bool:
    """
    Return True if the issue object was opened by `login`.

    Args:
        issue: An issue JSON object.
        login: GitHub username/login to match.

    Returns:
        True if the issue was opened by `login`, otherwise False.

    """
    user_obj = issue.get("user")
    if not isinstance(user_obj, dict):
        return False
    user_login = user_obj.get("login")
    return isinstance(user_login, str) and user_login == login


def issue_number(issue: JsonObject) -> int | None:
    """
    Extract the issue number if present.

    Args:
        issue: An issue JSON object.

    Returns:
        The issue number if present; otherwise None.

    """
    number = issue.get("number")
    if isinstance(number, int):
        return number
    return None


def write_issue_file(
    issue: JsonObject,
    *,
    out_dir: Path,
    repo_full_name: str,
    overwrite: bool,
) -> bool:
    """
    Write a single issue object to disk.

    Args:
        issue: An issue JSON object.
        out_dir: Output directory for issue JSON files.
        repo_full_name: Repository name in 'owner/repo' format.
        overwrite: Whether to overwrite an existing issue file.

    Returns:
        True if written; False if skipped.

    """
    number = issue_number(issue)
    if number is None:
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"issue_{number}.json"

    if out_path.exists() and not overwrite:
        return False

    payload: JsonObject = {
        "source": "github",
        "repo": repo_full_name,
        "fetched_at": utc_now_iso(),
        "issue": issue,
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        The parsed CLI arguments namespace.

    """
    parser = argparse.ArgumentParser(description="Collect editorialbot issues from joss-reviews.")
    parser.add_argument("--owner", default="openjournals", help="GitHub owner/org")
    parser.add_argument("--repo", default="joss-reviews", help="GitHub repo")
    parser.add_argument("--bot", default="editorialbot", help="User login to filter")
    parser.add_argument(
        "--out-dir",
        default="data/raw/openjournals_joss-reviews/issues",
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--state",
        default="all",
        choices=["open", "closed", "all"],
        help="Issue state filter",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=PER_PAGE_REQUIRED,
        help="Items per page (default 100)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to fetch (for testing). Default: no limit.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing issue files")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """
    Build a Config instance from parsed CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A Config instance for the run.

    """
    token = get_token()
    target = RepoTarget(owner=str(args.owner), repo=str(args.repo))
    out_dir = Path(str(args.out_dir))

    return Config(
        token=token,
        target=target,
        bot=str(args.bot),
        out_dir=out_dir,
        state=str(args.state),
        per_page=int(args.per_page),
        overwrite=bool(args.overwrite),
        max_pages=args.max_pages,
    )


def main() -> int:
    """
    Run the ingestion routine.

    Returns:
        Exit code (0 for success).

    """
    args = parse_args()

    level_name = str(args.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    config = build_config(args)
    if config.per_page != PER_PAGE_REQUIRED:
        LOGGER.warning("Project requirement is per_page=100. You set %s.", config.per_page)

    session = requests.Session()
    session.headers.update(build_headers(config.token))

    page = 1
    total_fetched = 0
    total_bot = 0
    total_written = 0

    LOGGER.info("Starting collection for %s (bot=%s).", config.target.full_name(), config.bot)
    LOGGER.info("Output directory: %s", config.out_dir.resolve())

    while True:
        issues = fetch_issues_page(
            session,
            config.target,
            page=page,
            per_page=config.per_page,
            state=config.state,
        )
        if issues == []:
            break

        total_fetched += len(issues)

        bot_issues = [issue for issue in issues if opened_by_login(issue, config.bot)]
        total_bot += len(bot_issues)

        written_this_page = 0
        for issue in bot_issues:
            if write_issue_file(
                issue,
                out_dir=config.out_dir,
                repo_full_name=config.target.full_name(),
                overwrite=config.overwrite,
            ):
                total_written += 1
                written_this_page += 1

        LOGGER.info(
            "Page %s: fetched=%s bot_issues=%s written=%s (total_written=%s)",
            page,
            len(issues),
            len(bot_issues),
            written_this_page,
            total_written,
        )

        if len(issues) < config.per_page:
            break

        if config.max_pages is not None and page >= config.max_pages:
            LOGGER.info("Reached max-pages=%s; stopping early.", config.max_pages)
            break

        page += 1

    LOGGER.info(
        "Done. total_fetched=%s total_bot=%s total_written=%s",
        total_fetched,
        total_bot,
        total_written,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
