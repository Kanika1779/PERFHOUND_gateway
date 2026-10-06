"""Gateway command line.

    python -m perfhound.gateway range <link> --good v1.2 --bad main     # ONLY that range, live (range_fetch.py)
    python -m perfhound.gateway range <link>/compare/v1.2...main        # same, as a compare link
    python -m perfhound.gateway <repository link>                       # EVERYTHING into a folder (ingest.py)
    python -m perfhound.gateway login                                   # GitHub token, once (5,000 requests/hour)
    python -m perfhound.gateway https://github.com/psf/requests
    python -m perfhound.gateway https://github.com/psf/requests --limit 200 --out ingest/requests_test
    python -m perfhound.gateway https://github.com/psf/requests --comments
    python -m perfhound.gateway whoami | logout
"""

from __future__ import annotations

import argparse
import sys

TOKEN_COMMANDS = ("login", "whoami", "logout")


def token_command(action: str, token: str | None = None) -> int:
    """GitHub token: login (checked with GitHub, then saved to ~/.perfhound) / whoami / logout."""
    from .github import GitHubClient, GitHubError, delete_token, find_token, mask, save_token

    if action == "login":
        import getpass

        token = (token or getpass.getpass("GitHub token (input hidden): ")).strip()
        if not token:
            print("no token given", file=sys.stderr)
            return 1
        client = GitHubClient(token)
        try:
            user = client.get_json("/user")                     # check BEFORE saving
        except GitHubError as exc:
            print(f"token not saved: {exc}", file=sys.stderr)
            return 1
        path = save_token(token)
        print(f"logged in as {user.get('login')} - token {mask(token)} saved to {path}")
        print(f"requests left this hour: {client.rate_remaining}/{client.rate_limit}")
        return 0
    if action == "logout":
        print("token removed" if delete_token() else "no saved token")
        left, source = find_token()
        if left:
            print(f"note: a token is still set by {source}")
        return 0
    token, source = find_token()                                 # whoami
    client = GitHubClient(token)
    try:
        if not token:
            rate = client.get_json("/rate_limit").get("rate", {})
            print(f"no token (anonymous: {rate.get('remaining')}/{rate.get('limit')} requests left this hour) - "
                  f"`python -m perfhound.gateway login` raises the limit to 5,000")
            return 0
        user = client.get_json("/user")
    except GitHubError as exc:
        print(f"token {mask(token)} from {source}: {exc}" if token else str(exc), file=sys.stderr)
        return 1
    print(f"{user.get('login')}  (token {mask(token)} from {source})")
    print(f"requests left this hour: {client.rate_remaining}/{client.rate_limit}")
    return 0


def range_command(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m perfhound.gateway range",
                                 description="Only the commits between good and bad (+ their PRs and linked issues).")
    ap.add_argument("repo", help="GitHub link, compare link (.../compare/GOOD...BAD) or a folder with a git repo")
    ap.add_argument("--good", help="last fast commit / tag / branch (default: bad's parent)")
    ap.add_argument("--bad", help="first slow commit / tag / branch (default: HEAD)")
    ap.add_argument("--out", default="perfhound-range.json", help="JSON file to write (default perfhound-range.json)")
    ap.add_argument("--summary", help="also write a Markdown summary here (e.g. $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--github-repo", help="owner/name on GitHub (default: from the link, $GITHUB_REPOSITORY or origin)")
    ap.add_argument("--no-github", action="store_true", help="git data only (no PRs / issues)")
    ap.add_argument("--evaluation", action="store_true",
                    help="testing mode: hide GitHub data that appeared after `bad` landed")
    ap.add_argument("--from-merge-base", action="store_true", help="pull requests: start at the merge base of good and bad")
    ap.add_argument("--batch-size", type=int, default=500, help="commits per step (default 500)")
    args = ap.parse_args(argv)

    from .errors import GatewayError
    from .range_fetch import fetch_range, summary_markdown

    try:
        result = fetch_range(args.repo, args.good, args.bad, out=args.out, github=not args.no_github,
                             github_repo=args.github_repo, evaluation=args.evaluation,
                             from_merge_base=args.from_merge_base, batch_size=args.batch_size)
    except GatewayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.summary:
        import os
        # GitHub's job-summary file may already hold other steps' output: add to it; any other file: replace
        mode = "a" if os.path.abspath(args.summary) == os.path.abspath(os.environ.get("GITHUB_STEP_SUMMARY", "-")) else "w"
        with open(args.summary, mode, encoding="utf-8") as fh:
            fh.write(summary_markdown(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):       # Windows consoles: never crash on unusual characters
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "range":
        return range_command(argv[1:])
    if argv and argv[0] in TOKEN_COMMANDS:
        tp = argparse.ArgumentParser(prog=f"python -m perfhound.gateway {argv[0]}")
        tp.add_argument("--token", help="token for login (default: asked for, hidden)")
        return token_command(argv[0], tp.parse_args(argv[1:]).token)

    ap = argparse.ArgumentParser(prog="python -m perfhound.gateway",
                                 description="Repository link -> folder with all commits, PRs and issues (JSON). "
                                             "Also: range (only good..bad, see `range -h`) and login | whoami | logout.")
    ap.add_argument("link", help="e.g. https://github.com/psf/requests (or owner/name)")
    ap.add_argument("--out", help="output folder (default: ingest/<owner>__<repo>)")
    ap.add_argument("--limit", type=int, help="only the newest N commits (quick test); default: all")
    ap.add_argument("--comments", action="store_true", help="also every issue comment (1 GitHub request per issue)")
    ap.add_argument("--no-github", action="store_true", help="git data only (no PRs / issues)")
    ap.add_argument("--batch-size", type=int, default=500, help="commits per batch (default 500)")
    args = ap.parse_args(argv)

    from .errors import GatewayError
    from .ingest import ingest

    try:
        result = ingest(args.link, args.out, limit=args.limit, github=not args.no_github, comments=args.comments,
                        batch_size=args.batch_size)
    except GatewayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0 if result.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
