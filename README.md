# Perfhound gateway

Connects the Perfhound system to GitHub: give it a repository and two points in history
(good = still fast, bad = slow) and it fetches every commit in between - message, author, dates,
files, diff, changed functions - with the pull request behind each commit and the issues that
PR closes. Runs from the command line or as a GitHub Action.

## Setup (Windows PowerShell, inside this folder)

    py -3 -m venv .venv
    .venv\Scripts\python -m pip install -e ".[dev]"
    .venv\Scripts\python -m pytest
    .venv\Scripts\python -m perfhound.gateway login        # GitHub token, once

Needs git on PATH.

## Fetch a range (what real users need)

    .venv\Scripts\python -m perfhound.gateway range https://github.com/psf/requests --good v2.31.0 --bad v2.32.0
    .venv\Scripts\python -m perfhound.gateway range https://github.com/psf/requests/compare/v2.31.0...v2.32.0

Writes `perfhound-range.json` (`--out` to change, `--summary report.md` for a readable table).
Only that range is processed. The repository is cloned once (history only, file contents on
demand) and refreshed when you ask for a branch name like `main`.

- `--no-github`: git data only.
- `--evaluation`: testing mode, hides GitHub data that appeared after `bad` (e.g. a later
  "slow since #123" issue). Real use keeps it - it is a useful clue.
- With a token, PRs and issues come 50 commits per request (GraphQL); without one,
  1 request per commit and only 60 requests/hour.

## As a GitHub Action

1. Copy the two workflow files into place (PowerShell, in this folder):

       mkdir .github\workflows -Force
       copy examples\perfhound-self-test.yml .github\workflows\perfhound.yml
       copy examples\tests.yml .github\workflows\tests.yml

2. Push this folder to a GitHub repository (e.g. `perfhound-gateway`). The perfhound workflow
   runs on every push and pull request, and by hand: Actions tab -> perfhound -> Run workflow
   -> type good and bad. The tests workflow runs the test suite on every push.
3. Results: the run's summary page shows the table; the JSON file is attached as an artifact.
4. In ANY other repository: copy `examples/perfhound-in-another-repo.yml` to
   `.github/workflows/perfhound.yml` and put your repository name in `uses:`.

## Everything at once (datasets only)

    .venv\Scripts\python -m perfhound.gateway https://github.com/psf/requests

Downloads all commits, PRs and issues into `ingest\psf__requests\`. Big repos take long and
need a token. The folder holds the future of every commit (later PRs, "slow since #123"
issues) - for experiments read it with `perfhound.gateway.ingest.load(folder, until=...)`.

## Inside the full Perfhound project

Same paths: copy `src\perfhound\gateway` and `tests\gateway`. Do not copy this
`pyproject.toml` over the project's own.
