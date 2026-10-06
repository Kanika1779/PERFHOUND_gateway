"""Gateway: the single entry point to git / GitHub data.

Every other Perfhound module asks the gateway for commit data and never
calls git or GitHub directly (Facade + Adapter patterns).

Repository link -> folder with every commit, PR and issue (JSON):
    python -m perfhound.gateway https://github.com/owner/name
"""

from .api import Gateway
from .cases import BenchmarkSpec, Observation, RegressionCase
from .errors import GatewayError
from .fetcher import FetchError, FetchWarning, RepoFetcher
from .ingest import IngestResult, ingest
from .models import CandidateCommit, FileChange, PRInfo
from .range import CommitRange
from .snapshot import Snapshot, SnapshotError

__all__ = [
    "BenchmarkSpec", "Observation", "RegressionCase", "FetchError", "FetchWarning", "RepoFetcher",
    "Gateway", "GatewayError", "IngestResult", "ingest", "CandidateCommit", "FileChange", "PRInfo", "CommitRange", "Snapshot", "SnapshotError"]
