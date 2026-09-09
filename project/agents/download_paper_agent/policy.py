from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from research_agents.downloads import DownloadPolicy


PAPER_POLICY = DownloadPolicy(
    max_bytes=40 * 1024 * 1024,
    allowed_content_types=frozenset({"application/pdf"}),
    max_redirects=3,
    require_pdf_signature=True,
)

RECENT_YEAR_FLOOR = datetime.now(timezone.utc).year - 2
PORTFOLIO_SIZE = 10
AUTHORITATIVE_RATIO = 0.70
RECENT_RATIO = 0.30

# Exact or boundary-aware aliases only. Short names such as ACL must not match
# arbitrary words (for example "molecular"). The list is deliberately explicit
# so venue quality is decided by code rather than by an LLM.
AUTHORITATIVE_VENUES: dict[str, tuple[str, ...]] = {
    "NeurIPS": ("neurips", "nips", "advances in neural information processing systems", "neural information processing systems"),
    "ICML": ("icml", "international conference on machine learning"),
    "ICLR": ("iclr", "international conference on learning representations"),
    "ACL": ("acl", "annual meeting of the association for computational linguistics"),
    "EMNLP": ("emnlp", "empirical methods in natural language processing"),
    "AAAI": ("aaai", "aaai conference on artificial intelligence"),
    "IJCAI": ("ijcai", "international joint conference on artificial intelligence"),
    "KDD": ("kdd", "knowledge discovery and data mining"),
    "Nature Communications": ("nature communications",),
    "Nature Machine Intelligence": ("nature machine intelligence",),
    "Nature Methods": ("nature methods",),
    "Nature Computational Science": ("nature computational science",),
    "Bioinformatics": ("bioinformatics",),
    "Briefings in Bioinformatics": ("briefings in bioinformatics",),
    "BMC Bioinformatics": ("bmc bioinformatics",),
    "Journal of Chemical Information and Modeling": ("journal of chemical information and modeling",),
    "Journal of Cheminformatics": ("journal of cheminformatics",),
    "Nucleic Acids Research": ("nucleic acids research",),
    "PLOS Computational Biology": ("plos computational biology", "plos comput. biol."),
    "PNAS": ("proceedings of the national academy of sciences", "pnas"),
    "Science Advances": ("science advances",),
    "Nature Biotechnology": ("nature biotechnology",),
    "Communications Biology": ("communications biology",),
    "Patterns": ("patterns",),
    "Chemical Science": ("chemical science",),
    "Computational and Structural Biotechnology Journal": ("computational and structural biotechnology journal",),
    "Journal of Computer-Aided Molecular Design": ("journal of computer-aided molecular design",),
    "WIREs Computational Molecular Science": ("wires computational molecular science",),
    "Information Sciences": ("information sciences",),
    "Molecular Informatics": ("molecular informatics",),
    "iScience": ("iscience",),
    "Methods": ("methods",),
    "IEEE/ACM TCBB": ("ieee/acm transactions on computational biology and bioinformatics", "ieee/acm transactions on computational biology & bioinformatics"),
    "Scientific Reports": ("scientific reports",),
    "PLOS ONE": ("plos one",),
    "ACS Omega": ("acs omega",),
    "RSC Advances": ("rsc advances",),
    "Molecules": ("molecules",),
    "International Journal of Molecular Sciences": ("international journal of molecular sciences",),
    "Frontiers in Molecular Biosciences": ("frontiers in molecular biosciences",),
    "Frontiers in Chemistry": ("frontiers in chemistry",),
    "Frontiers in Pharmacology": ("frontiers in pharmacology",),
    "Biomolecules": ("biomolecules",),
    "Biomedicines": ("biomedicines",),
    "PeerJ": ("peerj",),
    "BMC Genomics": ("bmc genomics",),
    "Computational Biology and Chemistry": ("computational biology and chemistry", "comput. biol. chem."),
    "Computers in Biology and Medicine": ("computers in biology and medicine", "comput. biol. medicine"),
    "Proteins": ("proteins: structure, function, and bioinformatics",),
    "Protein Science": ("protein science",),
    "Journal of Physical Chemistry B": ("journal of physical chemistry b",),
    "Physical Chemistry Chemical Physics": ("physical chemistry chemical physics", "physical chemistry, chemical physics - pccp"),
}

CORE_AUTHORITATIVE_VENUE_NAMES = frozenset(
    {
        "NeurIPS", "ICML", "ICLR", "ACL", "EMNLP", "AAAI", "IJCAI", "KDD",
        "Nature Communications", "Nature Machine Intelligence", "Nature Methods",
        "Nature Computational Science", "Bioinformatics", "Briefings in Bioinformatics",
        "Journal of Chemical Information and Modeling", "Journal of Cheminformatics",
        "Nucleic Acids Research", "PLOS Computational Biology", "PNAS", "Science Advances",
        "Nature Biotechnology", "Communications Biology", "Patterns", "IEEE/ACM TCBB",
    }
)


def _normalized_venue(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").lower().replace("–", "-").strip())


def canonical_authoritative_venue(*values: str | None) -> str | None:
    texts = [_normalized_venue(value) for value in values if value]
    for canonical, aliases in AUTHORITATIVE_VENUES.items():
        for alias in aliases:
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            if any(re.search(pattern, text) for text in texts):
                # Avoid classifying BMC/Frontiers/etc. as the exact journal Bioinformatics.
                if canonical == "Bioinformatics" and any(
                    qualifier in text
                    for text in texts
                    for qualifier in ("bmc bioinformatics", "briefings in bioinformatics", "frontiers in bioinformatics")
                ):
                    continue
                return canonical
    return None


def classify_paper_quality(source: str, venue: str | None, year: int | None) -> tuple[str, str | None]:
    canonical = canonical_authoritative_venue(venue, source)
    if canonical:
        return (
            "authoritative" if canonical in CORE_AUTHORITATIVE_VENUE_NAMES else "established_peer_reviewed",
            canonical,
        )
    text = _normalized_venue(f"{venue or ''} {source}")
    is_preprint = any(name in text for name in ("arxiv", "biorxiv", "medrxiv", "chemrxiv"))
    if year is not None and year >= RECENT_YEAR_FLOOR:
        return ("recent_arxiv" if is_preprint else "recent_peer_reviewed"), None
    return ("arxiv_fallback" if is_preprint else "other"), None


def portfolio_targets(total: int) -> tuple[int, int]:
    if total < 1:
        return 0, 0
    authoritative = min(total, math.ceil(total * AUTHORITATIVE_RATIO))
    return authoritative, total - authoritative


def priority_search_queries(objective: str) -> list[tuple[str, str]]:
    authority = (
        'NeurIPS OR ICML OR ICLR OR ACL OR EMNLP OR AAAI OR "Nature Communications" '
        'OR Bioinformatics OR "Briefings in Bioinformatics" OR "Journal of Cheminformatics"'
    )
    return [
        ("authoritative", f'{objective} ({authority}) open access filetype:pdf'),
        ("recent", f'{objective} {RECENT_YEAR_FLOOR}..{datetime.now(timezone.utc).year} open access PDF'),
        ("arxiv_fallback", f'{objective} site:arxiv.org PDF'),
    ]


GITHUB_REPO_PATTERN = re.compile(
    r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.IGNORECASE
)


def normalized_github_repository(value: str) -> str | None:
    match = GITHUB_REPO_PATTERN.search(value)
    if not match:
        return None
    owner, repository = match.groups()
    repository = repository.removesuffix(".git")
    if repository.lower() in {"issues", "pull", "blob", "tree"}:
        return None
    return f"https://github.com/{owner}/{repository}"


def github_repository_from_results(
    title: str, results: list[dict[str, Any]]
) -> tuple[str | None, str | None]:
    normalized_full_title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    title_tokens = {
        token
        for token in re.findall(r"[a-z0-9]{3,}", title.lower())
        if token not in {"with", "from", "using", "based", "deep", "learning", "prediction", "binding", "affinity"}
    }
    method = re.match(r"^([A-Za-z][A-Za-z0-9_-]{2,30})\s*[:：]", title)
    method_token = method.group(1).lower() if method else None
    for item in results:
        evidence_url = str(item.get("url") or "")
        combined = " ".join(
            (str(item.get("title") or ""), evidence_url, str(item.get("content") or ""))
        )
        repository_url = normalized_github_repository(combined)
        if not repository_url:
            continue
        lower = combined.lower()
        overlap = sum(token in lower for token in title_tokens)
        coverage = overlap / max(len(title_tokens), 1)
        normalized_evidence = re.sub(r"[^a-z0-9]+", " ", lower).strip()
        repository_name = repository_url.rstrip("/").rsplit("/", 1)[-1].lower()
        method_exact = bool(
            method_token
            and (
                re.sub(r"[^a-z0-9]", "", method_token)
                == re.sub(r"[^a-z0-9]", "", repository_name)
                or re.search(rf"(?<![a-z0-9]){re.escape(method_token)}(?![a-z0-9])", lower)
            )
        )
        title_exact = len(normalized_full_title) >= 25 and normalized_full_title in normalized_evidence
        strong_coverage = len(title_tokens) >= 5 and coverage >= 0.80
        accepted = method_exact or title_exact if method_token else title_exact or strong_coverage
        if accepted:
            return repository_url, evidence_url or repository_url
    return None, None


def file_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def bounded_search_results(
    results: list[dict[str, Any]], *, phase: str | None = None, limit: int = 10
) -> list[dict[str, Any]]:
    bounded: list[dict[str, Any]] = []
    for item in results[:limit]:
        score = item.get("score")
        bounded.append(
            {
                "title": str(item.get("title", ""))[:500],
                "url": str(item.get("url", ""))[:4096],
                "content": str(item.get("content", ""))[:4000],
                "score": score if isinstance(score, (int, float)) else None,
                "search_phase": phase,
            }
        )
    return bounded
