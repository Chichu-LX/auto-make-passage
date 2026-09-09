from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import re
import socket
import stat
import zipfile
from html.parser import HTMLParser
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Iterable
from urllib.parse import urljoin, urlsplit

import httpx
import py7zr


class DownloadPolicyError(RuntimeError):
    """The remote resource violated an explicit download safety policy."""


@dataclass(frozen=True)
class DownloadPolicy:
    max_bytes: int
    allowed_content_types: frozenset[str]
    max_redirects: int = 3
    require_pdf_signature: bool = False
    max_archive_members: int = 1000
    max_archive_uncompressed_bytes: int = 512 * 1024 * 1024
    max_archive_ratio: float = 100.0


@dataclass(frozen=True)
class DownloadPayload:
    content: bytes
    final_url: str
    content_type: str
    sha256: str
    redirect_count: int

    @property
    def size_bytes(self) -> int:
        return len(self.content)


Resolver = Callable[[str], Iterable[str]]


def _system_resolver(hostname: str) -> Iterable[str]:
    return {
        item[4][0]
        for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    }


def _is_forbidden_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return not address.is_global


def validate_public_http_url(url: str, resolver: Resolver = _system_resolver) -> None:
    if len(url) > 4096:
        raise DownloadPolicyError("URL exceeds the 4096 character limit")
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise DownloadPolicyError("only HTTP(S) downloads are allowed")
    if not parsed.hostname:
        raise DownloadPolicyError("download URL must include a hostname")
    if parsed.username or parsed.password:
        raise DownloadPolicyError("credentials in download URLs are forbidden")
    try:
        port = parsed.port
    except ValueError as exc:
        raise DownloadPolicyError("download URL contains an invalid port") from exc
    if port is not None and port not in {80, 443}:
        raise DownloadPolicyError("only ports 80 and 443 are allowed")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise DownloadPolicyError("localhost downloads are forbidden")
    try:
        addresses = [str(ipaddress.ip_address(hostname))]
    except ValueError:
        try:
            addresses = list(resolver(hostname))
        except OSError as exc:
            raise DownloadPolicyError("download hostname could not be resolved") from exc
    if not addresses:
        raise DownloadPolicyError("download hostname resolved to no addresses")
    try:
        if any(_is_forbidden_address(address) for address in addresses):
            raise DownloadPolicyError("private or non-global network targets are forbidden")
    except ValueError as exc:
        raise DownloadPolicyError("resolver returned an invalid IP address") from exc


class SafeHttpDownloader:
    """Small, policy-enforcing HTTP client used by both download agents.

    Redirects are handled manually so every hop receives the same SSRF check. The
    decompressed byte stream is counted, which also bounds servers that ignore the
    requested identity encoding.
    """

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        resolver: Resolver = _system_resolver,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = client
        self._resolver = resolver
        self._timeout = httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 10.0))

    def fetch(self, url: str, policy: DownloadPolicy) -> DownloadPayload:
        if self._client is not None:
            return self._fetch_with_client(self._client, url, policy)
        with httpx.Client(
            follow_redirects=False,
            timeout=self._timeout,
            trust_env=False,
        ) as client:
            return self._fetch_with_client(client, url, policy)

    def resolve_citation_pdf_url(
        self,
        landing_url: str,
        *,
        max_bytes: int = 2 * 1024 * 1024,
    ) -> str | None:
        """Resolve only a standard citation_pdf_url from one safe public HTML page.

        The returned URL is not trusted as a PDF until ``fetch`` independently applies
        all redirect, MIME, size, and signature gates.  Europe PMC article pages receive
        a deterministic render URL because some PMC landing pages use a browser proof-of-
        work interstitial instead of exposing their ordinary citation meta tags.
        """

        validate_public_http_url(landing_url, self._resolver)
        parsed = urlsplit(landing_url)
        host = (parsed.hostname or "").rstrip(".").lower()
        match = re.fullmatch(r"/articles/(PMC\d+)/?", parsed.path, re.IGNORECASE)
        if host == "pmc.ncbi.nlm.nih.gov" and match:
            return f"https://europepmc.org/articles/{match.group(1).upper()}?pdf=render"
        if self._client is not None:
            return self._resolve_meta_with_client(
                self._client, landing_url, max_bytes=max_bytes
            )
        with httpx.Client(
            follow_redirects=False,
            timeout=self._timeout,
            trust_env=False,
        ) as client:
            return self._resolve_meta_with_client(
                client, landing_url, max_bytes=max_bytes
            )

    def _resolve_meta_with_client(
        self,
        client: httpx.Client,
        landing_url: str,
        *,
        max_bytes: int,
    ) -> str | None:
        current_url = landing_url
        redirects = 0
        while True:
            validate_public_http_url(current_url, self._resolver)
            try:
                with client.stream(
                    "GET",
                    current_url,
                    headers={
                        "Accept": "text/html,application/xhtml+xml",
                        "Accept-Encoding": "identity",
                        "User-Agent": "auto-make-passage/0.1 research downloader",
                    },
                    follow_redirects=False,
                ) as response:
                    if response.is_redirect:
                        if redirects >= 3:
                            raise DownloadPolicyError(
                                "landing-page redirect limit exceeded"
                            )
                        location = response.headers.get("location")
                        if not location:
                            raise DownloadPolicyError(
                                "landing-page redirect has no Location header"
                            )
                        current_url = urljoin(str(response.url), location)
                        redirects += 1
                        continue
                    if response.status_code != 200:
                        raise DownloadPolicyError(
                            f"landing page returned HTTP {response.status_code}"
                        )
                    content_type = (
                        response.headers.get("content-type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if content_type not in {"text/html", "application/xhtml+xml"}:
                        raise DownloadPolicyError(
                            "landing page did not return allowlisted HTML"
                        )
                    chunks: list[bytes] = []
                    received = 0
                    for chunk in response.iter_bytes():
                        received += len(chunk)
                        if received > max_bytes:
                            raise DownloadPolicyError(
                                "landing page exceeds metadata size policy"
                            )
                        chunks.append(chunk)
                    try:
                        text = b"".join(chunks).decode("utf-8-sig")
                    except UnicodeDecodeError as exc:
                        raise DownloadPolicyError(
                            "landing-page metadata must be UTF-8"
                        ) from exc
                    parser = _CitationPdfMetaParser()
                    parser.feed(text)
                    if parser.pdf_url is None:
                        return None
                    resolved = urljoin(str(response.url), parser.pdf_url)
                    validate_public_http_url(resolved, self._resolver)
                    return resolved
            except httpx.HTTPError as exc:
                raise DownloadPolicyError(
                    f"landing-page transport failed: {type(exc).__name__}"
                ) from exc

    def _fetch_with_client(
        self, client: httpx.Client, url: str, policy: DownloadPolicy
    ) -> DownloadPayload:
        current_url = url
        redirects = 0
        while True:
            validate_public_http_url(current_url, self._resolver)
            try:
                with client.stream(
                    "GET",
                    current_url,
                    headers={
                        # Some repository content endpoints (notably Zenodo) return
                        # HTTP 406 for a narrow Accept header even when their response
                        # MIME is the requested type.  Content negotiation is therefore
                        # broad, while the response below is still rejected unless its
                        # actual Content-Type is in the explicit policy allowlist.
                        "Accept": "*/*",
                        "Accept-Encoding": "identity",
                        "User-Agent": "auto-make-passage/0.1 research downloader",
                    },
                    follow_redirects=False,
                ) as response:
                    if response.is_redirect:
                        if redirects >= policy.max_redirects:
                            raise DownloadPolicyError("redirect limit exceeded")
                        location = response.headers.get("location")
                        if not location:
                            raise DownloadPolicyError("redirect response has no Location header")
                        current_url = urljoin(str(response.url), location)
                        redirects += 1
                        continue

                    if response.status_code != 200:
                        raise DownloadPolicyError(
                            f"download returned HTTP {response.status_code}"
                        )
                    content_type = (
                        response.headers.get("content-type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if content_type not in policy.allowed_content_types:
                        raise DownloadPolicyError(
                            f"content type {content_type or '<missing>'} is not allowed"
                        )
                    length = response.headers.get("content-length")
                    if length:
                        try:
                            declared_size = int(length)
                        except ValueError as exc:
                            raise DownloadPolicyError("invalid Content-Length header") from exc
                        if declared_size < 0 or declared_size > policy.max_bytes:
                            raise DownloadPolicyError("declared file size exceeds policy")

                    chunks: list[bytes] = []
                    received = 0
                    for chunk in response.iter_bytes():
                        received += len(chunk)
                        if received > policy.max_bytes:
                            raise DownloadPolicyError("streamed file size exceeds policy")
                        chunks.append(chunk)
                    payload = b"".join(chunks)
                    if not payload:
                        raise DownloadPolicyError("downloaded file is empty")
                    if policy.require_pdf_signature and not payload.startswith(b"%PDF-"):
                        raise DownloadPolicyError("download is not a PDF according to its signature")
                    return DownloadPayload(
                        content=payload,
                        final_url=str(response.url),
                        content_type=content_type,
                        sha256=hashlib.sha256(payload).hexdigest(),
                        redirect_count=redirects,
                    )
            except httpx.HTTPError as exc:
                raise DownloadPolicyError(f"HTTP transport failed: {type(exc).__name__}") from exc


class _CitationPdfMetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pdf_url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.pdf_url is not None or tag.lower() != "meta":
            return
        values = {key.lower(): value for key, value in attrs if value is not None}
        name = (values.get("name") or values.get("property") or "").lower()
        content = values.get("content")
        if name == "citation_pdf_url" and content:
            self.pdf_url = content.strip()


SAFE_ARCHIVE_MEMBER_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".parquet",
    ".txt",
}


def validate_zip_archive(payload: bytes, policy: DownloadPolicy) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (zipfile.BadZipFile, OSError) as exc:
        raise DownloadPolicyError("dataset is not a valid ZIP archive") from exc
    with archive:
        members = archive.infolist()
        if not members or len(members) > policy.max_archive_members:
            raise DownloadPolicyError("ZIP member count violates policy")
        total_uncompressed = 0
        data_members = 0
        for member in members:
            normalized = PurePosixPath(member.filename.replace("\\", "/"))
            if normalized.is_absolute() or ".." in normalized.parts:
                raise DownloadPolicyError("ZIP contains a path traversal entry")
            if member.flag_bits & 0x1:
                raise DownloadPolicyError("encrypted ZIP entries require manual review")
            unix_mode = member.external_attr >> 16
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise DownloadPolicyError("ZIP symbolic links are forbidden")
            if member.is_dir():
                continue
            total_uncompressed += member.file_size
            if total_uncompressed > policy.max_archive_uncompressed_bytes:
                raise DownloadPolicyError("ZIP uncompressed size exceeds policy")
            ratio = member.file_size / max(member.compress_size, 1)
            if ratio > policy.max_archive_ratio:
                raise DownloadPolicyError("ZIP compression ratio exceeds policy")
            if normalized.suffix.lower() not in SAFE_ARCHIVE_MEMBER_EXTENSIONS:
                raise DownloadPolicyError("ZIP contains a file type that is not data-only")
            data_members += 1
        if data_members == 0:
            raise DownloadPolicyError("ZIP contains no supported data files")


def validate_7z_archive(payload: bytes, policy: DownloadPolicy) -> None:
    if not payload.startswith(b"7z\xbc\xaf\x27\x1c"):
        raise DownloadPolicyError("dataset is not a 7z archive according to its signature")
    try:
        archive = py7zr.SevenZipFile(io.BytesIO(payload), mode="r")
    except (py7zr.Bad7zFile, OSError) as exc:
        raise DownloadPolicyError("dataset is not a valid 7z archive") from exc
    with archive:
        if archive.password_protected:
            raise DownloadPolicyError("password-protected 7z archives require manual review")
        members = archive.list()
        if not members or len(members) > policy.max_archive_members:
            raise DownloadPolicyError("7z member count violates policy")
        total_uncompressed = 0
        data_members = 0
        for member in members:
            normalized = PurePosixPath(member.filename.replace("\\", "/"))
            if normalized.is_absolute() or ".." in normalized.parts:
                raise DownloadPolicyError("7z contains a path traversal entry")
            if member.is_symlink:
                raise DownloadPolicyError("7z symbolic links are forbidden")
            if member.is_directory:
                continue
            total_uncompressed += int(member.uncompressed or 0)
            if total_uncompressed > policy.max_archive_uncompressed_bytes:
                raise DownloadPolicyError("7z uncompressed size exceeds policy")
            ratio = int(member.uncompressed or 0) / max(int(member.compressed or 0), 1)
            if ratio > policy.max_archive_ratio:
                raise DownloadPolicyError("7z compression ratio exceeds policy")
            if normalized.suffix.lower() not in SAFE_ARCHIVE_MEMBER_EXTENSIONS:
                raise DownloadPolicyError("7z contains a file type that is not data-only")
            data_members += 1
        if data_members == 0:
            raise DownloadPolicyError("7z contains no supported data files")


def validate_dataset_payload(payload: bytes, dataset_format: str, policy: DownloadPolicy) -> None:
    if dataset_format == "zip":
        validate_zip_archive(payload, policy)
        return
    if dataset_format == "7z":
        validate_7z_archive(payload, policy)
        return
    if dataset_format == "parquet":
        if len(payload) < 8 or not (payload.startswith(b"PAR1") and payload.endswith(b"PAR1")):
            raise DownloadPolicyError("Parquet signature validation failed")
        return
    if b"\x00" in payload and dataset_format in {"csv", "tsv", "json", "jsonl", "txt"}:
        raise DownloadPolicyError("text dataset contains NUL bytes")
    if dataset_format in {"json", "jsonl"}:
        try:
            text = payload.decode("utf-8-sig")
            if dataset_format == "json":
                json.loads(text)
            else:
                rows = [line for line in text.splitlines() if line.strip()]
                if not rows:
                    raise ValueError("JSONL contains no records")
                for line in rows:
                    json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DownloadPolicyError(f"{dataset_format.upper()} validation failed") from exc
    elif dataset_format in {"csv", "tsv", "txt"}:
        try:
            payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DownloadPolicyError("text dataset must be UTF-8") from exc
