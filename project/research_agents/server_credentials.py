from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import re
import stat


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_CREDENTIAL_ENV = (
    WORKSPACE_ROOT / "project" / "agents" / "server_schedule_agent" / ".env"
)
MAX_SERVER_ENV_BYTES = 64 * 1024
MAX_PRIVATE_KEY_BYTES = 1024 * 1024
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,127}$")
_PREFIX = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_USER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_DNS_HOST = re.compile(
    r"^(?=.{1,253}\Z)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
)
_FIELDS = (
    "PRIVATE_KEY_PASSPHRASE",
    "PRIVATE_KEY_PATH",
    "AUTH_MODE",
    "PASSWORD",
    "HOST",
    "PORT",
    "USER",
)


class ServerCredentialError(RuntimeError):
    """Raised without embedding any credential value in the message."""


@dataclass(frozen=True, repr=False)
class SSHCredentials:
    credential_prefix: str
    host: str
    port: int
    user: str
    auth_mode: str
    password: str | None = None
    private_key_path: Path | None = None
    private_key_passphrase: str | None = None

    def __repr__(self) -> str:
        return (
            "SSHCredentials(credential_prefix="
            f"{self.credential_prefix!r}, host=<redacted>, port={self.port}, "
            f"user=<redacted>, auth_mode={self.auth_mode!r})"
        )

    @property
    def askpass_secret(self) -> str | None:
        if self.auth_mode == "password":
            return self.password
        return self.private_key_passphrase


def _decode_value(raw: str, *, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            raise ServerCredentialError(
                f"unterminated server credential value on line {line_number}"
            )
        value = value[1:-1]
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise ServerCredentialError(
            f"invalid control character in server credential file on line {line_number}"
        )
    return value


def _split_server_name(name: str, *, line_number: int) -> tuple[str, str]:
    if not name.startswith("SERVER_"):
        raise ServerCredentialError(
            f"server Agent .env contains a non-server variable on line {line_number}"
        )
    for field in _FIELDS:
        suffix = f"_{field}"
        if name.endswith(suffix):
            prefix = name[len("SERVER_") : -len(suffix)]
            if not _PREFIX.fullmatch(prefix):
                break
            return prefix, field
    raise ServerCredentialError(
        f"unknown server credential variable on line {line_number}"
    )


def _read_assignments(path: Path) -> dict[str, dict[str, str]]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ServerCredentialError("server Agent credential .env is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ServerCredentialError(
            "server Agent credential .env must be a regular non-symbolic file"
        )
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ServerCredentialError(
            "server Agent credential .env permissions must be exactly 600"
        )
    if metadata.st_size > MAX_SERVER_ENV_BYTES:
        raise ServerCredentialError("server Agent credential .env exceeds its size limit")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ServerCredentialError(
            "server Agent credential .env is not readable UTF-8"
        ) from exc

    groups: dict[str, dict[str, str]] = {}
    seen_names: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ServerCredentialError(
                f"invalid server credential assignment on line {line_number}"
            )
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME.fullmatch(name):
            raise ServerCredentialError(
                f"invalid server credential variable name on line {line_number}"
            )
        if name in seen_names:
            raise ServerCredentialError(
                f"duplicate server credential variable on line {line_number}"
            )
        seen_names.add(name)
        prefix, field = _split_server_name(name, line_number=line_number)
        groups.setdefault(prefix, {})[field] = _decode_value(
            raw_value, line_number=line_number
        )
    return groups


def _validate_host(value: str) -> str:
    host = value.strip()
    if not host or any(character.isspace() for character in host) or host.startswith("-"):
        raise ServerCredentialError("server SSH host is invalid")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if not _DNS_HOST.fullmatch(host):
            raise ServerCredentialError("server SSH host is invalid")
    return host


def _validate_private_key(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ServerCredentialError("server SSH private-key path must be absolute")
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise ServerCredentialError("server SSH private-key file is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ServerCredentialError(
            "server SSH private key must be a regular non-symbolic file"
        )
    mode = stat.S_IMODE(metadata.st_mode)
    if not mode & stat.S_IRUSR or mode & 0o077:
        raise ServerCredentialError(
            "server SSH private-key permissions must deny group/world access"
        )
    if metadata.st_size <= 0 or metadata.st_size > MAX_PRIVATE_KEY_BYTES:
        raise ServerCredentialError("server SSH private-key size is invalid")
    return candidate.resolve()


class ServerCredentialStore:
    """Strict reader for credentials owned only by Server Schedule Agent."""

    def __init__(self, path: str | Path | None = None) -> None:
        candidate = Path(path) if path is not None else DEFAULT_SERVER_CREDENTIAL_ENV
        self.path = candidate if candidate.is_absolute() else candidate.absolute()

    def resolve(self, credential_prefix: str) -> SSHCredentials:
        if not _PREFIX.fullmatch(credential_prefix):
            raise ServerCredentialError("server credential prefix is invalid")
        values = _read_assignments(self.path).get(credential_prefix)
        if values is None:
            raise ServerCredentialError(
                "server Agent credential .env has no matching credential prefix"
            )
        try:
            host = _validate_host(values["HOST"])
            user = values["USER"].strip()
            port = int(values.get("PORT", "22"))
            auth_mode = values["AUTH_MODE"].strip().lower()
        except (KeyError, ValueError) as exc:
            raise ServerCredentialError(
                "server SSH host, user, port, or authentication mode is incomplete"
            ) from exc
        if not _USER.fullmatch(user):
            raise ServerCredentialError("server SSH user is invalid")
        if not 1 <= port <= 65535:
            raise ServerCredentialError("server SSH port is outside 1..65535")
        if auth_mode == "password":
            password = values.get("PASSWORD", "")
            if not password:
                raise ServerCredentialError("password SSH authentication requires a password")
            if len(password) > 4096:
                raise ServerCredentialError("server SSH password exceeds its size limit")
            if values.get("PRIVATE_KEY_PATH") or values.get("PRIVATE_KEY_PASSPHRASE"):
                raise ServerCredentialError(
                    "password SSH authentication cannot also declare private-key values"
                )
            return SSHCredentials(
                credential_prefix=credential_prefix,
                host=host,
                port=port,
                user=user,
                auth_mode=auth_mode,
                password=password,
            )
        if auth_mode == "key":
            raw_key_path = values.get("PRIVATE_KEY_PATH", "")
            if not raw_key_path:
                raise ServerCredentialError(
                    "key SSH authentication requires a private-key path"
                )
            if values.get("PASSWORD"):
                raise ServerCredentialError(
                    "key SSH authentication cannot also declare a password"
                )
            passphrase = values.get("PRIVATE_KEY_PASSPHRASE") or None
            if passphrase is not None and len(passphrase) > 4096:
                raise ServerCredentialError(
                    "server SSH private-key passphrase exceeds its size limit"
                )
            return SSHCredentials(
                credential_prefix=credential_prefix,
                host=host,
                port=port,
                user=user,
                auth_mode=auth_mode,
                private_key_path=_validate_private_key(raw_key_path),
                private_key_passphrase=passphrase,
            )
        raise ServerCredentialError("server SSH auth mode must be password or key")
