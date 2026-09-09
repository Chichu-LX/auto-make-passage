# Security model

Auto Make Passage treats model output, downloaded content, third-party repositories, archives,
remote hosts, and persisted artifacts as untrusted inputs.

## Trust boundaries

| Component | Trust level | Enforcement |
|---|---|---|
| Lead/Specialist/Feedback model output | Untrusted proposal | Pydantic contracts, allowlists, deterministic validation |
| Workflow snapshot | Versioned control record | Run identity, observed version, atomic persistence |
| Artifact reference | Untrusted until verified | Namespace, path scope, symlink checks, SHA-256 |
| Public URL and redirects | Untrusted network input | HTTP(S) only, DNS/IP checks, redirect revalidation, size/MIME/signature policy |
| Dataset archive | Untrusted binary | Extension/content checks and archive traversal rejection |
| Generated code | Untrusted implementation | Static checks, bounded validation, foreground demo before background jobs |
| Remote execution target | Disabled by default | Explicit backend/config, dedicated credentials, bounded paths and commands |
| Third-party Skill | Registry metadata only | Not redistributed, downloaded, activated, or executed by default |

## Credentials

Real credentials belong in ignored local files or the process environment. The repository ships
only `.env.example` files. Server credentials are isolated to the Server Schedule Agent and cannot
be embedded in the public server JSON configuration. Passwords are passed to child processes via a
bounded environment mechanism rather than command-line arguments.

Never commit:

- `.env` or `servers.local.json`;
- API keys, access tokens, SSH private keys, or passphrases;
- local run logs, snapshots, datasets, downloaded papers, or generated code;
- remote hostnames or usernames that should remain private.

## Logging and privacy

Process logs retain control metadata needed for diagnosis: timestamp, Agent, event, workflow state,
state version, attempt, duration, status, and bounded validation details. They do not retain full
prompts, human comments, PDF bodies, dataset rows, result payloads, authorization headers, or keys.
Credential-like values are redacted before persistence.

Redaction reduces accidental exposure; it is not a substitute for reviewing logs before sharing
them.

## Fail-closed behavior

The system moves to `NEEDS_HUMAN` or refuses a commit when it encounters an invalid transition,
wrong Agent identity, stale version, out-of-scope path, malformed Feedback control, exhausted
bounded retry budget, missing required evidence, or unverified remote/download input.

## Non-goals

The project does not claim to provide:

- a hardened sandbox for arbitrary generated or third-party code;
- proof that model-generated research is correct;
- automatic ethics, privacy, copyright, or dataset-license approval;
- safe autonomous publication without human scientific review;
- multi-tenant isolation suitable for a hosted service.

Use a separate OS/container sandbox and least-privilege credentials for any live execution.

## Reporting vulnerabilities

Do not open a public issue for a vulnerability that may expose secrets or enable exploitation.
Follow [SECURITY.md](../SECURITY.md) instead.
