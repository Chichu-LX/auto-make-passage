# Security policy

## Supported versions

Security fixes are applied to the latest revision of the `main` branch. This project is currently
an alpha research prototype and does not maintain older release branches.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability that may expose credentials, private data,
remote execution access, path traversal, SSRF, prompt/log contents, or a reliable security bypass.

Use GitHub's private vulnerability reporting flow:

<https://github.com/Chichu-LX/auto-make-passage/security/advisories/new>

Include the affected commit, a minimal reproduction, expected impact, and any proposed mitigation.
Remove real credentials, private host details, copyrighted datasets, and sensitive research data
from the report.

You should receive an acknowledgement within seven days. Disclosure timing will be coordinated
after the issue is reproduced and a mitigation is available.

## Operational warning

Generated code and third-party content are untrusted. Use a dedicated container or machine,
least-privilege credentials, and an isolated compute directory for live runs. Remote execution is
disabled by default and must remain an explicit operator choice.
