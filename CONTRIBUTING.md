# Contributing

Thank you for helping improve Auto Make Passage. Contributions are welcome for Runtime safety,
Agent contracts, research-quality gates, provider adapters, documentation, and tests.

## Before opening an issue

- Search existing issues and documentation.
- Use a public issue for bugs and feature proposals.
- Use the private process in [SECURITY.md](SECURITY.md) for vulnerabilities or credential exposure.
- For scientific-policy changes, describe the evidence boundary being strengthened or relaxed.

## Development setup

```bash
git clone https://github.com/Chichu-LX/auto-make-passage.git
cd auto-make-passage/project
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python scripts/check_open_source_release.py
```

The default test suite is offline and must not require API keys, paid services, a GPU, or an SSH
host.

## Change requirements

1. Add or update tests for behavior changes.
2. Preserve the invariant that Runtime is the only workflow-state writer.
3. Keep model output behind a typed contract and deterministic validation.
4. Do not weaken path, URL, hash, credential, or remote-execution checks for convenience.
5. Update exported JSON Schemas when Pydantic contracts change:

   ```bash
   .venv/bin/python scripts/export_agent_schemas.py
   ```

6. Keep examples bounded to evidence actually present in the repository.
7. Update `CHANGELOG.md` for user-visible behavior.

## Third-party Skills and code

The public repository is registry-only for optional third-party Skills. Do not commit downloaded
Skill directories, copied repository code, papers, datasets, generated environments, or other
material unless redistribution rights have been reviewed and documented.

`scripts/sync_agent_skills.py` may create ignored local inspection copies. Those copies remain
inactive and are not part of a contribution.

## Pull requests

Keep pull requests focused. Explain:

- the failure mode or use case;
- the trust boundary affected;
- the new invariant or behavior;
- how it was verified;
- whether stored schemas, snapshots, or artifacts change.

By contributing, you agree that your contribution is licensed under the repository's MIT License.
