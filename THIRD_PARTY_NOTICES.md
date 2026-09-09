# Third-party notices

The repository's MIT License applies to original Auto Make Passage code and documentation. It does
not replace the terms of external packages, templates, services, research sources, or optional
Skills.

## Python dependencies

Runtime dependencies are installed from their publishers through Python package indexes and are
not vendored in this repository. Each dependency remains subject to its own license. The complete,
version-bounded dependency list is in `project/pyproject.toml`.

## Optional Agent Skills

Each `project/agents/<agent>/skills/registry.json` file records optional external Skill sources,
their purpose, repository URL, and the commit observed during curation. The public repository does
not include or redistribute those Skill bodies or scripts.

Running `project/scripts/sync_agent_skills.py` asks an external package tool to create local,
Git-ignored inspection copies. Downloading a Skill does not grant permission to use or redistribute
it. Review the upstream license and executable content before any activation. Auto Make Passage
does not activate or execute synced Skills by default.

## IEEEtran example source

`project/New_IEEEtran_how-to.tex` is an IEEE Publication Technology Department template example
distributed under the LaTeX Project Public License (LPPL), version 1.3. Its copyright, attribution,
license statement, and warranty disclaimer remain in the source file. It is not covered by this
repository's MIT License.

## Research citations

Bibliographic entries and references in the bundled demonstration manuscript identify their
original scholarly sources. Inclusion of a citation does not relicense the cited work. Downloaded
papers and datasets used by local runs are excluded from version control.
