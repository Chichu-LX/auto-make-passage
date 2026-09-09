# Code Make Agent

Code Make runs only after the user accepts Download Data. It requires a project-scoped
`ResearchDesign` and a measured `DatasetManifest`. Runtime first stream-verifies every raw and
materialized data file against its declared project namespace, size, and SHA-256. Licensed
`integrate` GitHub repositories are downloaded as non-executed fixed-commit snapshots below
`code/<project_slug>/third_party/`, with a local per-file SHA-256 `SOURCE_MANIFEST.json`.

Each attempt writes an isolated revision below `code/<project_slug>/implementations/<token>/`.
Models, experiments, training, and testing must occupy separate named folders. After AST safety
preflight, Code Make runs only the fixed dependency-free minimal Demo and unittest suite in a
secret-free environment with 30-second timeouts. The `CodeImplementationManifest` records paths,
hashes, component/dataset/experiment/source traceability, local source provenance, parameter-update
evidence, commands, and known gaps before the result is staged for user review.
