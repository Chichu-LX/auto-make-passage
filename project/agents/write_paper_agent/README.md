# Write Paper Agent

This package converts the approved research evidence graph into a professional Markdown and LaTeX manuscript plus a BibTeX file. It starts only after Code Review, the main result gate, and every supplementary computational experiment have passed.

DeepSeek authors a typed `PaperWritingDraft`, but it cannot provide accepted numeric results. The Runtime re-verifies every local PDF, parses reviewed Python with `ast` without importing it, resolves every formula to exact content-addressed symbols, and recomputes paper-facing experiment rows from persisted server observations. Model-authored result prose containing numeric literals is rejected.

Outputs are isolated under `project/paper/<project_slug>/<task-token>/`. The manifest records every upstream URI, citation PDF digest, formula-to-code binding, result row, manuscript hash, LaTeX hash and bibliography hash. All phases emit redacted records into the shared per-Agent logging tree. Runtime Feedback and the user-evaluation Hook still run after the Agent returns.

Optional GitHub Skills are registry-only in the public repository. An explicit local sync creates
Git-ignored inspection copies; they remain inactive until their licenses and executable contents
are independently reviewed.
