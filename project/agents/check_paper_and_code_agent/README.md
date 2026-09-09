# Check Paper and Code Agent

This read-only Agent is the final truthfulness gate before general paper review. It does not trust the Write Paper manifest: it reopens every declared source artifact, verifies hashes and namespaces, reparses reviewed Python with `ast`, rebuilds main and supplementary result rows from server observations, and rerenders Markdown, LaTeX and BibTeX byte for byte.

The deterministic audit checks manuscript integrity, local PDF citations, required original-paper comparisons, formula-to-code bindings, method-to-code traceability, result-table equality, claim evidence and claim scope. A bounded local-PDF excerpt is supplied to DeepSeek for a typed qualitative advisory on semantics and overclaiming. Blank or unextractable PDFs fall back to manifest abstracts and are recorded as warnings rather than causing an unhandled error.

Any deterministic failure or advisory `error` produces `needs_revision` and normally routes back to Write Paper. Reports are isolated under `project/reviews/consistency/<project_slug>/`, and every phase is logged outside the Agent package for Feedback and user-evaluation monitoring. Optional GitHub Skills are registry-only; explicitly synced local copies remain inactive and are never executed.
