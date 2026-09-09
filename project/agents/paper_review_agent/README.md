# Paper Review Agent

The final reviewer reads the latest project-matched manuscript and a passed paper/code
consistency report. It reviews novelty, methodology, experiments, reproducibility, results,
citations, writing, limitations and submission readiness from the perspective of a senior
reviewer at an authoritative ML conference or bioinformatics journal.

Every round writes immutable JSON and Markdown reports under
`project/reviews/paper/<project_slug>/`. A manuscript passes only when the validated review
contains zero `blocking` or `major` findings. Major findings route to the smallest responsible
upstream Agent and may cycle without a fixed count; every round remains subject to the normal
specialist review and the independent Feedback Agent review.

The reviewer-model verdict is not sufficient by itself. A deterministic preflight blocks missing
structured comparisons against at least two closest prior works, unsupported preregistration,
unsubstantiated superiority claims, claimed reproduction bundles without a content-addressed archive,
and proof/causal language outside the typed claim strengths.
These findings are merged into the final report even when the reviewer model returns `accept`.
