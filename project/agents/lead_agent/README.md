# Lead Agent

Lead is the global semantic controller, not an executor. Its tool list and write scope are both
empty. It makes two typed decisions: which allowlisted Specialist to dispatch, and which
policy-compiled state candidate to adopt after Specialist results, Feedback controls, user
evaluations, and human resume/retry operations.

Normal order:

`download-paper -> make-idea -> idea-design -> download-data -> code-make -> code-review -> server-schedule -> check-result -> experiment-code-run -> write-paper -> check-paper-and-code -> paper-review`

Make Idea ranks paper-grounded novelty candidates; Idea Design converts the selected candidate into
the code/data/experiment contract. Lead chooses state changes and controlled retries; Runtime alone
validates the choice against the LangGraph topology and atomically persists the selected candidate.
Code review may return to Code Make; scheduling may retry or return to Code Make; result checking may
route to Make Idea, Idea Design, Download Data, Code Make, or scheduling. Each gate has a finite retry
budget, after which the workflow enters `NEEDS_HUMAN`.

Lead cannot search, download, inspect files, write code, schedule hardware, execute a tool, or bypass
the runtime state/version check.

After every Specialist result and Feedback audit, Runtime enters `AWAITING_USER_REVIEW`. In that
state Lead receives no allowed target and cannot dispatch. A satisfied user evaluation accepts the
staged transition; a revision evaluation returns to one Runtime-allowlisted repair Agent and carries
the bounded user comments in the next `TaskEnvelope`. Lead observes and routes the explicit verdict,
while Runtime prevents it from changing the user's selected repair target or committing an edge
outside the policy plan.
