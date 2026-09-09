# Code Review Agent

Code Review is an independent, non-editing gate after the user accepts Code Make. It verifies every
generated file and locally vendored GitHub source against SHA-256 manifests, checks the required
model/experiment/training/testing separation, compares component/dataset/experiment/source coverage
with the approved Research Design, parses Python ASTs, detects circular imports and unbounded loops,
then runs only three fixed sanitized commands: the minimal Demo twice and the unittest suite once.

The report records whether loss decreases, parameters change, iterations occur, values stay finite,
and repeated results are deterministic. Failure creates both JSON and Markdown reports under
`design/<project_slug>/reviews/code/`, provides concrete repair instructions, and routes to Code Make.
The state machine permits at most five Code Review failures before requiring human intervention.

