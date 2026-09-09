# Project showcase

This page is the shortest technical tour for reviewers, interviewers, and contributors.

## One-sentence pitch

Auto Make Passage is a Python multi-agent Runtime that lets models reason about a research workflow
without granting them direct authority over state, files, experiments, or scientific claims.

## Engineering highlights

1. **14-role orchestration** across 12 research stages, one semantic Lead, and one independent
   Feedback coordinator.
2. **18-state executable topology** with a LangGraph conditional guard and version-bound commits.
3. **Typed control plane** using Pydantic and exported JSON Schemas for dispatch, tasks, results,
   feedback, user evaluation, and snapshots.
4. **Dual human gates**: Specialist output and Feedback output are reviewed independently.
5. **Content-addressed evidence**: artifacts carry hashes and remain inside Agent-specific scopes.
6. **Claim-evidence consistency checks**: paper citations, formulas, code symbols, result rows, and
   provenance links are independently reconstructed before final review.
7. **Failure recovery**: durable checkpoints resume persisted Feedback/results without duplicate
   model calls or experiment launches.
8. **Security-first ingestion**: SSRF, private redirects, MIME/signature mismatch, archive traversal,
   symlink escape, and oversized payloads are rejected.

## Demonstrated output

The bundled StrictColdCal study is a six-page IEEE-style artifact generated from persisted local
CPU observations:

- [manuscript.pdf](../project/paper/dta-live-pipeline-check-v2/strictcoldcal_ieee/manuscript.pdf)
- [manuscript.tex](../project/paper/dta-live-pipeline-check-v2/strictcoldcal_ieee/manuscript.tex)
- [artifact_manifest.json](../project/paper/dta-live-pipeline-check-v2/strictcoldcal_ieee/artifact_manifest.json)
- [evidence_summary.json](../project/paper/dta-live-pipeline-check-v2/strictcoldcal_ieee/evidence_summary.json)

The example intentionally limits itself to the observed Davis pilot. KIBA and Metz are recorded as
materialized datasets, not executed results. That boundary is the point of the demonstration: the
paper layer cannot silently promote an available dataset into a completed experiment.

## Suggested code-reading path

| Question | Start here |
|---|---|
| Who is allowed to change state? | `project/research_agents/runtime.py` |
| How are safe transitions compiled? | `project/research_agents/state_machine.py` |
| Is the graph actually executed? | `project/research_agents/workflow_graph.py` |
| What contracts constrain model output? | `project/research_agents/schemas.py` |
| How are writes and hashes enforced? | `project/research_agents/storage.py` |
| How does Feedback remain advisory? | `project/agents/feedback_agent/agent.py` |
| How are paper claims checked? | `project/agents/check_paper_and_code_agent/agent.py` |
| What failure modes are tested? | `project/tests/` |

## Reproduce the control flow without paid services

```bash
cd project
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m research_agents demo \
  --topic "Evidence-governed research agents"
.venv/bin/python -m unittest discover -s tests -q
```

After Python dependencies are installed, the demo makes no model or search API calls. It uses
deterministic mock Agents but the same Runtime, state machine, persistence, Feedback checkpoint,
and human-review contracts as live mode.
