# Idea Design Agent

`idea-design-agent` is the core engineering research architect. It runs after Make Idea has ranked
paper-grounded novelty candidates and before any dataset is downloaded. The online default is
`deepseek-v4-pro` with high reasoning effort, with `DEEPSEEK_MODEL` retained as an explicit override.

It reads the selected `IdeaProposal` and registered paper evidence, then writes a typed
`ResearchDesign`. The design covers method interfaces, reusable licensed GitHub modules, code layout,
dataset acquisition and distribution requirements, expected-but-unobserved results, fair paper
comparisons, computational experiments, milestones, risks, and fallbacks.

When Experiment Code Run finds a necessary missing computational experiment, this Agent also reads
the typed `ExperimentCoverageAudit`. Every gap must become an exact-ID, exact-type ExperimentPlan
with preserved paper evidence and at least one linked ExpectedResult before the revised design can
pass. The revised workflow then repeats data acquisition, implementation, Code Review, the main
result gate, and supplementary execution.

Before synthesis it invokes exactly five tool-free, write-free auxiliary Agents once each:

- method architecture;
- code reuse and clean architecture;
- data choice and audit gates;
- experiment and statistical decisions;
- critical risk and claim-scope review.

Their outputs must validate as `AdvisorReport`, cite only the supplied paper IDs, and are injected
into the final design as `advisor_consensus`. A malformed, out-of-scope, or missing report fails
closed. The synthesis adds ADR-style decisions, one audit plan per dataset, a pre-specified
statistical analysis, baseline-first stage gates, bounded attempt budgets, stopping rules, unresolved
questions, and claim boundaries. It is not called preregistered without an external time-stamped
registry record. Random seeds are execution replicates rather than inferential units, and coverage
decisions require a finite-sample interval/test over observations or entity clusters. Every supervised
design must audit model-representation or group leakage, repeated inputs with conflicting outcomes,
and measurement/endpoint semantics; domain adapters may add stricter checks such as protein-sequence
and chemical-scaffold audits for DTA. Each advisor phase writes only redacted counts,
durations, and issue codes to the external process log; prompts and report bodies are not logged.

It cannot search the web, download data, execute third-party code, write implementation code, run a
server, or claim measured results. Repository integration is rejected unless the source paper records
a verified open-source license. Optional third-party Skills are registry-only in the public
repository; explicitly synced local inspection copies remain inactive until reviewed.
