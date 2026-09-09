# Make Idea Agent

`make-idea-agent` runs immediately after Download Paper. It reads only registered paper evidence and
generates a diverse, ranked portfolio of falsifiable innovation candidates. It deliberately does not
wait for a dataset and does not specify code or experiments.

Its typed `IdeaProposal` records the recommended candidate, paper identifiers, novelty category,
baseline gap, differentiators, feasibility, implementation risk, and evidence quality. The next
`idea-design-agent` expands that recommendation into the code/data/experiment blueprint.

The Agent can read bounded paper evidence and write only below
`design/<project_slug>/ideas/`. It cannot search, download data, execute repositories, write
implementation code, or claim measured results. Model schema errors receive one bounded internal
repair attempt before the workflow gate handles failure.
