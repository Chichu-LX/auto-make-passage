from __future__ import annotations

from collections.abc import Iterable

from research_agents.schemas import (
    ObservedExperimentResult,
    PaperCitationEvidence,
    PaperClaimDraft,
    PaperFormulaEvidence,
    PaperSectionDraft,
)


SECTION_ORDER = (
    "introduction",
    "related_work",
    "method",
    "experimental_setup",
    "results",
    "discussion",
    "limitations",
    "conclusion",
)


def _number(value: float | None) -> str:
    return "not-applicable" if value is None else format(value, ".17g")


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _citation_command(ids: Iterable[str]) -> str:
    values = list(dict.fromkeys(ids))
    return f"\\cite{{{','.join(values)}}}" if values else ""


def _result_markdown(result: ObservedExperimentResult) -> list[str]:
    seeds = ", ".join(
        f"{item.seed}:{_number(item.value)}" for item in result.seed_results
    )
    return [
        f"### Observed result `{result.result_key}`",
        "",
        "| Scope | Experiment | Metric | Direction | Per-seed values | Mean | Sample SD | Min | Max | Target |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            _markdown_cell(value)
            for value in (
                result.scope,
                result.experiment_id,
                result.metric_name,
                result.direction,
                seeds,
                _number(result.mean),
                _number(result.sample_standard_deviation),
                _number(result.minimum),
                _number(result.maximum),
                _number(result.target_value),
            )
        )
        + " |",
        "",
        f"Evidence source: `{result.source_report_uri}`. Values are rendered from persisted server observations.",
        "",
    ]


def render_markdown(
    *,
    title: str,
    abstract: str,
    keywords: list[str],
    sections: list[PaperSectionDraft],
    formulas: list[PaperFormulaEvidence],
    claims: list[PaperClaimDraft],
    citations: list[PaperCitationEvidence],
    results: list[ObservedExperimentResult],
    acknowledgements: str | None,
) -> str:
    section_map = {item.section_id: item for item in sections}
    formula_map = {item.formula_id: item for item in formulas}
    result_map = {item.result_key: item for item in results}
    lines = [
        f"# {title}",
        "",
        "## Abstract",
        "",
        abstract,
        "",
        f"**Keywords:** {', '.join(keywords)}",
        "",
    ]
    for section_id in SECTION_ORDER:
        section = section_map[section_id]
        lines.extend([f"## {section.title}", ""])
        for paragraph in section.paragraphs:
            lines.extend([paragraph, ""])
        if section.citation_ids:
            lines.extend(
                [
                    "Section evidence: "
                    + ", ".join(f"[{paper_id}]" for paper_id in section.citation_ids),
                    "",
                ]
            )
        for formula_id in section.formula_ids:
            formula = formula_map[formula_id]
            lines.extend(
                [
                    f"### {formula.label} (`{formula.formula_id}`)",
                    "",
                    f"$$\n{formula.latex}\n$$",
                    "",
                    formula.explanation,
                    "",
                    "Code grounding: "
                    + ", ".join(
                        f"`{item.file_path}:{item.line_start}` (`{item.symbol}`, `{item.file_sha256}`)"
                        for item in formula.code_bindings
                    ),
                    "",
                ]
            )
        for result_key in section.result_keys:
            lines.extend(_result_markdown(result_map[result_key]))

    if acknowledgements:
        lines.extend(["## Acknowledgements", "", acknowledgements, ""])

    lines.extend(
        [
            "## Claim evidence register",
            "",
            "| Claim | Type | Strength | Citations | Code | Results | Statement |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for claim in claims:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    claim.claim_id,
                    claim.claim_type,
                    claim.strength,
                    ", ".join(claim.citation_ids) or "not-applicable",
                    ", ".join(claim.code_file_paths) or "not-applicable",
                    ", ".join(claim.result_keys) or "not-applicable",
                    claim.text,
                )
            )
            + " |"
        )

    lines.extend(["", "## References", ""])
    for citation in citations:
        authors = ", ".join(citation.authors) or "Authors unavailable"
        year = str(citation.year) if citation.year is not None else "year unavailable"
        doi = f" DOI: {citation.doi}." if citation.doi else ""
        lines.append(
            f"- [{citation.paper_id}] {authors}. *{citation.title}*. "
            f"{citation.venue}, {year}.{doi} Local PDF: `{citation.local_pdf.uri}`"
        )
    lines.extend(
        [
            "",
            "> Integrity note: experimental values in this manuscript are deterministic renderings of passed main and supplementary server reports; model-authored prose cannot introduce numeric result literals.",
            "",
        ]
    )
    return "\n".join(lines)


def _latex_result_table(result: ObservedExperimentResult) -> list[str]:
    seeds = "; ".join(
        f"{item.seed}:{_number(item.value)}" for item in result.seed_results
    )
    values = (
        result.scope,
        result.experiment_id,
        result.metric_name,
        result.direction,
        seeds,
        _number(result.mean),
        _number(result.sample_standard_deviation),
        _number(result.minimum),
        _number(result.maximum),
        _number(result.target_value),
    )
    return [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllllrrrrr}",
        r"\hline",
        r"Scope & Experiment & Metric & Direction & Seeds & Mean & SD & Min & Max & Target \\",
        r"\hline",
        " & ".join(_latex_escape(value) for value in values) + r" \\",
        r"\hline",
        r"\end{tabular}",
        "\\caption{" + _latex_escape(f"Observed result {result.result_key} from {result.source_report_uri}.") + "}",
        r"\end{table*}",
    ]


def render_latex(
    *,
    title: str,
    abstract: str,
    keywords: list[str],
    sections: list[PaperSectionDraft],
    formulas: list[PaperFormulaEvidence],
    claims: list[PaperClaimDraft],
    citations: list[PaperCitationEvidence],
    results: list[ObservedExperimentResult],
    acknowledgements: str | None,
) -> str:
    del claims, citations  # Their grounded content is rendered through sections and BibTeX.
    section_map = {item.section_id: item for item in sections}
    formula_map = {item.formula_id: item for item in formulas}
    result_map = {item.result_key: item for item in results}
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs}",
        r"\usepackage[T1]{fontenc}",
        "\\title{" + _latex_escape(title) + "}",
        r"\author{Anonymous Authors}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        _latex_escape(abstract),
        r"\end{abstract}",
        r"\noindent\textbf{Keywords:} " + _latex_escape(", ".join(keywords)),
    ]
    for section_id in SECTION_ORDER:
        section = section_map[section_id]
        lines.append("\\section{" + _latex_escape(section.title) + "}")
        for paragraph in section.paragraphs:
            lines.extend([_latex_escape(paragraph), ""])
        citation_command = _citation_command(section.citation_ids)
        if citation_command:
            lines.extend([r"\noindent\textit{Section evidence:} " + citation_command, ""])
        for formula_id in section.formula_ids:
            formula = formula_map[formula_id]
            lines.extend(
                [
                    "\\subsection{" + _latex_escape(formula.label) + "}",
                    r"\begin{equation}",
                    formula.latex,
                    r"\end{equation}",
                    _latex_escape(formula.explanation),
                    "",
                    r"\noindent\textit{Code grounding:} "
                    + _latex_escape(
                        "; ".join(
                            f"{item.file_path}:{item.line_start} {item.symbol} {item.file_sha256}"
                            for item in formula.code_bindings
                        )
                    ),
                ]
            )
        for result_key in section.result_keys:
            lines.extend(_latex_result_table(result_map[result_key]))

    if acknowledgements:
        lines.extend([r"\section*{Acknowledgements}", _latex_escape(acknowledgements)])
    lines.extend(
        [
            r"\bibliographystyle{plain}",
            r"\bibliography{references}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def _bibtex_escape(value: str) -> str:
    return value.replace("\\", "").replace("{", r"\{").replace("}", r"\}")


def render_bibtex(citations: list[PaperCitationEvidence]) -> str:
    entries: list[str] = []
    for citation in citations:
        fields = [
            f"  title = {{{_bibtex_escape(citation.title)}}}",
            f"  author = {{{_bibtex_escape(' and '.join(citation.authors) or 'Authors unavailable')}}}",
            f"  journal = {{{_bibtex_escape(citation.venue)}}}",
        ]
        if citation.year is not None:
            fields.append(f"  year = {{{citation.year}}}")
        if citation.doi:
            fields.append(f"  doi = {{{_bibtex_escape(citation.doi)}}}")
        entries.append(
            f"@article{{{citation.paper_id},\n" + ",\n".join(fields) + "\n}"
        )
    return "\n\n".join(entries) + "\n"
