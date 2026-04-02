# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DISCLAIMER: This software is provided "as is" without any warranty,
# express or implied, including but not limited to the warranties of
# merchantability, fitness for a particular purpose, and non-infringement.
#
# In no event shall the authors or copyright holders be liable for any
# claim, damages, or other liability, whether in an action of contract,
# tort, or otherwise, arising from, out of, or in connection with the
# software or the use or other dealings in the software.
# -----------------------------------------------------------------------------
 
# @Author  : Tek Raj Chhetri
# @Email   : tekraj@mit.edu
# @Web     : https://tekrajchhetri.com/
# @File    : make_latex_table.py
# @Software: PyCharm

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


JsonLike = Union[str, Path, Dict[str, Any]]


def load_json_input(item: JsonLike) -> Dict[str, Any]:
    """
    Load one JSON object from:
      - dict already in memory
      - JSON file path
      - JSON string
    """
    if isinstance(item, dict):
        return item

    if isinstance(item, Path):
        with open(item, "r", encoding="utf-8") as f:
            return json.load(f)

    if isinstance(item, str):
        p = Path(item)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(item)

    raise TypeError(f"Unsupported input type: {type(item)}")


def latex_escape(text: str) -> str:
    text = str(text)
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
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "--"
    return f"{100 * x:.{digits}f}\\%"


def fmt_dec(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "--"
    return f"{x:.{digits}f}"


def fmt_pct_ci(ci: Optional[List[float]], digits: int = 1) -> str:
    if not ci or len(ci) != 2:
        return "--"
    return f"[{100 * ci[0]:.{digits}f}, {100 * ci[1]:.{digits}f}]"


def fmt_dec_ci(ci: Optional[List[float]], digits: int = 3) -> str:
    if not ci or len(ci) != 2:
        return "--"
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def get_block(data: Dict[str, Any], run_type: str, aggregate: str) -> Dict[str, Any]:
    """
    aggregate in {"micro", "macro"}
    """
    if aggregate == "micro":
        return data["micro_overall"][run_type]
    if aggregate == "macro":
        return data["macro_overall"][run_type]
    raise ValueError(f"Unsupported aggregate: {aggregate}")


def get_endpoint_block(data: Dict[str, Any], endpoint: str, run_type: str) -> Dict[str, Any]:
    return data["per_endpoint"][endpoint][run_type]


def maybe_bold(value_str: str, is_best: bool, tie_mode: bool = True) -> str:
    if is_best:
        return f"\\textbf{{{value_str}}}"
    return value_str


def maybe_underline(value_str: str, is_second: bool) -> str:
    if is_second:
        return f"\\underline{{{value_str}}}"
    return value_str


def find_top2_values(
    experiments: List[Dict[str, Any]],
    run_type: str,
    aggregate: str,
    metric_family: str,
    ks: List[str]
) -> Dict[str, tuple]:
    """
    Returns (best, second_best) raw numeric values for each k column.
    metric_family: "hit_rate_at" or "mrr_at"
    """
    top2 = {}
    for k in ks:
        vals = []
        for exp in experiments:
            data = load_json_input(exp["json"])
            block = get_block(data, run_type, aggregate)
            vals.append(block[metric_family][k])
        sorted_unique = sorted(set(vals), reverse=True)
        best = sorted_unique[0] if len(sorted_unique) > 0 else None
        second = sorted_unique[1] if len(sorted_unique) > 1 else None
        top2[k] = (best, second)
    return top2


def save_text(text: str, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_main_paper_table(
    experiments: List[Dict[str, Any]],
    aggregates: List[str] = ["micro", "macro"],
    run_types: List[str] = ["strict", "semantic"],
    percent_digits: int = 1,
    mrr_digits: int = 3,
    bold_best: bool = True,
    bold_second: bool = True,
    group_separators: Optional[List[int]] = None,
    caption: str = "Retrieval performance under strict and semantic matching.",
    label: str = "tab:retrieval_results_main",
) -> str:
    """
    Main paper table.
    rows = runs, columns = run_types × Hits@1/3/5 + MRR@1/3/5.
    Pass run_types=["strict"] or run_types=["semantic"] for a single-type table.
    group_separators: list of row indices after which to insert a thin rule
                      (e.g. [2, 5] adds lines after row 2 and row 5).
    """
    ks = ["1", "3", "5"]

    top2_lookup = {}
    for aggregate in aggregates:
        for run_type in run_types:
            for metric_family in ["hit_rate_at", "mrr_at"]:
                top2_lookup[(aggregate, run_type, metric_family)] = find_top2_values(
                    experiments, run_type, aggregate, metric_family, ks
                )

    # tabular spec: 2 label cols + 3 data cols per run_type
    ncols = 2 + 3 * len(run_types)
    col_spec = "ll" + "ccc" * len(run_types)

    # top header: multicolumn per run_type
    top_header_parts = " ".join(
        f"& \\multicolumn{{3}}{{c}}{{{rt.capitalize()}}}" for rt in run_types
    )
    top_header = f"Aggregate & Run {top_header_parts} \\\\"

    # sub-header: k labels per run_type
    sub_header_cols = " & ".join(
        "Hits@1 / MRR@1 & Hits@3 / MRR@3 & Hits@5 / MRR@5" for _ in run_types
    )
    sub_header = f"& & {sub_header_cols} \\\\"

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(top_header)
    lines.append(sub_header)
    lines.append("\\midrule")

    for agg_idx, aggregate in enumerate(aggregates):
        for row_idx, exp in enumerate(experiments):
            run_label = latex_escape(exp["run_label"])
            data = load_json_input(exp["json"])

            row_cells = []
            for run_type in run_types:
                block = get_block(data, run_type, aggregate)
                for k in ks:
                    hit_raw = block["hit_rate_at"][k]
                    mrr_raw = block["mrr_at"][k]

                    # Both Hits and MRR shown as percentages for consistent scale
                    hit_txt = fmt_pct(hit_raw, percent_digits)
                    mrr_txt = fmt_pct(mrr_raw, percent_digits)

                    if bold_best or bold_second:
                        best_hit, second_hit = top2_lookup[(aggregate, run_type, "hit_rate_at")][k]
                        best_mrr, second_mrr = top2_lookup[(aggregate, run_type, "mrr_at")][k]
                        if bold_best:
                            hit_txt = maybe_bold(hit_txt, best_hit is not None and abs(hit_raw - best_hit) < 1e-12)
                            mrr_txt = maybe_bold(mrr_txt, best_mrr is not None and abs(mrr_raw - best_mrr) < 1e-12)
                        if bold_second and not (bold_best and best_hit is not None and abs(hit_raw - best_hit) < 1e-12):
                            hit_txt = maybe_underline(hit_txt, second_hit is not None and abs(hit_raw - second_hit) < 1e-12)
                        if bold_second and not (bold_best and best_mrr is not None and abs(mrr_raw - best_mrr) < 1e-12):
                            mrr_txt = maybe_underline(mrr_txt, second_mrr is not None and abs(mrr_raw - second_mrr) < 1e-12)

                    # Append CI if available (micro and macro)
                    if "ci_hit_rate_at" in block:
                        hit_txt += f" {fmt_pct_ci(block['ci_hit_rate_at'].get(k), percent_digits)}"
                    if "ci_mrr_at" in block:
                        mrr_txt += f" {fmt_pct_ci(block['ci_mrr_at'].get(k), percent_digits)}"

                    row_cells.append(f"{hit_txt} / {mrr_txt}")

            agg_label = latex_escape(aggregate.capitalize()) if row_idx == 0 else ""
            lines.append(
                f"{agg_label} & {run_label} & " + " & ".join(row_cells) + r" \\"
            )

            if agg_idx == 0 and group_separators and row_idx in group_separators:
                lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

        if agg_idx != len(aggregates) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append(f"\\end{{tabular}}%")
    lines.append("}% end resizebox")
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{{latex_escape(label)}}}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def make_detailed_aggregate_table(
    experiments: List[Dict[str, Any]],
    run_type: str = "strict",
    aggregate: str = "micro",
    include_ci: bool = True,
    percent_digits: int = 1,
    mrr_digits: int = 3,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """
    One table per run_type + aggregate.
    Example: strict + micro, semantic + micro, strict + macro, semantic + macro
    """
    if caption is None:
        caption = f"{aggregate.capitalize()} results under {run_type} matching."
    if label is None:
        label = f"tab:{aggregate}_{run_type}"

    ks = ["1", "3", "5"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Run & Hits@1 & Hits@3 & Hits@5 & MRR@1 & MRR@3 & MRR@5 \\\\")
    lines.append("\\midrule")

    for exp in experiments:
        data = load_json_input(exp["json"])
        block = get_block(data, run_type, aggregate)
        run_label = latex_escape(exp["run_label"])

        row = []
        for k in ks:
            val = fmt_pct(block["hit_rate_at"].get(k), percent_digits)
            if include_ci and aggregate == "micro" and "ci_hit_rate_at" in block:
                val += f" {fmt_pct_ci(block['ci_hit_rate_at'].get(k), percent_digits)}"
            row.append(val)

        for k in ks:
            val = fmt_dec(block["mrr_at"].get(k), mrr_digits)
            if include_ci and aggregate == "micro" and "ci_mrr_at" in block:
                val += f" {fmt_dec_ci(block['ci_mrr_at'].get(k), mrr_digits)}"
            row.append(val)

        lines.append(f"{run_label} & " + " & ".join(row) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{{latex_escape(label)}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def make_endpoint_table(
    experiments: List[Dict[str, Any]],
    run_label: str,
    run_type: str = "strict",
    endpoints: List[str] = ["concept", "search", "batch"],
    percent_digits: int = 1,
    mrr_digits: int = 3,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """
    Per-endpoint breakdown for a single run label and a single matching type.
    Good for appendix / supplementary.
    """
    matching_exp = None
    for exp in experiments:
        if exp["run_label"] == run_label:
            matching_exp = exp
            break

    if matching_exp is None:
        raise ValueError(f"Run label not found: {run_label}")

    data = load_json_input(matching_exp["json"])

    if caption is None:
        caption = f"Per-endpoint results for {run_label} under {run_type} matching."
    if label is None:
        label = f"tab:endpoints_{run_label}_{run_type}"

    ks = ["1", "3", "5"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Endpoint & Hits@1 & Hits@3 & Hits@5 & MRR@1 & MRR@3 & MRR@5 \\\\")
    lines.append("\\midrule")

    for endpoint in endpoints:
        block = get_endpoint_block(data, endpoint, run_type)
        endpoint_name = latex_escape(endpoint.capitalize())

        row = []
        for k in ks:
            val = fmt_pct(block["hit_rate_at"].get(k), percent_digits)
            if "ci_hit_rate_at" in block:
                val += f" {fmt_pct_ci(block['ci_hit_rate_at'].get(k), percent_digits)}"
            row.append(val)

        for k in ks:
            val = fmt_dec(block["mrr_at"].get(k), mrr_digits)
            if "ci_mrr_at" in block:
                val += f" {fmt_dec_ci(block['ci_mrr_at'].get(k), mrr_digits)}"
            row.append(val)

        lines.append(f"{endpoint_name} & " + " & ".join(row) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{{latex_escape(label)}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_all_tables_to_txt(
    experiments: List[Dict[str, Any]],
    output_dir: Union[str, Path] = "latex_txt_tables",
) -> Dict[str, str]:
    """
    Generate a set of .txt files containing LaTeX tables.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # 1) Main combined table
    main_table = make_main_paper_table(
        experiments=experiments,
        aggregates=["micro", "macro"],
        caption=(
            "Retrieval performance across run configurations under strict and semantic matching. "
            "Each cell reports Hits@k / MRR@k. Micro aggregates all query--endpoint pairs, "
            "whereas macro averages endpoint-level scores."
        ),
        label="tab:main_strict_semantic_results",
    )
    main_path = output_dir / "main_strict_semantic_table.txt"
    save_text(main_table, main_path)
    outputs["main"] = str(main_path)

    # 2) Aggregate tables
    for aggregate in ["micro", "macro"]:
        for run_type in ["strict", "semantic"]:
            table = make_detailed_aggregate_table(
                experiments=experiments,
                run_type=run_type,
                aggregate=aggregate,
                include_ci=(aggregate == "micro"),
                caption=f"{aggregate.capitalize()} retrieval results under {run_type} matching.",
                label=f"tab:{aggregate}_{run_type}_results",
            )
            out_path = output_dir / f"{aggregate}_{run_type}_table.txt"
            save_text(table, out_path)
            outputs[f"{aggregate}_{run_type}"] = str(out_path)

    # 3) Per-endpoint appendix tables
    for exp in experiments:
        run_label = exp["run_label"]
        for run_type in ["strict", "semantic"]:
            table = make_endpoint_table(
                experiments=experiments,
                run_label=run_label,
                run_type=run_type,
                caption=f"Per-endpoint results for {run_label} under {run_type} matching.",
                label=f"tab:{run_label}_{run_type}_endpoints",
            )
            out_path = output_dir / f"endpoint_{run_label}_{run_type}.txt"
            save_text(table, out_path)
            outputs[f"endpoint_{run_label}_{run_type}"] = str(out_path)

    return outputs


_LATEX_PREAMBLE = r"""\documentclass[a4paper,10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{caption}
\usepackage{microtype}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

\begin{document}
"""

_LATEX_SUFFIX = r"""
\end{document}
"""


def build_latex_document(table_blocks: List[str]) -> str:
    """Wrap table blocks in a full compilable LaTeX document."""
    body = "\n\n\\clearpage\n\n".join(table_blocks)
    return _LATEX_PREAMBLE + body + _LATEX_SUFFIX


if __name__ == "__main__":
    import argparse
    import subprocess
    import sys as _sys

    parser = argparse.ArgumentParser(description="Generate retrieval results tables.")
    parser.add_argument(
        "--type",
        choices=["latex", "txt"],
        default="latex",
        help="Output format: 'latex' (default) compiles a single .tex + PDF; "
             "'txt' writes one .txt per group.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: 'latex_output' for latex, 'latex_txt_tables' for txt).",
    )
    parser.add_argument(
        "--split-semantic-strict",
        action="store_true",
        default=False,
        help="Generate 6 tables split by group (single/dual/ensemble) × run type "
             "instead of the default 2 tables (one strict, one semantic).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # All experiments — plain run labels; latex_escape handles _
    # ------------------------------------------------------------
    all_experiments = [
        {"run_label": "single_late_interaction", "json": "single_late_interation_evaluation_results_20260329_204307.json"},
        {"run_label": "single_biomedical",        "json": "single_biomedical_evaluation_results_20260329_204257.json"},
        {"run_label": "single_llm",               "json": "single_llm_evaluation_results_20260329_013638.json"},
        {"run_label": "dual_llm_late",            "json": "llm_late_evaluation_results_20260329_103524.json"},
        {"run_label": "dual_llm_biomedical",      "json": "llm_biomedical_evaluation_results_20260329_152838.json"},
        {"run_label": "dual_late",                "json": "dual_late_evaluation_results_20260329_011021.json"},
        {"run_label": "ensemble",                 "json": "ensemble_evaluation_results_20260329_152910.json"},
    ]

    groups = [
        {
            "name": "single",
            "experiments": [e for e in all_experiments if e["run_label"].startswith("single")],
            "caption": (
                "Retrieval performance for single-reranker configurations under strict and semantic matching. "
                "Each cell reports Hits@k / MRR@k. "
                "Micro aggregates all query--endpoint pairs; macro averages endpoint-level scores."
            ),
            "label": "tab:single_reranker_results",
        },
        {
            "name": "dual",
            "experiments": [e for e in all_experiments if e["run_label"].startswith("dual")],
            "caption": (
                "Retrieval performance for dual-reranker ensemble configurations under strict and semantic matching. "
                "Each cell reports Hits@k / MRR@k. "
                "Micro aggregates all query--endpoint pairs; macro averages endpoint-level scores."
            ),
            "label": "tab:dual_reranker_results",
        },
        {
            "name": "ensemble",
            "experiments": [e for e in all_experiments if e["run_label"].startswith("ensemble")],
            "caption": (
                "Retrieval performance for the full three-reranker ensemble under strict and semantic matching. "
                "Each cell reports Hits@k / MRR@k. "
                "Micro aggregates all query--endpoint pairs; macro averages endpoint-level scores."
            ),
            "label": "tab:ensemble_reranker_results",
        },
    ]

    run_type_labels = {
        "strict":   "Strict Matching",
        "semantic": "Semantic Matching",
    }

    def _build_blocks_split():
        """Yield (name, block) for 6 tables: 3 groups × 2 run_types."""
        for group in groups:
            if not group["experiments"]:
                print(f"  [SKIP] '{group['name']}' — no experiments")
                continue
            for rt in ["strict", "semantic"]:
                rt_label = run_type_labels[rt]
                caption = (
                    group["caption"].rstrip(".")
                    + f" ({rt_label})."
                )
                label = group["label"].replace(
                    "_results", f"_{rt}_results"
                )
                block = make_main_paper_table(
                    experiments=group["experiments"],
                    aggregates=["micro", "macro"],
                    run_types=[rt],
                    caption=caption,
                    label=label,
                )
                name = f"{group['name']}_{rt}"
                print(f"  [{name}] {len(group['experiments'])} run(s)")
                yield name, block

    def _build_blocks_default():
        """Yield (name, block) for 2 tables: one strict, one semantic (all experiments)."""
        single_count = sum(1 for e in all_experiments if e["run_label"].startswith("single"))
        dual_count = sum(1 for e in all_experiments if e["run_label"].startswith("dual"))
        separators = [single_count - 1, single_count + dual_count - 1]

        for rt in ["strict", "semantic"]:
            rt_label = run_type_labels[rt]
            caption = (
                f"Retrieval performance across all run configurations under {rt_label}. "
                "Each cell reports Hits@k / MRR@k. "
                "Micro aggregates all query--endpoint pairs; macro averages endpoint-level scores."
            )
            label = f"tab:all_results_{rt}"
            block = make_main_paper_table(
                experiments=all_experiments,
                aggregates=["micro", "macro"],
                run_types=[rt],
                caption=caption,
                label=label,
                group_separators=separators,
            )
            name = f"all_{rt}"
            print(f"  [{name}] {len(all_experiments)} run(s)")
            yield name, block

    _build_blocks = _build_blocks_split if args.split_semantic_strict else _build_blocks_default

    if args.type == "latex":
        output_dir = Path(args.output_dir or "latex_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        table_blocks: List[str] = [block for _, block in _build_blocks()]

        tex_path = output_dir / "retrieval_results.tex"
        save_text(build_latex_document(table_blocks), tex_path)
        print(f"\nWrote {tex_path}")

        # Compile to PDF (two passes so floats/refs settle)
        print("Compiling PDF...")
        compile_ok = True
        for _pass in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  pdflatex pass {_pass + 1} failed (exit {result.returncode})")
                print(result.stdout[-3000:])
                compile_ok = False
                break

        if compile_ok:
            print(f"  PDF ready → {tex_path.with_suffix('.pdf')}")
        else:
            print("  PDF compilation failed — check the .tex file for errors.")
            _sys.exit(1)

    else:  # --type txt
        output_dir = Path(args.output_dir or "latex_txt_tables")
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, block in _build_blocks():
            out_path = output_dir / f"table_{name}.txt"
            save_text(block, out_path)
            print(f"    → {out_path}")

    print("\nDone.")