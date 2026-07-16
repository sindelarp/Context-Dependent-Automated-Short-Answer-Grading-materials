import itertools
import json
import os
import re
import sys
import traceback
import argparse
import numpy as np
import scipy
from pathlib import Path


def replace_multiple_dashes_func(x):
    """Replaces multiple dashes in a string with a single dash."""
    replace_multiple_dashes_re = re.compile(r'-+')
    return replace_multiple_dashes_re.sub("-", x).strip("-")

def save_dotplot(name, data, models, datasets, ylabel):
    import matplotlib.pyplot as plt
    import numpy as np
    present_indices = [i for i, d in enumerate(data) if d is not None]
    present_models = [models[i] for i in present_indices]

    y = np.arange(len(present_models))
    num_datasets = len(datasets)
    assert num_datasets > 0

    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 0.85, num_datasets))

    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', 'H', 'p']
    dashreg = re.compile(r'^-*')
    num_dashes = [len(dashreg.match(m).group(0)) for m in datasets]
    print(list(zip(num_dashes, datasets)))

    spacing_step = 0.08
    h = 5+len(present_models) * num_datasets * spacing_step
    plt.figure(figsize=(3, 15), dpi=150)

    offsets = [(j - num_datasets / 2) * spacing_step + spacing_step / 2 for j in range(num_datasets)]

    for j in range(num_datasets):
        x_values = [data[i][j] for i in present_indices]
        y_positions = y + offsets[j]

        plt.hlines(
            y=y_positions,
            xmin=0,
            xmax=x_values,
            color=colors[j],
            linewidth=1.5,
            alpha=0.8,
            zorder=2
        )

        dataset_map = {"counseling": "counselling"}
        d = replace_multiple_dashes_func(datasets[j])
        d = dataset_map.get(d, d)
        plt.scatter(
            x_values,
            y_positions,
            label=d,
            color=colors[j],
            marker=markers[num_dashes[j] % len(markers)],
            s=50,
            alpha=1.0,
            edgecolors='white',
            linewidth=0.5,
            zorder=3
        )

    for pos in range(len(present_models) - 1):
        plt.axhline(pos + 0.5, color='gray', linestyle='-', linewidth=0.8, alpha=0.3, zorder=1)

    plt.xlabel(f'{ylabel} (%)', fontsize=25)

    plt.yticks(y, present_models, fontsize=20)
    plt.gca().invert_yaxis()

    plt.xticks(np.arange(0, 101, 10), fontsize=10, rotation=45)
    plt.xlim(0, 102)

    legend = plt.legend(title="Datasets", loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=min(num_datasets, 5), fontsize=20)
    plt.setp(legend.get_title(),fontsize=25)

    plt.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)

    plt.savefig(name, bbox_inches='tight')
    plt.close()

def get_color_and_suround(val, min_val, max_val, values, higher_is_better):
    """
    Interpolates between 5 colors based on a value's position between min_val and max_val.
    Best (1.0) -> Dark Blue, Light Blue, Black, Light Red, Dark Red <- Worst (0.0)
    """
    colors = [
        'NavyBlue',  # Dark Blue (Worst)
        'White!50!NavyBlue',  # Light Blue
        'Black',  # Black (Middle)
        'White!50!BrickRed',  # Light Red
        'BrickRed'  # Dark Red (Best)
    ]

    if max_val == min_val:
        return 'Black', ("", "")

    # Normalize to 0-1 (1 is best, 0 is worst)
    norm = (val - min_val) / (max_val - min_val)
    if not higher_is_better:
        norm = 1 - norm
    pos = norm * (len(colors) - 1)
    rank = sorted(values, reverse=higher_is_better).index(val) + 1
    return colors[int(round(pos))], ("\\textbf{", "}") if rank < 4 else ("", "")  # Bold the best value


def format_latex_header(text):
    """
    Splits text by underscores and spaces, placing each word on a new line
    inside a bold makecell formatting block.
    """
    words = re.split(r'[_ ]+', text)
    latex_words = [f"\\textbf{{{w}}}" for w in words if w]
    return "\\makecell[b]{" + " \\\\ ".join(latex_words) + "}"


def extract_value(val_raw):
    if isinstance(val_raw, str):
        val = float(val_raw.split()[0])
        sig_override = len(val_raw.split()) > 1 and val_raw.split()[1] == "*"
    else:
        val = float(val_raw)
        sig_override = False
    return val, sig_override


def generate_latex_table(subsections, sortfunc=None, metric='balanced_accuracy', higher_is_better=True, alpha=0.05, keep_empty=True, complex_funcs=False, label=None, caption=None, do_full_table=False, repl_dict=None, data_dict = None, **kwargs):
    """
    Generates a tabularx LaTeX table comparing models across dataset subsections.
    """
    # 1. Load and merge all data from the provided files
    data = {}
    subsections_with_fps_named = {(x[0], (k, x[1])): x for k, v in subsections.items() for x in v}
    for name, sub in subsections_with_fps_named:
        data[sub] = {}
    def load_data(file_data, data):
        """Loads data from file_data to a data dict."""
        for sub, sub_data in file_data.items():
            k = (fp, sub)
            if k not in data:
                data[k] = {}
            for d in sub_data.values():
                for k2 in list(d.keys()):
                    e = "_ci_spread"
                    le = len(e)
                    if k2.endswith(e):
                        d[k2[:-le] + "_low"] = d[k2][0]
                        d[k2[:-le] + "_high"] = d[k2][1]
            data[k].update(sub_data)

    for fp in subsections:
        if data_dict is not None and fp in data_dict:
            load_data(data_dict[fp], data)
        try:
            with open(fp, 'r') as f:
                file_data = json.load(f)
            load_data(file_data, data)

        except Exception as e:
            print(f"Warning: Could not read {fp}: {e}")
            continue
    assert any([len(v) > 0 for v in data.values()]), f"No valid data found in the provided files. {subsections}"
    # 2. Identify all models present in the requested subsections_with_fps
    models = set()
    for sub in data:
        models.update(data[sub].keys())
    models = sorted(list(models))
    models = [x for x in models if repl_dict is None or x not in repl_dict or repl_dict[x] is not None]
    assert len(models) > 0
    # 3. Extract metrics and bootstrap distributions
    table_data = {name: {} for name, sub in subsections_with_fps_named}
    bootstraps = {name: {} for name, sub in subsections_with_fps_named}
    confidence_intervals = {name: {} for name, sub in subsections_with_fps_named}

    for name, sub in subsections_with_fps_named:
        if sub not in data:
            print(f"Warning: Subsection {sub} not found in data. Skipping.", file=sys.stderr)
            continue

        def print_model_missing(model):
            print(f"Warning: Model {model} not found in subsection {sub}. Skipping. {data[sub].keys()}",
                  file=sys.stderr)

        for model in models:
            custom_metric = subsections_with_fps_named[name, sub][2] if len(subsections_with_fps_named[name, sub]) > 2 and isinstance(subsections_with_fps_named[name, sub][2], str) else None
            if len(subsections_with_fps_named[name, sub]) < 3 or custom_metric is not None:
                inner_metric = custom_metric if custom_metric else metric
                if model not in data[sub]:
                    print_model_missing(model)
                    continue
                if inner_metric not in data[sub][model]:
                    print_model_missing(model)
                    continue
                val_raw = data[sub][model][inner_metric]
                if inner_metric+"_low" in data[sub][model] and inner_metric+"_high" in data[sub][model]:
                    confidence_intervals[name][model] = extract_value(data[sub][model][inner_metric+"_low"])[0], extract_value(data[sub][model][inner_metric+"_high"])[0]
            else:
                try:
                    if complex_funcs:
                        val_raw = subsections_with_fps_named[name, sub][2](
                            {sub: data[sub][model] for sub in data if model in data[sub]})
                        if val_raw is None:
                            print(
                                f"Warning: Complex function returned None for model {model} in subsection {sub}. Skipping.",
                                file=sys.stderr)
                            continue
                    else:
                        if model not in data[sub]:
                            print_model_missing(model)
                            continue
                        val_raw = subsections_with_fps_named[name, sub][2](data[sub][model])
                except Exception as e:
                    print(f"--- WARNING: NON-FATAL EXCEPTION {model} {sub}---", file=sys.stderr)
                    traceback.print_exception(type(e), e, e.__traceback__)
                    print("------------------------------------", file=sys.stderr)
                    raise e

            # Parse format like "0.307 4" into float 0.307
            val, override_sig = extract_value(val_raw)

            table_data[name][model] = val

            # Store bootstrap list if it exists
            if override_sig:
                bootstraps[name][model] = "*"
            elif model in data[sub] and 'bootstrapl' in data[sub][model]:
                bootstraps[name][model] = np.array(data[sub][model]['bootstrapl'])

    # 4. Store significance test results (only used by override in the final version)
    sig_marks = {name: {m: "" for m in models} for name, sub in subsections_with_fps_named}

    for name, sub in subsections_with_fps_named:
        if not table_data[name]:
            continue

        # Identify the best performing model for this subsection
        if higher_is_better:
            best_model = max(table_data[name].keys(), key=lambda k: table_data[name][k])
        else:
            best_model = min(table_data[name].keys(), key=lambda k: table_data[name][k])

        best_boot = bootstraps[name].get(best_model)

        # Test all other models against the best model (only used by override in the final version)
        for model in models:
            mod_boot = bootstraps[name].get(model)
            if isinstance(mod_boot, str):
                if mod_boot == "*":
                    sig_marks[name][model] = "^{*}"
            else:
                if model == best_model and best_boot is not None:
                    continue
                is_significant = False
                if is_significant:
                    sig_marks[name][model] = "^{*}"

    # 5. Build LaTeX Table
    col_format = "X | " + " ".join(["c"] * len(subsections_with_fps_named))

    latex = ([
                 "\\begin{table}[h!]",
                 "\\centering",
             ] if do_full_table else []) + [
                f"\\begin{{tabularx}}{{\\textwidth}}{{{col_format}}}",
                "\\hline"
            ]

    headers = [format_latex_header("Model Name")] + [format_latex_header(sub_name) for sub_name, sub in
                                                     subsections_with_fps_named]
    latex.append(" & ".join(headers) + " \\\\")
    latex.append("\\hline")

    if sortfunc is not None:
        models = sorted(models,
                        key=lambda x: sortfunc(x), reverse=True)

    for model in models:
        separ = r"\\\subcatarrow{}"
        system_name = model if repl_dict is None or model not in repl_dict else repl_dict[model]
        system_name_escaped = system_name
        if "/" not in system_name:
            system_name_escaped = system_name_escaped.replace(".", "/")
        system_name_escaped = system_name_escaped.replace("PRIMARY", "").replace("/_", "/").strip("/")
        system_name_escaped = system_name_escaped.replace("/", separ).replace("_", r"\-\_").strip("\\").strip("/-_ ")
        row = [system_name_escaped]
        skip = False
        for name, sub in subsections_with_fps_named:
            if model in table_data[name]:
                val = table_data[name][model]

                sub_vals = list(table_data[name].values())
                min_val, max_val = min(sub_vals), max(sub_vals)

                color_name, suround = get_color_and_suround(val, min_val, max_val, sub_vals, higher_is_better)

                mark = sig_marks[name][model]
                confidence_interval = confidence_intervals[name].get(model, None)
                # Apply textcolor, textbf, and attach significance asterisk inside math mode if any
                marks = f"${mark}$" if mark != "" else ""
                if confidence_interval is not None:
                    #We do not override the value using the confidence interval midpoint because the initial reported results which the contestants work with were calculated by one pass over the test set the same as val. We wish the results reported to match.
                    ci_mark = rf"$\pm{{}}{int(((confidence_interval[1]-confidence_interval[0])/2)*1000):03d}$"
                else:
                    ci_mark = ""
                cell = f"{suround[0]}\\textcolor{{{color_name}}}{{{val:.3f}{marks}{ci_mark}}}{suround[1]}"
                row.append(cell)
            else:
                if keep_empty:
                    row.append("NA")
                else:
                    skip = True
                    break
        if skip:
            continue
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabularx}")
    newl_ = '\\_'
    if do_full_table:
        caption = caption if caption is not None else f"Comparison of \\texttt{{{metric.replace('_', newl_)}}} across sub-datasets. Colors are scaled from best (dark blue) to worst (dark red). $^{{*}}$ indicates significantly worse than the best model ($p < {alpha}$) based on 999 bootstrap samples of balanced accuracy (chosen because it largely tracks with other metrics while being less volatile)."
        if label is not None:
            latex.append(f"\\label{{{label}}}")
        latex.append(f"\\caption{{{caption}}}")
        latex.append("\\end{table}")

    return "\n".join(latex)


def make_tables_per_lang(p="out", is_lg=False, track="rubric", **kwargs):
    is_lg_string = "_lower_granularity" if is_lg else ""
    langs_sensemaking_manual = ['all_langs'] + (['en', 'cs'] if False else [])
    for lang in langs_sensemaking_manual:
        subsections_to_compare = {
            f"{p}/testset_manual.{track}/results.json.{lang}_evaluation_results.json": (
                ("Addi manual non translated", "NotTranslated"),),
            f"{p}/pisa_testset.{track}/results.json.{lang}_evaluation_results.json": (
                ("PISA manual non translated", "NotTranslated"),)}

        try:
            latex_code = generate_latex_table(
                subsections=subsections_to_compare,
                higher_is_better=True,
                alpha=0.05,
                **kwargs
            )
        except Exception as e:
            print(f"--- WARNING: NON-FATAL EXCEPTION --- {lang}", file=sys.stderr)
            traceback.print_exception(type(e), e, e.__traceback__)
            print("------------------------------------", file=sys.stderr)
            continue
        with open(f"tables/manual.{track}.{lang}_{kwargs['metric']}{is_lg_string}.tex", "w") as f:
            f.write(latex_code)


def gn(lang, resourceg):
    resourcegs = f" ({resourceg})" if resourceg is not None else ""
    return f"{lang.capitalize()}{resourcegs}"


def make_tables_domain(basename, track="rubric", lang="all_langs", is_lg=False, **kwargs):
    lg_string = "_lower_granularity" if is_lg else ""
    subsections_to_compare = {
        f"{basename}/testset.{track}{lg_string}/results.json.{lang}_evaluation_results.json": tuple(
            (name, domain) for name, domain in
            [("Over- all", "All"), ("Legal Counsell- ing", "counselling"), ("History", "worldhist"),
             ("Consti- tutional Law", "lawf")])}

    latex_code = generate_latex_table(
        subsections=subsections_to_compare,
        metric="quadratic_cohen_kappa",
        higher_is_better=True,
        alpha=0.05,
        **kwargs
    )

    with open(f"tables/{track}_{lang}_domain_comparison_qwk{lg_string}.tex", "w") as f:
        f.write(latex_code)


def make_tables_rubric(p="out", track="rubric", lang="all_langs", is_lg=False, metric="f_pos_rate_1", **kwargs):
    types = ["Rubr Abstract", "Rubr Clean", "Rubr Orig", "Rubr NoEx"]
    lg_string = "_lower_granularity" if is_lg else ""
    subsections_to_compare = {
        f"{p}/pisa_testset.{track}{lg_string}/results.json.{lang}_evaluation_results.json": tuple(
            (name, name.replace(" ", "")) for name in types)}

    latex_code = generate_latex_table(
        subsections=subsections_to_compare,
        metric=metric,
        higher_is_better=False if metric == "f_pos_rate_1" else True,
        alpha=0.05,
        **kwargs
    )

    with open(f"tables/{track}_{lang}_{metric}_rubric_comparison_qwk{lg_string}.tex", "w") as f:
        f.write(latex_code)


def make_tables_translated_comparison(basename, track="rubric", **kwargs):
    langs_sensemaking_selection = [('all_langs', None)] + [('en', '5'), ('cs', '4'), ('da', '3')]
    subsections_to_compare = {
        f"{basename}/pisa_testset.{track}/results.json.{lang}_evaluation_results.json": (
            (gn(lang, resourceg), "NotTranslated"),) for lang, resourceg in langs_sensemaking_selection}

    latex_code = generate_latex_table(
        subsections=subsections_to_compare,
        metric="quadratic_cohen_kappa",
        higher_is_better=True,
        alpha=0.05,
        **kwargs
    )

    with open(f"tables/{track}_langs_comparison_qwk.tex", "w") as f:
        f.write(latex_code)


def make_tables_translated_paired_comparison(basename, track="rubric", **kwargs):
    langs_sensemaking_selection2 = [('pt', '4'), ('da', '3')]

    def get_vs(lang, resourceg):
        def f(x):
            return str(x[f"agreement_after_translation_{lang}_compared_qwk_significance"][2]) + (
                " *" if (x[f"agreement_after_translation_{lang}_compared_qwk_significance"][0] and
                         x[f"agreement_after_translation_{lang}_compared_qwk_significance"][
                             2] > x[f"agreement_after_translation_{lang}_compared_qwk_significance"][3]) else "")

        return (gn(lang, resourceg) + " Reference Not Translated", "All",
                lambda x: f(x)), (gn(lang, resourceg) + " Translated", "All", lambda x:
        str(x[f"agreement_after_translation_{lang}_compared_qwk_significance"][3]) + (
            " *" if (x[f"agreement_after_translation_{lang}_compared_qwk_significance"][0] and
                     x[f"agreement_after_translation_{lang}_compared_qwk_significance"][
                         3] > x[f"agreement_after_translation_{lang}_compared_qwk_significance"][2]) else ""))

    subsections_to_compare = {
        f"{basename}/pisa_testset.{track}/results.json.all_langs_evaluation_results.json": tuple(
            itertools.chain(*[get_vs(lang, resourceg) for lang, resourceg in langs_sensemaking_selection2]))}
    latex_code = generate_latex_table(
        subsections=subsections_to_compare,
        metric="quadratic_cohen_kappa",
        higher_is_better=True,
        alpha=0.05,
        complex_funcs=False,
        **kwargs
    )
    with open(f"tables/pisa.{track}.langs_pairwise_comparison_qwk.tex", "w") as f:
        f.write(latex_code)


def make_tables_rubr_kind_paired_comparison(basename, **kwargs):
    types = ["Rubr Abstract", "Rubr Orig", "Rubr NoEx"]

    def get_vs(t):
        n = t.split(" ")[1]
        t_no_space = t.replace(" ", "")
        return (f"Rubr Clean where {n}", "All",
                lambda x: str(x[f"agreement_rubric_ablation_{t_no_space}_RubrClean_compared_qwk_significance"][2]) + (
                    " *" if (x[f"agreement_rubric_ablation_{t_no_space}_RubrClean_compared_qwk_significance"][0] and
                             x[f"agreement_rubric_ablation_{t_no_space}_RubrClean_compared_qwk_significance"][
                                 2] > x[f"agreement_rubric_ablation_{t_no_space}_RubrClean_compared_qwk_significance"][
                                 3]) else "")), (t, "All", lambda x:
        str(x[f"agreement_rubric_ablation_{t_no_space}_RubrClean_compared_qwk_significance"][3]) + (
            " *" if (x[f"agreement_rubric_ablation_{t_no_space}_RubrClean_compared_qwk_significance"][0] and
                     x[f"agreement_rubric_ablation_{t_no_space}_RubrClean_compared_qwk_significance"][
                         3] > x[f"agreement_rubric_ablation_{t_no_space}_RubrClean_compared_qwk_significance"][
                         2]) else ""))

    subsections_to_compare = {
        f"{basename}/pisa_testset.rubric/results.json.all_langs_evaluation_results.json": tuple(
            itertools.chain(*[get_vs(t) for t in types]))}
    latex_code = generate_latex_table(
        subsections=subsections_to_compare,
        metric="quadratic_cohen_kappa",
        higher_is_better=True,
        alpha=0.05,
        complex_funcs=False,
        **kwargs
    )
    with open(f"tables/pisa.rubrics_pairwise_comparison_qwk.tex", "w") as f:
        f.write(latex_code)

def make_tables_rubric_agreement(basename,track="rubric", **kwargs):
    types = ["Rubr Abstract", "Rubr Orig", "Rubr NoEx"]

    def get_func(t):
        return f"agreement_rubric_ablation_{t.replace(' ', '')}_RubrClean"

    agreement_rubric_re = re.compile(r"agreement_rubric_ablation_(RubrAbstract|RubrOrig|RubrNoEx)_RubrClean$")
    def get_vs():
        return (("Avg Agree- ment", "All",
                 lambda x: np.mean(
                     [extract_value(v)[0] for k, v in x.items() if
                      agreement_rubric_re.match(k)])),) + tuple(
            (f"{t} Agree- ment", "All", get_func(t)) for t in
            types)

    subsections_to_compare = {
        f"{basename}/pisa_testset.{track}/results.json.all_langs_evaluation_results.json": get_vs()}
    latex_code = generate_latex_table(
        subsections=subsections_to_compare,
        metric="quadratic_cohen_kappa",
        higher_is_better=True,
        alpha=0.05,
        complex_funcs=False,
        **kwargs
    )
    with open(f"tables/pisa.{track}.rubric_agreement_qwk.tex", "w") as f:
        f.write(latex_code)

def make_tables_translated_agreement(basename, track="rubric", **kwargs):
    langs_sensemaking_selection3 = [('en', '5'), ('pt', '4'), ('da', '3')]

    def get_func(lang):
        return f"agreement_after_translation_{lang}"

    agreement_after_translation_re = re.compile(r"agreement_after_translation_[a-z]{2}$")
    def get_vs():
        return (("Avg Agree- ment", "All",
                 lambda x: np.mean(
                     [extract_value(v)[0] for k, v in x.items() if
                      agreement_after_translation_re.match(k)])),
                ("Min Agree- ment", "All", lambda x: min(
                    [extract_value(v)[0] for k, v in x.items() if
                     agreement_after_translation_re.match(k)]))) + tuple(
            (f"{gn(lang, category)} Agree- ment", "All", get_func(lang)) for lang, category in
            langs_sensemaking_selection3)

    subsections_to_compare = {
        f"{basename}/pisa_testset.{track}/results.json.all_langs_evaluation_results.json": get_vs()}
    latex_code = generate_latex_table(
        subsections=subsections_to_compare,
        metric="quadratic_cohen_kappa",
        higher_is_better=True,
        alpha=0.05,
        complex_funcs=False,
        **kwargs
    )
    with open(f"tables/pisa.{track}.langs_agreement_qwk.tex", "w") as f:
        f.write(latex_code)


def make_tables_scoreboard(p="out", is_lg=False, **kwargs):
    is_lg_string = "_lower_granularity" if is_lg else ""
    bn_string = f"{p}/"
    for metric in ["accuracy", "cohen_kappa", "quadratic_cohen_kappa", "balanced_accuracy"]:
        for track in ["simple", "rubric"]:
            files_score_board = [
                f"testset.{track}{is_lg_string}/results.json.all_langs_evaluation_results.json",
                f"pisa_testset.{track}{is_lg_string}/results.json.all_langs_evaluation_results.json",
                f"testset_manual.{track}{is_lg_string}/results.json.all_langs_evaluation_results.json",
                f"devset.{track}{is_lg_string}/results.json.all_langs_evaluation_results.json"]

            def getc(x, metric):
                if (files_score_board[0], "All") not in x or (files_score_board[1], "All") not in x:
                    return None

                return (extract_value(x[files_score_board[0], "All"][metric])[0] + extract_value(
                    x[files_score_board[1], "All"][metric])[0]) / 2

            map = {"pisa_testset": "PISA test set", "testset": "Addi test set", "testset_manual": "Addi test set manu",
                   "devset": "devel set"}
            files_score_board = [bn_string + x for x in files_score_board if x is not None]
            subsections_to_compare_score_board = {f: ((map[f.split("/")[-2].split(".")[0]], "All"),) for f in
                                                  files_score_board}
            subsections_to_compare_score_board[files_score_board[0]] = (("Overall results", "All",
                                                                         lambda x: getc(x, metric)),) + \
                                                                       subsections_to_compare_score_board[
                                                                           files_score_board[0]]
            latex_code = generate_latex_table(
                subsections=subsections_to_compare_score_board,
                metric=metric,
                higher_is_better=True,
                alpha=0.05,
                complex_funcs=True,
                **kwargs
            )
            with open(f"tables/latex_table_score_board_{track}{is_lg_string}_{metric}.tex", "w") as f:
                f.write(latex_code)


def get_dotplot(inpath, outpath, metric, filter1, filter2, sortfunc=None, repl_dict=None, data_dict=None, **kwargs):
    if data_dict is not None:
        results_all = data_dict[inpath]
    else:
        with open(inpath) as f:
            results_all = json.load(f)
    with open(kwargs["property_map_path"]) as f:
        prop_map = json.load(f)
    results_all = {prop_map[k] if k in prop_map else k: v for k, v in results_all.items()}
    comb_names_all = list(results_all.keys())
    model_names = sorted(
        list(set([k for k in results_all[comb_names_all[0]].keys() if metric in results_all[comb_names_all[0]][k]])),
        key=lambda x: sortfunc(x), reverse=True)

    model_names = [x for x in model_names if repl_dict is None or x not in repl_dict or repl_dict[x] is not None]
    model_names_cleaned = [(x if repl_dict is None else repl_dict.get(x.replace(".json", ""),x)).replace(".", "\n") for x in model_names]
    comb_names1 = [k for k in comb_names_all if filter1(k)] if filter1 is not None else comb_names_all
    comb_names2 = [k for k in comb_names_all if filter2(k)] if filter2 is not None else comb_names_all
    for comb_names, suffix in [(comb_names1, "_domain"), (comb_names2, "")]:
        save_dotplot(outpath + suffix + ".png",
                     [np.array([extract_value(results_all[k][model_name][metric])[0] * 100 for k in
                                comb_names]) if all([model_name in results_all[k] for k in comb_names]) else None for
                      model_name in
                      model_names],
                     model_names_cleaned, comb_names, metric.replace("_", " ").title())

def aggregate_json_files(directory_path):
    aggregated_data = {}
    base_dir = Path(directory_path)

    for file_path in base_dir.rglob("*.json"):
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    aggregated_data[str(file_path)] = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping {file_path}: Invalid JSON format.")
            except (PermissionError, OSError) as e:
                print(f"Skipping {file_path}: Unable to read file. Error: {e}")

    return aggregated_data

# =======================================================
# Example usage that recreates the figures and tables
 # =======================================================
if __name__ == "__main__":
    argparse.ArgumentParser(description="Generate tables and figures from evaluation results.")
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--store-json", action="store_true", help="Store all results in a single JSON file.")
    argument_parser.add_argument("--results-path", default="../results/results.json", help="The path to the results json file")
    argument_parser.add_argument("--property-map-path", default="../results/property_map.json", help="The path to the property map json file")
    argument_parser.add_argument("--overall-results-path", default="../results/overall_results.json", help="The path to the overall results json file")
    args = argument_parser.parse_args()
    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


    p = "results_agreement_no_languages"
    p2 = "results_agreement_no_languages"
    p3 = "results_no_agreement_languages"

    if args.store_json:
        data_dict1 = aggregate_json_files(p)
        data_dict2 = aggregate_json_files(p3)
        final_dict = data_dict1 | data_dict2
        for k,v in final_dict.items():
            for k2, v2 in v.items():
                for k3, v3 in v2.items():
                    if isinstance(v3, str):
                        continue
                    print(v3)
                    final_dict[k][k2][k3] = {k4: v4 for k4, v4 in v3.items() if "label_distribution" not in k4 and "confusion_matrix" not in k4 and "f1" not in k4 and "recall" not in k4 and "pred_distribution" not in k4 and "t_neg_" not in k4 and "prec" not in k4}
        with open(args.results_path, "w", encoding="utf-8") as f:
            json.dump(final_dict, f, ensure_ascii=False, indent=4)

    with open(args.overall_results_path) as f:
        final_order = list(json.load(f).keys())
    if os.path.exists(args.results_path):
        with open(args.results_path) as f:
            data_dict = json.load(f)
    else:
        data_dict = None

    langs_sensemaking_all = ['all_langs', 'en', 'cs', 'pt', 'de', 'hu', 'fi', 'da', 'sv', 'sr', 'el', 'ga', 'ro', 'uk']
    keep_empty = False


    def sortfunc(x):
        if x in final_order:
            r = final_order.index(x)
        else:
            r = 9999
        return -r


    repl_dict = None if False else {
                "baselines.EuroBERT-EuroBERT-210m_combo_1004_ctxy_rubn_wd5_ls5_ga64_p20__std_better_devset": "Baselines/EuroBERT-210m only trainset",
                "baselines.EuroBERT-EuroBERT-610m_combo_1003_ctxy_ruby_wd1_ls5_ga64_p20_10_std_better_devset": "baselines/EuroBERT-610m",
                                    "baselines.MoritzLaurer-mDeBERTa-v3-base-xnli-multilingual-nli-2mil7_combo_1000_ctxn_rubn_wd1_ls5_ga64_p0__std_better_devset": "baselines/mDeBERTa V3", "fianso.exeriment_qwen_8b_instruct_logreg": None, "baselines.EuroBERT-210m_separate_rubrics.json.tree_predicted": None, "baselines.gpt_oss_20b": None,
                                    "writerslogic.PRIMARY": "WritersLogic/mDeBERTa-v3 PoE",
                                    "pythoneers.PRIMARY": "Pythoneers/BGE-M3 or ModernBERT",
                                    "baselines.EuroBERT-210m_separate_rubrics": "Baselines/EuroBERT-210m",
                                    "baselines.EuroBERT-210m_no_context_separate_rubrics": "Baselines/EuroBERT-210m no context",
                                    "wse_research.gemma3_native": None,
                                    "wse_research.PRIMARY_qwen35": "WSE Research/Qwen 3.5 27B",
                                    "baselines.gemma_4_E4B_it": "Baselines/Gemma4-E4B",
                                    "fianso.experiment_qwen8b_logreg.PRIMARY": "Fianso/Qwen 3 Embedding 8B LogReg",
                                    "baselines.gemma_4_E2B_it": "Baselines/Gemma4-E2B",
                                    "baselines.multilingual-e5-large_random_forest": "Baselines/Multilingual-E5-Large Random Forest",
        "baselines.ProductOfExperts": "baselines/Product Of Experts",
        "baselines.MajorityVote": "baselines/Majority Vote",
        "baselines.gemma_4_E4B_it_validated": "Baselines/Gemma4-E4B validated",
                                    "dipf_tba.PRIMARY_tolegra_llm_conf_gt05_all": "DIPF TBA/Tolegra or LLM",
                                    }
    def_kwargs = {"keep_empty": keep_empty, "sortfunc": sortfunc, "repl_dict": repl_dict, "data_dict": data_dict, "property_map_path": args.property_map_path}
    for split in ["testset", "pisa_testset"]:
        for track in ["rubric", "simple"]:
            for lg_dotplot in [False]:
                lg_dotplot_string = "_lower_granularity" if lg_dotplot else ""
                get_dotplot(
                    inpath=f"{p}/{split}.{track}{lg_dotplot_string}/results.json.all_langs_evaluation_results.json",
                    outpath=f"./figures/{split}.{track}{lg_dotplot_string}.qwk",
                    metric="quadratic_cohen_kappa",
                    filter1=lambda x: (not x.startswith("-")) or x == "All",
                    filter2=lambda x: not (not x.startswith("-")) or x == "All", **def_kwargs)

    make_tables_scoreboard(p=p, is_lg=False, **def_kwargs)
    make_tables_per_lang(p=p, metric="quadratic_cohen_kappa", **def_kwargs)
    make_tables_per_lang(p=p, metric="accuracy", **def_kwargs)
    make_tables_per_lang(p=p, is_lg=True, metric="accuracy", **def_kwargs)
    make_tables_per_lang(p=p, track="simple", metric="accuracy", **def_kwargs)
    make_tables_per_lang(p=p, track="simple", is_lg=True, metric="accuracy", **def_kwargs)
    make_tables_rubric(p=p, **def_kwargs)
    make_tables_domain(p, **def_kwargs)
    make_tables_translated_paired_comparison(p2, track="rubric", **def_kwargs)
    make_tables_rubr_kind_paired_comparison(p2, **def_kwargs)
    make_tables_translated_agreement(p2, track="simple", **def_kwargs)
    make_tables_translated_agreement(p2, track="rubric", **def_kwargs)
    make_tables_rubric_agreement(p2, track="rubric", **def_kwargs)
    make_tables_translated_comparison(p3, **def_kwargs)