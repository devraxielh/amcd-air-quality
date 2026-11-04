"""02 — Comparación principal: 7 métodos + Oracle-S × 3 escenarios × 50 submuestras (n=400).
Guarda results/fold_scores.csv, results/edge_predictions.parquet (análogo OOF: adyacencias estimadas
por réplica), results/amcd_traces.json; genera table_02 y fig_02."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from utils import (ALPHA_CI, BASE_VARS, DOUBLE_COL, METHOD_COLORS, METHOD_ORACLE, METHOD_V1, METHODS_MAIN, N_JOBS,
                   PROPOSED, apply_mark, mark_best_two, RESULTS_DIR, SCENARIOS, SEED, add_panel_label, amcd, bootstrap_ci, evaluate,
                   load_reference_labels, load_scenarios, run_method, save_figure, save_table, standardize)

S = load_scenarios()
labels = load_reference_labels()
splits = np.load(RESULTS_DIR / "subsamples_main.npy")
METHODS = METHODS_MAIN + [METHOD_V1, METHOD_ORACLE]
iu = np.triu_indices(len(BASE_VARS), 1)


def task(scenario, method, r):
    idx = splits[r]
    skel, directed, t = run_method(method, S[scenario][idx], alpha=ALPHA_CI, seed=SEED + r, labels=labels[idx])
    rec = {"scenario": scenario, "model": method, "replicate": r, "n": len(idx), **evaluate(skel, directed, scenario),
           "runtime_s": t}
    edges = {"scenario": scenario, "model": method, "replicate": r,
             **{f"{BASE_VARS[i]}|{BASE_VARS[j]}": bool(skel[i, j]) for i, j in zip(*iu)},
             **{f"{BASE_VARS[i]}->{BASE_VARS[j]}": bool(directed[i, j])
                for i in range(len(BASE_VARS)) for j in range(len(BASE_VARS)) if i != j}}
    return rec, edges


jobs = [(s, m, r) for s in SCENARIOS for m in METHODS for r in range(splits.shape[0])]
out = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(task)(*j) for j in jobs)
scores = pd.DataFrame([o[0] for o in out])
edges = pd.DataFrame([o[1] for o in out])
scores.to_csv(RESULTS_DIR / "fold_scores.csv", index=False)
edges.to_parquet(RESULTS_DIR / "edge_predictions.parquet", index=False)


# Traza MAPE-K de AMCD (KNOWLEDGE) por réplica y escenario
def trace_task(scenario, r):
    _, _, tr = amcd(standardize(S[scenario][splits[r]]), seed=SEED + r, return_trace=True)
    return {"scenario": scenario, "replicate": r,
            "trace": [{k: (v if k != "bic" else [[c, kk, float(b)] for c, kk, b in v]) for k, v in t.items()}
                      for t in tr]}


traces = Parallel(n_jobs=N_JOBS)(delayed(trace_task)(s, r) for s in SCENARIOS for r in range(splits.shape[0]))
(RESULTS_DIR / "amcd_traces.json").write_text(json.dumps(traces))

# ---------------- table_02 ----------------
rows, rows_tex = [], []
for s in SCENARIOS:
    sub = scores[scores.scenario == s]
    g = sub[~sub.model.isin([METHOD_ORACLE, METHOD_V1])].groupby("model")
    # Mejor (negrita) y segundo (subrayado) por columna, sin el Oracle; dirección de cada métrica (R3.1)
    mk = {"SHD": mark_best_two(g.SHD.mean().to_dict(), True), "FP": mark_best_two(g.FP.mean().to_dict(), True),
          "FN": mark_best_two(g.FN.mean().to_dict(), True), "F1": mark_best_two(g.F1.mean().to_dict(), False),
          "VS": mark_best_two(g.vstruct.mean().to_dict(), False),
          "T": mark_best_two(g.runtime_s.mean().to_dict(), True)}
    for m in METHODS:
        d = sub[sub.model == m]
        lo, hi = bootstrap_ci(d.SHD)
        flo, fhi = bootstrap_ci(d.F1)
        row = {"scenario": s, "model": m, "SHD_mean": d.SHD.mean(), "SHD_sd": d.SHD.std(ddof=1),
               "SHD_ci_low": lo, "SHD_ci_high": hi, "FP_mean": d.FP.mean(), "FN_mean": d.FN.mean(),
               "precision_mean": d.precision.mean(), "recall_mean": d.recall.mean(),
               "F1_mean": d.F1.mean(), "F1_sd": d.F1.std(ddof=1), "F1_ci_low": flo, "F1_ci_high": fhi,
               "vstruct_rate": d.vstruct.mean(), "runtime_mean_s": d.runtime_s.mean()}
        rows.append(row)
        mm = (lambda key: None) if m in (METHOD_ORACLE, METHOD_V1) else (lambda key: mk[key].get(m))
        rows_tex.append({
            "Scenario": s.replace("_", r"\_") if m == METHODS[0] else "",
            "Method": m + (r"$^\dagger$" if m == METHOD_ORACLE else r"$^\ddagger$" if m == METHOD_V1 else ""),
            r"SHD $\downarrow$": apply_mark(f"{row['SHD_mean']:.3f} $\\pm$ {row['SHD_sd']:.3f}", mm("SHD")),
            "SHD 95\\% CI": f"[{lo:.3f}, {hi:.3f}]",
            r"FP $\downarrow$": apply_mark(f"{row['FP_mean']:.3f}", mm("FP")),
            r"FN $\downarrow$": "n/a" if s == "D_real" else apply_mark(f"{row['FN_mean']:.3f}", mm("FN")),
            r"F1 $\uparrow$": "n/a" if s == "D_real" else
            apply_mark(f"{row['F1_mean']:.3f} $\\pm$ {row['F1_sd']:.3f}", mm("F1")),
            r"V-struct. $\uparrow$": "n/a" if s == "D_real" else apply_mark(f"{row['vstruct_rate']:.3f}", mm("VS")),
            r"Time (s) $\downarrow$": apply_mark(f"{row['runtime_mean_s']:.3f}", mm("T"))})
t2 = pd.DataFrame(rows)
save_table(t2, "table_02_main_results_cv", pd.DataFrame(rows_tex), column_format="llccccccc")
print(t2[["scenario", "model", "SHD_mean", "SHD_sd", "SHD_ci_low", "SHD_ci_high", "F1_mean", "vstruct_rate",
          "runtime_mean_s"]].to_string())

# ---------------- fig_02: SHD por réplica en los 3 escenarios ----------------
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.9), sharey=False)
order = METHODS
pal = [METHOD_COLORS[m] for m in order]
for ax, s, lab in zip(axes, SCENARIOS, "abc"):
    d = scores[scores.scenario == s]
    sns.boxplot(data=d, x="model", y="SHD", order=order, palette=pal, ax=ax, width=0.65, fliersize=0,
                linewidth=0.7, hue="model", legend=False)
    sns.stripplot(data=d, x="model", y="SHD", order=order, color="black", size=1.6, alpha=0.45, jitter=0.2, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Skeleton SHD" if s == "D_semi" else "")
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8)
    ax.set_yscale("symlog", linthresh=2)
    ax.set_yticks([0, 1, 2, 5, 10, 20, 45], ["0", "1", "2", "5", "10", "20", "45"])
    ax.set_ylim(-0.3, 50)
    ax.text(0.97, 0.97, {"D_semi": r"$D_{\mathrm{semi}}$", "D_real": r"$D_{\mathrm{real}}$",
                          "D_null": r"$D_{\mathrm{null}}$"}[s], transform=ax.transAxes, ha="right", va="top")
    add_panel_label(ax, lab, x=-0.2)
fig.tight_layout()
save_figure(fig, "fig_02_main_performance_comparison")
print("02 OK")
