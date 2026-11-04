"""03 — Ablación de componentes del lazo MAPE-K y sensibilidad a n y α.
Mismas submuestras y semillas que 02. Genera results/ablation_scores.csv, results/sensitivity_scores.csv,
table_03 y fig_03."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import (ALPHA_CI, DOUBLE_COL, METHOD_COLORS, METHOD_MARKERS, N_JOBS, N_REPLICATES, N_SUB,
                   OKABE_ITO, RESULTS_DIR, SEED, add_panel_label, apply_mark, bootstrap_ci, evaluate,
                   load_reference_labels, load_scenarios, mark_best_two, rank_biserial_wilcoxon, run_method,
                   save_figure, save_table, subsample_indices)
from scipy.stats import wilcoxon

S = load_scenarios()
labels = load_reference_labels()
splits = np.load(RESULTS_DIR / "subsamples_main.npy")
n_total = S["D_semi"].shape[0]

ABLATIONS = [
    ("A0a", "AMCD (full)", "D_semi", {}),
    ("A1", "w/o cross-fitting", "D_semi", {"crossfit": False}),
    ("A2", "w/o Analyze (fixed k=2)", "D_semi", {"k_fixed": 2}),
    ("A3", "w/o Analyze (fixed k=4)", "D_semi", {"k_fixed": 4}),
    ("A4a", "w/o re-planning", "D_semi", {"replan": False}),
    ("A6a", "v1 Analyze (full mixtures)", "D_semi", {"full_k_max": 6}),
    ("A0b", "AMCD (full)", "D_real", {}),
    ("A4b", "w/o re-planning", "D_real", {"replan": False}),
    ("A6b", "v1 Analyze (full mixtures)", "D_real", {"full_k_max": 6}),
    ("A0c", "AMCD (full)", "D_null", {}),
    ("A5", "w/o trigger (forced k=3)", "D_null", {"k_fixed": 3}),
    ("A6c", "v1 Analyze (full mixtures)", "D_null", {"full_k_max": 6}),
]
REFERENCE = {"D_semi": "A0a", "D_real": "A0b", "D_null": "A0c"}


def abl_task(aid, name, scen, kw, r):
    idx = splits[r]
    skel, directed, t = run_method("AMCD", S[scen][idx], alpha=ALPHA_CI, seed=SEED + r, **kw)
    return {"ablation": aid, "variant": name, "scenario": scen, "replicate": r,
            **evaluate(skel, directed, scen), "runtime_s": t}


abl = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(abl_task)(*a, r) for a in ABLATIONS for r in range(N_REPLICATES)))
abl.to_csv(RESULTS_DIR / "ablation_scores.csv", index=False)

# Mejor / segundo SHD y F1 dentro de cada bloque de escenario (R3.1)
blk = abl.groupby(["scenario", "ablation"]).agg(SHD=("SHD", "mean"), F1=("F1", "mean")).reset_index()
MK = {s: {"SHD": mark_best_two(dict(zip(b.ablation, b.SHD)), True),
          "F1": mark_best_two(dict(zip(b.ablation, b.F1)), False)} for s, b in blk.groupby("scenario")}
rows, rows_tex = [], []
for aid, name, scen, _ in ABLATIONS:
    ref_id = REFERENCE[scen]
    d = abl[abl.ablation == aid].sort_values("replicate")
    ref = abl[abl.ablation == ref_id].sort_values("replicate")
    delta = d.SHD.to_numpy() - ref.SHD.to_numpy()
    lo, hi = bootstrap_ci(delta)
    if aid in REFERENCE.values() or np.all(delta == 0):
        p, rb = np.nan, 0.0
    else:
        p = wilcoxon(d.SHD, ref.SHD).pvalue
        rb = rank_biserial_wilcoxon(d.SHD, ref.SHD)
    row = {"ablation": aid, "variant": name, "scenario": scen, "SHD_mean": d.SHD.mean(), "SHD_sd": d.SHD.std(ddof=1),
           "delta_SHD_mean": delta.mean(), "delta_ci_low": lo, "delta_ci_high": hi, "F1_mean": d.F1.mean(),
           "vstruct_rate": d.vstruct.mean(), "wilcoxon_p": p, "rank_biserial": rb}
    rows.append(row)
    rows_tex.append({"ID": aid, "Variant": name, "Scenario": scen.replace("_", r"\_"),
                     r"SHD $\downarrow$": apply_mark(f"{row['SHD_mean']:.3f} $\\pm$ {row['SHD_sd']:.3f}",
                                                    MK[scen]["SHD"].get(aid)),
                     r"$\Delta$SHD [95\% CI] $\downarrow$": "reference" if aid in REFERENCE.values()
                     else f"{delta.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]",
                     r"F1 $\uparrow$": "n/a" if scen == "D_real" else apply_mark(f"{row['F1_mean']:.3f}",
                                                                                 MK[scen]["F1"].get(aid)),
                     r"V-struct. $\uparrow$": "n/a" if pd.isna(row["vstruct_rate"]) else f"{row['vstruct_rate']:.3f}",
                     "$p$ (Wilcoxon)": "n/a" if np.isnan(p) else f"{p:.2e}",
                     "$r_{rb}$": "n/a" if aid in REFERENCE.values() else f"{rb:+.3f}"})
t3 = pd.DataFrame(rows)
save_table(t3, "table_03_ablation_study", pd.DataFrame(rows_tex), column_format="lllcccccc")
print(t3.to_string())

# ---------------- Sensibilidad ----------------
N_GRID = [100, 200, 300, 400, 450]
A_GRID = [0.001, 0.005, 0.01, 0.05, 0.1]
SENS_N_METHODS = ["PC", "GES", "NOTEARS", "AMCD"]
SENS_A_METHODS = ["PC", "FCI", "AMCD"]


def sens_task(kind, method, n, a, r):
    idx = subsample_indices(n_total, n, r) if n != N_SUB else splits[r]
    skel, directed, _ = run_method(method, S["D_semi"][idx], alpha=a, seed=SEED + r, labels=labels[idx])
    return {"kind": kind, "model": method, "n": n, "alpha": a, "replicate": r, **evaluate(skel, directed, "D_semi")}


jobs = [("n", m, n, ALPHA_CI, r) for m in SENS_N_METHODS for n in N_GRID for r in range(N_REPLICATES)]
jobs += [("alpha", m, N_SUB, a, r) for m in SENS_A_METHODS for a in A_GRID for r in range(N_REPLICATES)]
sens = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(sens_task)(*j) for j in jobs))
sens.to_csv(RESULTS_DIR / "sensitivity_scores.csv", index=False)

# ---------------- fig_03 ----------------
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
ax = axes[0]
plot = t3[~t3.ablation.isin(list(REFERENCE.values()))].reset_index(drop=True)
ypos = np.arange(len(plot))[::-1]
scol = {"D_semi": OKABE_ITO[5], "D_real": OKABE_ITO[0], "D_null": OKABE_ITO[4]}
cols = [scol[s] for s in plot.scenario]
ax.errorbar(plot.delta_SHD_mean, ypos,
            xerr=[plot.delta_SHD_mean - plot.delta_ci_low, plot.delta_ci_high - plot.delta_SHD_mean],
            fmt="none", ecolor="0.3", elinewidth=0.9, capsize=2.5)
ax.scatter(plot.delta_SHD_mean, ypos, c=cols, s=28, zorder=3, edgecolor="black", linewidth=0.4)
ax.axvline(0, color="0.4", lw=0.8, ls=":")
ax.set_yticks(ypos, [f"{a}: {v} [{s.split('_')[1]}]" for a, v, s in
                     zip(plot.ablation, plot.variant, plot.scenario)], fontsize=7)
ax.set_xlabel(r"$\Delta$SHD vs. full AMCD")
add_panel_label(ax, "a", x=-0.95)

for ax, kind, xcol, methods, lab in [(axes[1], "n", "n", SENS_N_METHODS, "b"),
                                       (axes[2], "alpha", "alpha", SENS_A_METHODS, "c")]:
    d = sens[sens.kind == kind]
    for m in methods:
        g = d[d.model == m].groupby(xcol).SHD
        mean = g.mean()
        ci = np.array([bootstrap_ci(v.to_numpy()) for _, v in g])
        ax.plot(mean.index, mean.values, marker=METHOD_MARKERS[m], color=METHOD_COLORS[m], label=m, lw=1.2,
                ms=5 if m != "AMCD" else 8, ls="-" if m == "AMCD" else "--")
        ax.fill_between(mean.index, ci[:, 0], ci[:, 1], color=METHOD_COLORS[m], alpha=0.18, lw=0)
    ax.set_ylabel("Skeleton SHD" if lab == "b" else "")
    if kind == "alpha":
        ax.set_xscale("log")
        ax.set_xlabel(r"CI-test level $\alpha$")
    else:
        ax.set_xlabel("Subsample size $n$")
    ax.set_ylim(bottom=-0.5)
    ax.legend(loc="upper left", fontsize=8, handlelength=1.8)
    add_panel_label(ax, lab, x=-0.2)
fig.tight_layout()
save_figure(fig, "fig_03_ablation_and_sensitivity")
print(sens.groupby(["kind", "model", "n", "alpha"]).SHD.mean().unstack(0).to_string())
print("03 OK")
