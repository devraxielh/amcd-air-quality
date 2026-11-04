"""05 — Explicabilidad del lazo MAPE-K y análisis de errores.
fig_04: (a) frecuencia de selección de aristas PC vs. AMCD (D_semi); (b) curvas ΔBIC del ANALYZE por iteración;
        (c) distribución de k* seleccionado por escenario e iteración.
fig_05: (a) frecuencia de falsos positivos por par vs. |r| marginal (D_real); (b) frecuencia de FP de AMCD vs.
        |r| intra-componente; (c) caso de estudio: SHD de AMCD según k* elegido."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

from utils import (BASE_VARS, DOUBLE_COL, METHOD_COLORS, METHOD_MARKERS, OKABE_ITO, PROPOSED, RESULTS_DIR,
                   SCENARIOS, SEED, add_panel_label, load_reference_labels, load_scenarios, save_figure,
                   standardize, truth_matrix)

edges = pd.read_parquet(RESULTS_DIR / "edge_predictions.parquet")
scores = pd.read_csv(RESULTS_DIR / "fold_scores.csv")
traces = json.loads((RESULTS_DIR / "amcd_traces.json").read_text())
S = load_scenarios()
labels = load_reference_labels()
d = len(BASE_VARS)
iu = np.triu_indices(d, 1)
pair_cols = [f"{BASE_VARS[i]}|{BASE_VARS[j]}" for i, j in zip(*iu)]
tick = ["PM2.5", "PM10", "NOx", "NO2", "SO2", "VOCs", "CO", "CO2", "CH4", "T"]


def freq_matrix(scenario, model):
    f = edges[(edges.scenario == scenario) & (edges.model == model)][pair_cols].mean().to_numpy()
    M = np.zeros((d, d))
    M[iu] = f
    return M + M.T


# ---------------- fig_04 ----------------
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

fig = plt.figure(figsize=(DOUBLE_COL, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.95], wspace=0.9)
ax = fig.add_subplot(gs[0])
cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
F_pc, F_am = freq_matrix("D_semi", "PC"), freq_matrix("D_semi", PROPOSED)
C = np.full((d, d), np.nan)
C[np.tril_indices(d, -1)] = F_pc[np.tril_indices(d, -1)]
C[iu] = F_am[iu]
cmap = plt.get_cmap("cividis")
im = ax.imshow(C, cmap=cmap, vmin=0, vmax=1)
T = truth_matrix("D_semi")
for i in range(d):
    for j in range(d):
        if i != j and (T[i, j] or T[j, i]):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=OKABE_ITO[5], lw=1.4))
ax.plot([-0.5, d - 0.5], [-0.5, d - 0.5], color="white", lw=0.8)
ax.set_xticks(range(d), tick, rotation=90, fontsize=7)
ax.set_yticks(range(d), tick, fontsize=7)
ax.grid(False)
ax.text(0.5, 1.02, "upper: AMCD  |  lower: PC", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)
cb = fig.colorbar(im, cax=cax)
cb.set_label("Selection freq.", fontsize=8, labelpad=2)
cb.ax.tick_params(labelsize=7)
cb.ax.grid(False)
add_panel_label(ax, "a", x=-0.38, y=1.02)

ax = fig.add_subplot(gs[1])
styles = {("D_semi", 0): ("-", OKABE_ITO[5], r"$D_{\mathrm{semi}}$, iter. 1"),
          ("D_semi", 1): ("--", OKABE_ITO[1], r"$D_{\mathrm{semi}}$, iter. 2"),
          ("D_null", 0): (":", OKABE_ITO[2], r"$D_{\mathrm{null}}$, iter. 1")}
for (s, it), (ls, col, lab) in styles.items():
    curves = []
    for t in traces:
        if t["scenario"] == s and len(t["trace"]) > it and t["trace"][it]["bic"] is not None:
            b = pd.DataFrame(t["trace"][it]["bic"], columns=["cov", "k", "bic"])
            best = b.groupby("k").bic.min()
            curves.append((best - best.min()).to_numpy())
    curves = np.array(curves)
    ks = np.arange(1, curves.shape[1] + 1)
    med = np.median(curves, 0)
    ax.plot(ks, med, ls=ls, color=col, marker="o", ms=3.5, label=lab)
    ax.fill_between(ks, np.percentile(curves, 5, 0), np.percentile(curves, 95, 0), color=col, alpha=0.15, lw=0)
ax.set_yscale("symlog", linthresh=10)
ax.set_xticks(range(1, 7))
ax.set_xlabel("Mixture components $k$")
ax.set_ylabel(r"$\Delta$BIC (vs. best model)")
ax.legend(fontsize=7.5, loc="lower right")
add_panel_label(ax, "b", x=-0.3, y=1.02)

ax = fig.add_subplot(gs[2])
kt = pd.DataFrame([{"scenario": t["scenario"], "iter": tr["iter"] + 1, "k": tr["k"]}
                   for t in traces for tr in t["trace"]])
cats = [(s, it) for s in SCENARIOS for it in sorted(kt[kt.scenario == s].iter.unique())]
bottom = np.zeros(len(cats))
kcol = {1: OKABE_ITO[2], 2: OKABE_ITO[3], 3: OKABE_ITO[5], 4: OKABE_ITO[4], 5: OKABE_ITO[6], 6: "0.3"}
xl = []
for kval in sorted(kt.k.unique()):
    h = np.array([((kt.scenario == s) & (kt.iter == it) & (kt.k == kval)).sum() for s, it in cats])
    ax.bar(range(len(cats)), h, bottom=bottom, color=kcol[kval], edgecolor="black", lw=0.3, label=f"$k^*$={kval}")
    bottom += h
ax.set_xticks(range(len(cats)), [f"{s.split('_')[1]}·{it}" for s, it in cats], rotation=60, fontsize=7.5)
ax.set_ylabel("Replicates")
ax.set_xlabel("Scenario · MAPE iteration", fontsize=9)
ax.set_ylim(0, 66)
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.legend(fontsize=7, loc="upper center", ncol=4, columnspacing=0.6, handlelength=1.0, handletextpad=0.3,
          bbox_to_anchor=(0.5, 1.0))
add_panel_label(ax, "c", x=-0.35, y=1.02)
save_figure(fig, "fig_04_explainability_shap_or_latent")

# ---------------- fig_05 ----------------
R = S["D_real"]
r_marg = np.abs(np.corrcoef(R, rowvar=False)[iu])
r_within = np.mean([np.abs(np.corrcoef(R[labels == g], rowvar=False)[iu]) for g in range(3)], axis=0)
fp = {m: edges[(edges.scenario == "D_real") & (edges.model == m)][pair_cols].mean().to_numpy()
      for m in ["PC", "GES", "NOTEARS", PROPOSED]}
err = pd.DataFrame({"pair": pair_cols, "r_marg": r_marg, "r_within": r_within, **{f"fp_{m}": v for m, v in fp.items()}})
err.to_csv(RESULTS_DIR / "error_by_pair_D_real.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.5))
ax = axes[0]
stats_txt = []
jit = np.random.default_rng(SEED).uniform(-0.015, 0.015, size=(4, len(r_marg)))   # sólo visual (R3.3)
for i, m in enumerate(["PC", "GES", "NOTEARS", PROPOSED]):
    ax.scatter(r_marg + jit[i] * (m != PROPOSED), fp[m] + jit[i] * 1.5, s=12 if m != PROPOSED else 34,
               marker=METHOD_MARKERS[m], color=METHOD_COLORS[m], alpha=0.55 if m != PROPOSED else 0.95,
               edgecolor="black" if m == PROPOSED else "none", lw=0.3, label=m, zorder=3 if m == PROPOSED else 2)
    rho, p = spearmanr(r_marg, fp[m])
    stats_txt.append({"model": m, "x": "r_marg", "rho": rho, "p": p})
ax.set_xlabel(r"Marginal $|r|$ of pair")
ax.set_ylabel("False-positive frequency")
ax.legend(fontsize=7, loc="upper left", handletextpad=0.1)
ax.set_ylim(-0.05, 1.05)
add_panel_label(ax, "a", x=-0.28)

ax = axes[1]
ax.scatter(r_within, fp[PROPOSED], s=30, marker="*", color=METHOD_COLORS[PROPOSED], edgecolor="black", lw=0.3)
rho, p = spearmanr(r_within, fp[PROPOSED])
stats_txt.append({"model": PROPOSED, "x": "r_within", "rho": rho, "p": p})
ax.text(0.97, 0.95, rf"$\rho_s$ = {rho:.2f}, $p$ = {p:.1e}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
for i in np.argsort(-fp[PROPOSED])[:2]:
    a_, b_ = pair_cols[i].split("|")
    ax.annotate(f"{tick[BASE_VARS.index(a_)]}–{tick[BASE_VARS.index(b_)]}", (r_within[i], fp[PROPOSED][i]),
                fontsize=7, xytext=(-4, 3), textcoords="offset points", ha="right")
ax.set_xlabel(r"Mean within-component $|r|$")
ax.set_ylabel("AMCD false-positive freq.")
add_panel_label(ax, "b", x=-0.28)

ax = axes[2]
# Caso de estudio: ¿la calidad de la partición latente (iteración 1, cross-fitted) explica los errores?
# ARI por fold (las etiquetas de folds distintos no están alineadas) frente a los componentes de referencia.
splits = np.load(RESULTS_DIR / "subsamples_main.npy")
case_rows = []
for t in traces:
    s, r = t["scenario"], t["replicate"]
    if s == "D_null":
        continue
    tr0 = t["trace"][0]
    idx = splits[r]
    Z = standardize(S[s][idx])
    aris = []
    for trn, te in KFold(2, shuffle=True, random_state=SEED + r).split(Z):
        g = GaussianMixture(tr0["k"], covariance_type=tr0["cov"], n_init=3, random_state=SEED + r,
                            reg_covar=1e-4).fit(Z[trn])
        aris.append(adjusted_rand_score(labels[idx][te], g.predict(Z[te])))
    case_rows.append({"scenario": s, "replicate": r, "k_iter1": tr0["k"], "n_iter": len(t["trace"]),
                      "ARI_partition": float(np.mean(aris))})
case = pd.DataFrame(case_rows).merge(scores[scores.model == PROPOSED][["scenario", "replicate", "SHD", "FP", "FN"]],
                                     on=["scenario", "replicate"])
case.to_csv(RESULTS_DIR / "amcd_case_study.csv", index=False)
for s, mk, col in [("D_semi", "o", OKABE_ITO[5]), ("D_real", "s", OKABE_ITO[0])]:
    c = case[case.scenario == s]
    jit = np.random.default_rng(SEED).uniform(-0.12, 0.12, len(c))
    ax.scatter(c.ARI_partition, c.SHD + jit, s=12, marker=mk, color=col, alpha=0.8, edgecolor="black", lw=0.2,
               label={"D_semi": r"$D_{\mathrm{semi}}$", "D_real": r"$D_{\mathrm{real}}$"}[s])
    rho, p = spearmanr(c.ARI_partition, c.SHD)
    stats_txt.append({"model": PROPOSED, "x": f"ARI_partition_{s}", "rho": rho, "p": p})
ax.set_xlabel("ARI of latent partition (iter. 1)")
ax.set_ylabel("AMCD skeleton SHD")
ax.legend(fontsize=7, loc="upper left")
add_panel_label(ax, "c", x=-0.22)
fig.tight_layout()
save_figure(fig, "fig_05_error_calibration_or_case_study")

pd.DataFrame(stats_txt).to_csv(RESULTS_DIR / "error_spearman.csv", index=False)
print(pd.DataFrame(stats_txt).to_string())
print(err.sort_values("fp_AMCD", ascending=False).head(6).to_string())
print(case.groupby(["scenario", "n_iter"]).agg(n=("SHD", "size"), SHD=("SHD", "mean"), ARI=("ARI_partition", "mean"), ARI_min=("ARI_partition", "min")).to_string())
print(kt.groupby(["scenario", "iter", "k"]).size().to_string())
print("05 OK")
