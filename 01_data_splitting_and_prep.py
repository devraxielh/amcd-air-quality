"""01 — Construcción de escenarios (D_semi, D_real, D_null), verdad de terreno, esquema de
submuestreo repetido, table_01 y fig_01.

Los componentes de referencia (GMM-diag k=3 sobre todo el dataset) se usan SOLO para construir
la verdad de la simulación (aristas plantadas y des-mezcla), nunca como entrada de los métodos
evaluados (salvo el Oracle-S, declarado como cota superior)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from utils import (BASE_VARS, DATA_PATH, DOUBLE_COL, N_REPLICATES, N_SUB, OKABE_ITO, PLANTED_BETA,
                   PLANTED_EDGES, RESULTS_DIR, SCENARIOS, SEED, add_panel_label, analyze_mixture,
                   save_figure, save_table, standardize, subsample_indices, truth_matrix)

df = pd.read_csv(DATA_PATH)
X_real = df[BASE_VARS].to_numpy(float)
n_total = len(X_real)

# --- Componentes de referencia (generador de la simulación) ---
Zfull = standardize(X_real)
ref = GaussianMixture(3, covariance_type="diag", n_init=20, random_state=SEED).fit(Zfull)
labels = ref.predict(Zfull)
np.save(RESULTS_DIR / "reference_components.npy", labels)

mu = np.vstack([X_real[labels == g].mean(0) for g in range(3)])       # medias por componente
resid = X_real - mu[labels]
sd_w = resid.std(0, ddof=3)                                            # DE intra-componente agrupada

# --- D_semi: aristas plantadas dentro de cada fuente (padre centrado en su componente) ---
X_semi = X_real.copy()
for parent, child in PLANTED_EDGES:
    p, c = BASE_VARS.index(parent), BASE_VARS.index(child)
    X_semi[:, c] += PLANTED_BETA * sd_w[c] / sd_w[p] * resid[:, p]

# --- D_null: D_semi sin mezcla (se elimina la media de cada componente) ---
mu_semi = np.vstack([X_semi[labels == g].mean(0) for g in range(3)])
X_null = X_semi - mu_semi[labels] + X_semi.mean(0)

scen = {"D_semi": X_semi, "D_real": X_real, "D_null": X_null}
long = pd.concat([pd.DataFrame(v, columns=BASE_VARS).assign(scenario=k, row_id=np.arange(n_total))
                  for k, v in scen.items()], ignore_index=True)
long.to_parquet(RESULTS_DIR / "scenarios.parquet", index=False)

# --- Esquema de submuestreo repetido (mismo para todos los métodos) ---
splits = np.vstack([subsample_indices(n_total, N_SUB, r) for r in range(N_REPLICATES)])
np.save(RESULTS_DIR / "subsamples_main.npy", splits)

truth = {s: truth_matrix(s).tolist() for s in SCENARIOS}
(RESULTS_DIR / "ground_truth.json").write_text(json.dumps(
    {"vars": BASE_VARS, "planted_edges": PLANTED_EDGES, "beta": PLANTED_BETA, "truth": truth}, indent=1))

# --- table_01: resumen de escenarios ---
iu = np.triu_indices(len(BASE_VARS), 1)


def within_abs_r(X, lab):
    return np.mean([np.abs(np.corrcoef(X[lab == g], rowvar=False)[iu]).mean() for g in np.unique(lab)])


rows, rows_tex = [], []
for s, X in scen.items():
    a = analyze_mixture(standardize(X), seed=SEED)
    r_marg = np.abs(np.corrcoef(X, rowvar=False)[iu])
    T = truth_matrix(s)
    pl = list(PLANTED_EDGES)
    within_planted = [np.corrcoef(X[labels == g][:, BASE_VARS.index(a_)], X[labels == g][:, BASE_VARS.index(b_)])[0, 1]
                      for g in range(3) for a_, b_ in pl] if s != "D_real" else [np.nan]
    row = {"scenario": s, "n": n_total, "p": len(BASE_VARS), "true_edges": int(T.sum()),
           "latent_confounder": s != "D_null",
           "mean_abs_r_marginal": r_marg.mean(), "max_abs_r_marginal": r_marg.max(),
           "n_pairs_abs_r_gt_0.3": int((r_marg > 0.3).sum()),
           "mean_abs_r_within": within_abs_r(X, labels),
           "mean_r_planted_within": float(np.nanmean(within_planted)),
           "BIC_selected_k": a["k"], "BIC_selected_cov": a["cov"]}
    rows.append(row)
    rows_tex.append({"Scenario": s, "$n$": n_total, "$p$": len(BASE_VARS), "True edges": int(T.sum()),
                     "Latent $S$": "yes" if s != "D_null" else "no",
                     r"Mean $|r|$ marg.": f"{r_marg.mean():.3f}",
                     r"Mean $|r|$ within $S$": f"{row['mean_abs_r_within']:.3f}",
                     r"Pairs $|r|>0.3$": int((r_marg > 0.3).sum()),
                     r"$\hat{k}$ (BIC)": f"{a['k']} ({a['cov']})"})
t1 = pd.DataFrame(rows)
save_table(t1, "table_01_cohort_summary", pd.DataFrame(rows_tex), column_format="lrrrcrrrc")
print(t1.T)

# --- fig_01: (a) PCA coloreada por componente; (b) |r| marginal vs. intra-componente ---
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
pcs = PCA(2, random_state=SEED).fit(Zfull)
P = pcs.transform(Zfull)
markers = ["o", "s", "^"]
for g in range(3):
    m = labels == g
    axes[0].scatter(P[m, 0], P[m, 1], s=9, alpha=0.75, marker=markers[g], color=OKABE_ITO[[0, 4, 2][g]],
                    edgecolor="none", label=f"C{g + 1} ($n$={m.sum()})")
axes[0].set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0] * 100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1] * 100:.1f}%)")
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, markerscale=1.6, handletextpad=0.2,
               columnspacing=0.8, fontsize=8)
add_panel_label(axes[0], "a")

plot_rows = []
for s in ["D_real", "D_semi"]:
    X = scen[s]
    plot_rows += [{"scenario": s, "type": "Marginal", "r": v} for v in np.abs(np.corrcoef(X, rowvar=False)[iu])]
    for g in range(3):
        plot_rows += [{"scenario": s, "type": "Within component", "r": v}
                      for v in np.abs(np.corrcoef(X[labels == g], rowvar=False)[iu])]
pr = pd.DataFrame(plot_rows)
import seaborn as sns  # noqa: E402

sns.boxplot(data=pr, x="scenario", y="r", hue="type", ax=axes[1], palette=[OKABE_ITO[5], OKABE_ITO[1]],
            width=0.6, fliersize=0, linewidth=0.8)
sns.stripplot(data=pr, x="scenario", y="r", hue="type", dodge=True, ax=axes[1], size=2.2, alpha=0.5,
              palette=["#7a2e00", "#1d4f6e"], legend=False, jitter=0.18)
axes[1].set_xlabel("")
axes[1].set_xticks([0, 1], [r"$D_{\mathrm{real}}$", r"$D_{\mathrm{semi}}$"])
axes[1].set_ylabel(r"Pairwise $|r|$")
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, title=None, fontsize=8)
add_panel_label(axes[1], "b")
fig.tight_layout()
save_figure(fig, "fig_01_data_distribution_or_manifold")
print("01 OK")
