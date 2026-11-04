"""04 — Pruebas de hipótesis: Friedman + Nemenyi (rangos, CD), Wilcoxon AMCD vs. cada baseline con Holm,
tamaños de efecto (rango-biserial, Cliff's δ), McNemar exacto sobre la recuperación de la v-estructura.
Genera table_04 y la figura suplementaria fig_S1_critical_difference."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import binomtest, friedmanchisquare, studentized_range, wilcoxon
from statsmodels.stats.multitest import multipletests

from utils import (ALPHA_TEST, DOUBLE_COL, METHOD_COLORS, METHODS_MAIN, PROPOSED, RESULTS_DIR, SCENARIOS,
                   TABLES_DIR, cliffs_delta, fmt_p, rank_biserial_wilcoxon, save_figure, save_table)

scores = pd.read_csv(RESULTS_DIR / "fold_scores.csv")
rows, rows_tex, friedman_rows = [], [], []

for s in SCENARIOS:
    wide = scores[(scores.scenario == s) & scores.model.isin(METHODS_MAIN)].pivot(
        index="replicate", columns="model", values="SHD")[METHODS_MAIN]
    N, k = wide.shape
    chi2, p_f = friedmanchisquare(*[wide[m] for m in METHODS_MAIN])
    ranks = wide.rank(axis=1, method="average").mean()
    q_alpha = studentized_range.ppf(1 - ALPHA_TEST, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    nem = sp.posthoc_nemenyi_friedman(wide.to_numpy())
    nem.index = nem.columns = METHODS_MAIN
    nem.to_csv(RESULTS_DIR / f"nemenyi_{s}.csv")
    friedman_rows.append({"scenario": s, "chi2": chi2, "p": p_f, "N": N, "k": k, "CD": cd,
                          **{f"rank_{m}": ranks[m] for m in METHODS_MAIN}})
    rows.append({"scenario": s, "comparison": f"Friedman ({k} methods)", "statistic": chi2, "p": p_f, "p_holm": np.nan,
                 "effect_size": np.nan, "effect_name": "", "interpretation":
                 f"{'Reject' if p_f < ALPHA_TEST else 'Do not reject'} H0 of equal ranks; CD={cd:.3f}"})

    comps, ps = [], []
    for b in [m for m in METHODS_MAIN if m != PROPOSED]:
        x, y = wide[PROPOSED].to_numpy(), wide[b].to_numpy()
        if np.all(x == y):
            stat, p = np.nan, 1.0
        else:
            res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            stat, p = res.statistic, res.pvalue
        comps.append((b, stat, p, rank_biserial_wilcoxon(x, y), cliffs_delta(x, y), np.median(x - y),
                      nem.loc[PROPOSED, b]))
        ps.append(p)
    p_holm = multipletests(ps, method="holm")[1]
    for (b, stat, p, rb, cdl, med, p_nem), ph in zip(comps, p_holm):
        direction = "AMCD better" if med < 0 or (med == 0 and rb < 0) else ("AMCD worse" if (med > 0 or rb > 0) else "no difference")
        verdict = ("Reject H0" if ph < ALPHA_TEST else "Do not reject H0") + f" ({direction})"
        rows.append({"scenario": s, "comparison": f"AMCD vs {b} (Wilcoxon)", "statistic": stat, "p": p, "p_holm": ph,
                     "effect_size": rb, "effect_name": "rank-biserial r", "cliffs_delta": cdl,
                     "median_diff_SHD": med, "p_nemenyi": p_nem, "interpretation": verdict})

    # McNemar exacto sobre la v-estructura (sólo escenarios con aristas plantadas)
    if s != "D_real":
        v = scores[(scores.scenario == s) & scores.model.isin([PROPOSED, "PC"])].pivot(
            index="replicate", columns="model", values="vstruct")
        b01 = int(((v[PROPOSED] == 1) & (v["PC"] == 0)).sum())
        b10 = int(((v[PROPOSED] == 0) & (v["PC"] == 1)).sum())
        p_mc = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
        rows.append({"scenario": s, "comparison": "AMCD vs PC v-structure (exact McNemar)", "statistic": b01,
                     "p": p_mc, "p_holm": np.nan, "effect_size": (b01 - b10) / max(b01 + b10, 1),
                     "effect_name": "(b01-b10)/(b01+b10)",
                     "interpretation": f"discordant AMCD-only={b01}, PC-only={b10}; "
                                       f"{'Reject' if p_mc < ALPHA_TEST else 'Do not reject'} H0"})

t4 = pd.DataFrame(rows)
pd.DataFrame(friedman_rows).to_csv(RESULTS_DIR / "friedman_ranks.csv", index=False)
tex = []
for _, r in t4.iterrows():
    eff = "--" if pd.isna(r.effect_size) else f"{r.effect_size:+.3f}"
    tex.append({"Scenario": r.scenario, "Comparison": r.comparison,
                "Statistic": "--" if pd.isna(r.statistic) else f"{r.statistic:.3f}",
                "$p$": fmt_p(r.p), "$p_{\\mathrm{Holm}}$": "--" if pd.isna(r.p_holm) else fmt_p(r.p_holm),
                "Effect": eff, "Cliff's $\\delta$": "--" if pd.isna(r.get("cliffs_delta", np.nan)) else f"{r.cliffs_delta:+.3f}",
                "Interpretation": r.interpretation})
save_table(t4, "table_04_statistical_significance", pd.DataFrame(tex), column_format="llcccccp{4.2cm}")
print(t4.drop(columns=["effect_name"]).to_string())
print(pd.DataFrame(friedman_rows).T.to_string())

# ---------------- fig_S1: diagramas de Diferencia Crítica (Demšar, 2006) ----------------
fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(DOUBLE_COL, 1.75 * len(SCENARIOS)))
for ax, fr, lab in zip(axes, friedman_rows, "abc"):
    ranks = pd.Series({m: fr[f"rank_{m}"] for m in METHODS_MAIN}).sort_values()
    k, cd = fr["k"], fr["CD"]
    # Rango 1 (mejor) a la izquierda; mitad mejor rotulada a la izquierda, mitad peor a la derecha.
    ax.set_xlim(1, k)
    ax.set_ylim(0, 1)
    ax.spines[["left", "right", "bottom"]].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.set_xticks(range(1, k + 1))
    ax.set_yticks([])
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=8)
    n_left = int(np.ceil(len(ranks) / 2))
    for i, (m, rk) in enumerate(ranks.items()):
        left = i < n_left
        y = 0.74 - 0.14 * (i if left else len(ranks) - 1 - i)
        ax.plot([rk, rk], [1, y], color=METHOD_COLORS[m], lw=1, clip_on=False)
        ax.plot([rk, 1 if left else k], [y, y], color=METHOD_COLORS[m], lw=1, clip_on=False)
        ax.text(0.95 if left else k + 0.05, y, f"{m} ({rk:.2f})", ha="right" if left else "left",
                va="center", fontsize=8, clip_on=False)
    # Barras que unen métodos no distinguibles por Nemenyi (Δrango < CD)
    vals = ranks.to_numpy()
    cliques = []
    for i in range(len(vals)):
        j = max(jj for jj in range(len(vals)) if vals[jj] - vals[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c_i, (a, b) in enumerate(cliques):
        yb = 0.9 - 0.06 * c_i
        ax.plot([vals[a] - 0.04, vals[b] + 0.04], [yb, yb], color="black", lw=2.2, solid_capstyle="butt")
    ax.plot([k - cd, k], [0.06, 0.06], color="black", lw=1)
    ax.text(k - cd / 2, 0.1, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.text(-0.06, 1.15, f"({lab})", transform=ax.transAxes, fontweight="bold", fontsize=11)
fig.tight_layout()
save_figure(fig, "fig_S1_critical_difference", despine=False)
print("04 OK")
