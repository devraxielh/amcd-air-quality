"""06 — Estudio de robustez pre-registrado (manifiesto §1.9; responde a R1.2 y R2.1).
Cada réplica es un dataset generado de forma independiente (n=400, p=10) bajo 8 condiciones que violan,
una a la vez, los supuestos de AMCD. Genera results/robustness_scores.csv, table_05, table_06 y fig_06."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from utils import (ALPHA_CI, ALPHA_TEST, BASE_VARS, DOUBLE_COL, METHOD_ORACLE, METHOD_V1, N_JOBS, N_REPLICATES, N_SUB,
                   OKABE_ITO, PLANTED_BETA, PLANTED_EDGES, PROPOSED, RESULTS_DIR, SEED, add_panel_label,
                   apply_mark, bootstrap_ci, cliffs_delta, evaluate, fmt_p, load_reference_labels, load_scenarios,
                   mark_best_two, rank_biserial_wilcoxon, run_method, run_pc, save_figure, save_table, standardize)

CONDITIONS = [("C0", "Reference"), ("C1", "Laplace noise"), ("C2", "Heteroscedastic sources"),
              ("C3", "Five sources"), ("C4", "Overlapping sources"), ("C5", "Continuous mixing"),
              ("C6", "Weak edges"), ("C7", "No latent source")]
METHODS = ["Empty", "MargCorr", "PC", "GES", "FCI", "NOTEARS", "JCI-PC", "CD-NOD", "AMCD", METHOD_V1, METHOD_ORACLE]
BASELINES = [m for m in METHODS if m not in (PROPOSED, METHOD_V1, METHOD_ORACLE)]
# Bloques de semillas (manifiesto §1.10): desarrollo 10 000+ (usado para diagnosticar), prueba 20 000+ (confirmatorio)
SEED_BLOCKS = {"dev": 10_000, "test": 20_000}

# --- Perfiles de fuente calibrados con los datos observados (estandarizados) ---
X_real = load_scenarios()["D_real"]
lab = load_reference_labels()
Zr = standardize(X_real)
PROFILES = np.vstack([Zr[lab == g].mean(0) for g in range(3)])
SD_W = (Zr - PROFILES[lab]).std(0, ddof=3)
d = len(BASE_VARS)


def generate(cid: str, r: int, block: str = "dev"):
    """Devuelve X (n×p), etiqueta verdadera (o None) y pesos de mezcla (o None)."""
    rng = np.random.default_rng(SEED_BLOCKS[block] + 100 * int(cid[1]) + r)
    n, beta, profiles, sd_mult = N_SUB, PLANTED_BETA, PROFILES, np.ones(3)
    if cid == "C3":
        g = np.random.default_rng(SEED)                    # perfiles extra fijos entre réplicas
        w = g.dirichlet(np.ones(3), size=2)
        profiles = np.vstack([PROFILES, w @ PROFILES + g.normal(0, 0.5, size=(2, d))])
    if cid == "C4":
        profiles = PROFILES * 0.4
    if cid == "C6":
        beta = 0.3
    if cid == "C2":
        sd_mult = np.array([0.6, 1.0, 1.6])
    k = len(profiles)
    if cid == "C7":
        labels, W, mean = np.zeros(n, int), None, np.zeros((n, d))
    elif cid == "C5":
        W = rng.dirichlet([0.5, 0.5, 0.5], size=n)
        labels, mean = None, W @ PROFILES
    else:
        labels, W = rng.integers(0, k, size=n), None
        mean = profiles[labels]
    if cid == "C1":
        E = rng.laplace(0, 1 / np.sqrt(2), size=(n, d)) * SD_W
    else:
        E = rng.normal(0, 1, size=(n, d)) * SD_W
    if cid == "C2":
        E = E * sd_mult[labels][:, None]
    for parent, child in PLANTED_EDGES:
        p, c = BASE_VARS.index(parent), BASE_VARS.index(child)
        E[:, c] += beta * SD_W[c] / SD_W[p] * E[:, p]
    return mean + E, labels, W


def oracle_continuous(X, W, alpha):
    Z = standardize(X)
    A = np.column_stack([np.ones(len(W)), W[:, 1:]])
    R = Z - A @ np.linalg.lstsq(A, Z, rcond=None)[0]
    return run_pc(standardize(R), alpha)


def task(cid, r, block):
    X, labels, W = generate(cid, r, block)
    out = []
    for m in METHODS:
        if m == METHOD_ORACLE and cid == "C5":
            skel, directed = oracle_continuous(X, W, ALPHA_CI)
        else:
            skel, directed, _ = run_method(m, X, alpha=ALPHA_CI, seed=SEED + r, labels=labels)
        out.append({"block": block, "condition": cid, "replicate": r, "model": m, **evaluate(skel, directed, "D_semi" if cid != "C7"
                                                                             else "D_null")})
    return out


rows = Parallel(n_jobs=N_JOBS)(delayed(task)(c, r, b) for b in SEED_BLOCKS for c, _ in CONDITIONS
                               for r in range(N_REPLICATES))
allsc = pd.DataFrame([x for o in rows for x in o])
allsc[allsc.block == "dev"].to_csv(RESULTS_DIR / "robustness_scores_dev.csv", index=False)
sc = allsc[allsc.block == "test"].drop(columns="block").reset_index(drop=True)   # confirmatorio
sc.to_csv(RESULTS_DIR / "robustness_scores.csv", index=False)

# ---------------- table_05: SHD por condición × método ----------------
summ = sc.groupby(["condition", "model"]).agg(SHD_mean=("SHD", "mean"), SHD_sd=("SHD", "std"),
                                              F1_mean=("F1", "mean"), vstruct_rate=("vstruct", "mean")).reset_index()
t5_rows, t5_tex = [], []
for cid, cname in CONDITIONS:
    s = summ[summ.condition == cid].set_index("model")
    marks = mark_best_two({m: s.loc[m, "SHD_mean"] for m in METHODS if m not in (METHOD_ORACLE, METHOD_V1)},
                          lower_is_better=True)
    tex = {"ID": cid, "Condition": cname}
    for m in METHODS:
        v = s.loc[m]
        lo, hi = bootstrap_ci(sc[(sc.condition == cid) & (sc.model == m)].SHD)
        t5_rows.append({"condition": cid, "condition_name": cname, "model": m, "SHD_mean": v.SHD_mean,
                        "SHD_sd": v.SHD_sd, "SHD_ci_low": lo, "SHD_ci_high": hi, "F1_mean": v.F1_mean,
                        "vstruct_rate": v.vstruct_rate})
        txt = f"{v.SHD_mean:.2f}"
        tex[m + (r"$^\dagger$" if m == METHOD_ORACLE else r"$^\ddagger$" if m == METHOD_V1 else "")] = \
            apply_mark(txt, marks.get(m))
    t5_tex.append(tex)
save_table(pd.DataFrame(t5_rows), "table_05_robustness_results", pd.DataFrame(t5_tex),
           column_format="ll" + "c" * len(METHODS))

# ---------------- table_06: pruebas por condición ----------------
t6, t6_tex = [], []
for cid, cname in CONDITIONS:
    wide = sc[sc.condition == cid].pivot(index="replicate", columns="model", values="SHD")
    ranked = [m for m in METHODS if m not in (METHOD_ORACLE, METHOD_V1)]
    chi2, pf = friedmanchisquare(*[wide[m] for m in ranked])
    best_b = min(BASELINES, key=lambda m: wide[m].mean())
    comps = []
    for b in BASELINES + [METHOD_V1]:
        x, y = wide[PROPOSED].to_numpy(), wide[b].to_numpy()
        p = 1.0 if np.all(x == y) else wilcoxon(x, y).pvalue
        comps.append((b, p, rank_biserial_wilcoxon(x, y), cliffs_delta(x, y), (x - y).mean(), *bootstrap_ci(x - y)))
    ph = multipletests([c[1] for c in comps], method="holm")[1]
    for (b, p, rb, cd, md, lo, hi), phb in zip(comps, ph):
        verdict = ("AMCD better" if md < 0 else "AMCD worse" if md > 0 else "tie") if phb < ALPHA_TEST else "n.s."
        t6.append({"condition": cid, "condition_name": cname, "baseline": b, "is_best_baseline": b == best_b,
                   "delta_SHD_mean": md, "delta_ci_low": lo, "delta_ci_high": hi, "wilcoxon_p": p, "p_holm": phb,
                   "rank_biserial": rb, "cliffs_delta": cd, "friedman_chi2": chi2, "friedman_p": pf,
                   "verdict": verdict})
    for (b, p, rb, cd, md, lo, hi), phb in zip(comps, ph):
        if b in ("PC", best_b, "JCI-PC", METHOD_V1):
            t6_tex.append({"ID": cid, "Baseline": b + (" (best)" if b == best_b else ""),
                           r"$\Delta$SHD [95\% CI] $\downarrow$": f"{md:+.2f} [{lo:+.2f}, {hi:+.2f}]",
                           "$p_{\\mathrm{Holm}}$": fmt_p(phb), "$r_{rb}$": f"{rb:+.3f}", "Cliff's $\\delta$": f"{cd:+.3f}",
                           "Verdict": "AMCD better" if (phb < ALPHA_TEST and md < 0) else
                           ("AMCD worse" if (phb < ALPHA_TEST and md > 0) else "n.s.")})
t6 = pd.DataFrame(t6)
# eliminar duplicados (PC puede ser también el mejor baseline o JCI-PC)
t6_tex = pd.DataFrame(t6_tex).drop_duplicates(subset=["ID", "Baseline"])
save_table(t6, "table_06_robustness_significance", t6_tex, column_format="llccccc")

# ---------------- fig_06 ----------------
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
M = summ.pivot(index="condition", columns="model", values="SHD_mean").loc[[c for c, _ in CONDITIONS], METHODS]
im = ax.imshow(np.log1p(M.to_numpy()), cmap="cividis_r", aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.iat[i, j]
        ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                color="white" if np.log1p(v) > np.log1p(M.to_numpy()).max() * 0.55 else "black")
ax.set_xticks(range(len(METHODS)), [m.replace("Oracle-S", "Oracle") for m in METHODS], rotation=60, ha="right",
              fontsize=7.5)
ax.set_yticks(range(len(CONDITIONS)), [f"{c}  {n}" for c, n in CONDITIONS], fontsize=7.5)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
ticks = [0, 1, 3, 10, 30]
cb.set_ticks(np.log1p(ticks), labels=[str(t) for t in ticks])
cb.set_label("Mean skeleton SHD", fontsize=8)
cb.ax.tick_params(labelsize=7)
add_panel_label(ax, "a", x=-0.42, y=1.02)

ax = axes[1]
best = t6[t6.is_best_baseline & (t6.baseline != METHOD_V1)].set_index("condition").loc[[c for c, _ in CONDITIONS]]
y = np.arange(len(best))[::-1]
col = [OKABE_ITO[5] if (p < ALPHA_TEST and m < 0) else OKABE_ITO[0] if (p < ALPHA_TEST and m > 0) else "0.6"
       for p, m in zip(best.p_holm, best.delta_SHD_mean)]
ax.errorbar(best.delta_SHD_mean, y, xerr=[best.delta_SHD_mean - best.delta_ci_low,
                                          best.delta_ci_high - best.delta_SHD_mean],
            fmt="none", ecolor="0.3", elinewidth=0.9, capsize=2.5)
ax.scatter(best.delta_SHD_mean, y, c=col, s=30, zorder=3, edgecolor="black", linewidth=0.4)
ax.axvline(0, color="0.4", ls=":", lw=0.8)
ax.set_yticks(y, [f"{c} vs {b}" for c, b in zip(best.index, best.baseline)], fontsize=7.5)
ax.set_xlabel(r"$\Delta$SHD (AMCD $-$ best baseline)")
handles = [plt.Line2D([], [], marker="o", ls="", color=OKABE_ITO[5], mec="black", mew=0.4, label="AMCD better"),
           plt.Line2D([], [], marker="o", ls="", color=OKABE_ITO[0], mec="black", mew=0.4, label="AMCD worse"),
           plt.Line2D([], [], marker="o", ls="", color="0.6", mec="black", mew=0.4, label="n.s. (Holm)")]
ax.legend(handles=handles, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=0.6,
          handletextpad=0.2)
add_panel_label(ax, "b", x=-0.5, y=1.02)
fig.tight_layout()
save_figure(fig, "fig_06_robustness_generators")

print(M.round(2).to_string())
print(t6[t6.is_best_baseline | (t6.baseline == "PC")][["condition", "baseline", "delta_SHD_mean", "delta_ci_low",
                                                       "delta_ci_high", "p_holm", "rank_biserial", "verdict"]]
      .to_string())
print("06 OK")
