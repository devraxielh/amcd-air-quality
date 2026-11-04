"""08 — Beijing con la meteorología oculta (pre-registrado en §1.12): el régimen estacional/calefacción pasa a ser
un confusor genuinamente latente del sistema de contaminantes. Los métodos ven sólo los 6 contaminantes; la
referencia Oracle-MK es PC sobre las 10 variables residualizadas en K (bloque contaminante–contaminante).
Genera results/latent_real_scores.csv, results/latent_real_transport.csv, table_09, table_10 y fig_08."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from beijing_data import POLL, VARS, estimated_partition_nmi, load_daily, sample_dates
from utils import (ALPHA_CI, ALPHA_TEST, DOUBLE_COL, METHOD_COLORS, METHOD_V1, METHODS_MAIN, N_JOBS,
                   N_REPLICATES, OKABE_ITO, PROPOSED, RESULTS_DIR, SEED, add_panel_label, apply_mark,
                   bootstrap_ci, cliffs_delta, fmt_p, mark_best_two, rank_biserial_wilcoxon, run_method, run_oracle,
                   save_figure, save_table, standardize)

METHODS = METHODS_MAIN + [METHOD_V1]
BASELINES = [m for m in METHODS_MAIN if m != PROPOSED]
REF = "Oracle-MK"
POS_CONTROLS = [("PM2.5", "PM10"), ("NO2", "O3")]
dp = len(POLL)
iu = np.triu_indices(dp, 1)
POLL_IDX = [VARS.index(v) for v in POLL]

daily = load_daily()
STATIONS = sorted(daily.station.unique())


def reference(sub):
    """Oracle-MK: PC sobre contaminantes + meteorología residualizados en K; se devuelve el bloque de contaminantes."""
    sk, _ = run_oracle(standardize(sub[VARS].to_numpy(float)), sub.K.to_numpy(), ALPHA_CI)
    return sk[np.ix_(POLL_IDX, POLL_IDX)]


def task(r):
    rng = np.random.default_rng(32_000 + r)
    sub = sample_dates(daily, rng, STATIONS)
    X, K = sub[POLL].to_numpy(float), sub.K.to_numpy()
    rv = reference(sub)[iu]
    nmi, nmi_null, k_hat = estimated_partition_nmi(standardize(X), K, SEED + r, rng)
    out = []
    for m in METHODS + [REF]:
        if m == REF:
            v = rv
            sk = reference(sub)
        else:
            sk, _, _ = run_method(m, X, alpha=ALPHA_CI, seed=SEED + r)
            v = sk[iu]
        pc = {f"pos_{a}-{b}": int(sk[POLL.index(a), POLL.index(b)]) for a, b in POS_CONTROLS}
        out.append({"replicate": r, "model": m, "SHD_to_ref": int((v != rv).sum()), "n_edges": int(v.sum()),
                    "pos_recall": np.mean(list(pc.values())), **pc, "nmi": nmi, "nmi_null": nmi_null, "k_hat": k_hat})
    return out


def transport_task(r):
    rng = np.random.default_rng(33_000 + r)
    perm = rng.permutation(STATIONS)
    subs = [sample_dates(daily, rng, list(perm[:6])), sample_dates(daily, rng, list(perm[6:]))]
    out = []
    for m in METHODS + [REF]:
        vs = []
        for s_ in subs:
            sk = reference(s_) if m == REF else run_method(m, s_[POLL].to_numpy(float), alpha=ALPHA_CI,
                                                            seed=SEED + r)[0]
            vs.append(sk[iu])
        out.append({"replicate": r, "model": m, "SHD_between_halves": int((vs[0] != vs[1]).sum())})
    return out


sc = pd.DataFrame([x for o in Parallel(n_jobs=N_JOBS)(delayed(task)(r) for r in range(N_REPLICATES)) for x in o])
sc.to_csv(RESULTS_DIR / "latent_real_scores.csv", index=False)
tr = pd.DataFrame([x for o in Parallel(n_jobs=N_JOBS)(delayed(transport_task)(r) for r in range(N_REPLICATES))
                   for x in o])
tr.to_csv(RESULTS_DIR / "latent_real_transport.csv", index=False)

# ---------------- table_09 ----------------
g = sc[sc.model.isin(METHODS_MAIN)].groupby("model")
mk = {"F2": mark_best_two(g.SHD_to_ref.mean().to_dict(), True), "F3": mark_best_two(g.pos_recall.mean().to_dict(), False),
      "F5": mark_best_two(tr[tr.model.isin(METHODS_MAIN)].groupby("model").SHD_between_halves.mean().to_dict(), True)}
rows, tex = [], []
for m in METHODS + [REF]:
    dd, tt = sc[sc.model == m], tr[tr.model == m]
    lo, hi = bootstrap_ci(dd.SHD_to_ref)
    row = {"model": m, "F2_SHD_to_ref_mean": dd.SHD_to_ref.mean(), "F2_SHD_to_ref_sd": dd.SHD_to_ref.std(ddof=1),
           "F2_ci_low": lo, "F2_ci_high": hi, "F3_pos_recall": dd.pos_recall.mean(),
           **{c: dd[c].mean() for c in dd.columns if c.startswith("pos_") and c != "pos_recall"},
           "F4_n_edges_mean": dd.n_edges.mean(), "F5_SHD_halves_mean": tt.SHD_between_halves.mean(),
           "F5_SHD_halves_sd": tt.SHD_between_halves.std(ddof=1)}
    rows.append(row)
    f = (lambda key: None) if m in (REF, METHOD_V1) else (lambda key: mk[key].get(m))
    tex.append({"Method": m + (r"$^\dagger$" if m == REF else r"$^\ddagger$" if m == METHOD_V1 else ""),
                r"F2: SHD to Oracle-MK $\downarrow$": "reference" if m == REF else
                apply_mark(f"{row['F2_SHD_to_ref_mean']:.2f} $\\pm$ {row['F2_SHD_to_ref_sd']:.2f}", f("F2")),
                "F2 95\\% CI": "n/a" if m == REF else f"[{lo:.2f}, {hi:.2f}]",
                r"F3: pos.\ recall $\uparrow$": apply_mark(f"{row['F3_pos_recall']:.3f}", f("F3")),
                "F4: edges": f"{row['F4_n_edges_mean']:.1f}",
                r"F5: SHD halves $\downarrow$": apply_mark(f"{row['F5_SHD_halves_mean']:.2f} $\\pm$ "
                                                          f"{row['F5_SHD_halves_sd']:.2f}", f("F5"))})
t9 = pd.DataFrame(rows)
save_table(t9, "table_09_latent_regime_validation", pd.DataFrame(tex), column_format="lccccc")

# ---------------- table_10: pruebas ----------------
wide = sc.pivot(index="replicate", columns="model", values="SHD_to_ref")
tests, ps = [], []
for b in BASELINES + [METHOD_V1]:
    x, y = wide[PROPOSED].to_numpy(), wide[b].to_numpy()
    p = 1.0 if np.all(x == y) else wilcoxon(x, y).pvalue
    lo, hi = bootstrap_ci(x - y)
    tests.append({"criterion": "F2 SHD to Oracle-MK", "comparison": f"AMCD vs {b}", "delta_mean": (x - y).mean(),
                  "delta_ci_low": lo, "delta_ci_high": hi, "p": p, "rank_biserial": rank_biserial_wilcoxon(x, y),
                  "cliffs_delta": cliffs_delta(x, y), "primary": b == "PC"})
    ps.append(p)
for t, q in zip(tests, multipletests(ps, method="holm")[1]):
    t["p_holm"] = q
e1 = sc[sc.model == PROPOSED].drop_duplicates("replicate")
diff = (e1.nmi - e1.nmi_null).to_numpy()
p_e1 = 1.0 if np.all(diff == 0) else wilcoxon(diff, alternative="greater").pvalue
lo, hi = bootstrap_ci(diff)
tests.append({"criterion": "F1 context recovery", "comparison": "NMI(AMCD partition, K) − permutation null",
              "delta_mean": diff.mean(), "delta_ci_low": lo, "delta_ci_high": hi, "p": p_e1, "p_holm": p_e1,
              "rank_biserial": rank_biserial_wilcoxon(e1.nmi, e1.nmi_null),
              "cliffs_delta": cliffs_delta(e1.nmi, e1.nmi_null), "primary": False})
for t in tests:
    t["verdict"] = ("n.s." if t["p_holm"] >= ALPHA_TEST else
                    ("AMCD closer to reference" if t["delta_mean"] < 0 else "AMCD farther from reference")
                    if t["criterion"].startswith("F2") else "latent regime recovered above chance")
t10 = pd.DataFrame(tests)
tex10 = [{"Criterion": t["criterion"].split(" ")[0],
          "Comparison": t["comparison"] + (" (primary)" if t["primary"] else ""),
          r"$\Delta$ [95\% CI]": f"{t['delta_mean']:+.2f} [{t['delta_ci_low']:+.2f}, {t['delta_ci_high']:+.2f}]",
          "$p$": fmt_p(t["p"]), "$p_{\\mathrm{Holm}}$": fmt_p(t["p_holm"]), "$r_{rb}$": f"{t['rank_biserial']:+.3f}",
          "Verdict": t["verdict"]} for t in tests]
save_table(t10, "table_10_latent_regime_significance", pd.DataFrame(tex10), column_format="llccccl")

# ---------------- fig_08 ----------------
fig = plt.figure(figsize=(DOUBLE_COL, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.85, 0.75], wspace=0.85)
order = METHODS_MAIN + [METHOD_V1]
ax = fig.add_subplot(gs[0])
sns.boxplot(data=sc[sc.model.isin(order)], x="model", y="SHD_to_ref", order=order, hue="model", legend=False,
            palette=[METHOD_COLORS[m] for m in order], width=0.65, fliersize=0, linewidth=0.7, ax=ax)
sns.stripplot(data=sc[sc.model.isin(order)], x="model", y="SHD_to_ref", order=order, color="black", size=1.5,
              alpha=0.45, jitter=0.2, ax=ax)
ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=7.5)
ax.set_xlabel("")
ax.set_ylabel("SHD to Oracle-MK (F2)")
add_panel_label(ax, "a", x=-0.2)

ax = fig.add_subplot(gs[1])
pos_cols = [f"pos_{a}-{b}" for a, b in POS_CONTROLS]
H = sc[sc.model.isin(order)].groupby("model")[pos_cols].mean().reindex(order)
im = ax.imshow(H.to_numpy(), cmap="cividis", vmin=0, vmax=1, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="black" if H.iat[i, j] > 0.6 else "white")
ax.set_xticks(range(2), ["PM2.5–PM10", "NO2–O3"], rotation=30, ha="right", fontsize=7.5)
ax.set_yticks(range(len(order)), order, fontsize=7.5)
ax.grid(False)
cb = fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="9%", pad=0.05))
cb.set_label("F3 freq.", fontsize=8, labelpad=2)
cb.ax.tick_params(labelsize=7)
cb.ax.grid(False)
add_panel_label(ax, "b", x=-0.75)

ax = fig.add_subplot(gs[2])
kc = e1.k_hat.value_counts().reindex(range(1, 7), fill_value=0)
ax.bar(kc.index, kc.values, color=OKABE_ITO[5], edgecolor="black", lw=0.4)
for k, v in kc.items():
    ax.text(k, v + 1, str(v), ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(1, 7))
ax.set_xlabel(r"$k^*$ (iter. 1)")
ax.set_ylabel("Replicates")
ax.set_ylim(0, N_REPLICATES * 1.45)
ax.text(0.97, 0.97, f"NMI(K) = {e1.nmi.mean():.3f}\nnull = {e1.nmi_null.mean():.3f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=7.5)
add_panel_label(ax, "c", x=-0.45)
save_figure(fig, "fig_08_latent_regime_validation")

print(t9[["model", "F2_SHD_to_ref_mean", "F2_SHD_to_ref_sd", "F2_ci_low", "F2_ci_high", "F3_pos_recall",
          "F4_n_edges_mean", "F5_SHD_halves_mean"]].round(3).to_string())
print(t10[["comparison", "delta_mean", "delta_ci_low", "delta_ci_high", "p", "p_holm", "rank_biserial",
           "cliffs_delta", "verdict"]].to_string())
print("k_hat:", e1.k_hat.value_counts().sort_index().to_dict(), "NMI mean", round(e1.nmi.mean(), 4),
      "null", round(e1.nmi_null.mean(), 4))
print("08 OK")
