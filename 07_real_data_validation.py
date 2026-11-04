"""07 — Validación externa con datos reales (Beijing Multi-Site, UCI id 501). Diseño pre-registrado en §1.11.
No existe grafo verdadero: criterios proxy E1–E5 frente a Oracle-K (PC condicionado al régimen conocido).
Genera results/real_scores.csv, results/real_transport.csv, table_07, table_08 y fig_07."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from utils import (ALPHA_CI, ALPHA_TEST, DOUBLE_COL, METHOD_COLORS, METHOD_V1, METHODS_MAIN, N_JOBS,
                   N_REPLICATES, N_SUB, PROPOSED, RESULTS_DIR, SEED, add_panel_label,
                   apply_mark, bootstrap_ci, cliffs_delta, fmt_p, mark_best_two,
                   rank_biserial_wilcoxon, run_method, run_oracle, save_figure, save_table, standardize)
from beijing_data import K_NAMES, VARS, estimated_partition_nmi, load_daily  # noqa: E402
from beijing_data import sample_dates as _sample_dates  # noqa: E402

LABELS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "T", "P", "Td", "WS"]
POS_CONTROLS = [("PM2.5", "PM10"), ("TEMP", "DEWP"), ("NO2", "O3")]
METHODS = METHODS_MAIN + [METHOD_V1]
BASELINES = [m for m in METHODS_MAIN if m != PROPOSED]
REF = "Oracle-K"
d = len(VARS)
iu = np.triu_indices(d, 1)

# ---------------- Carga y preprocesamiento determinista (§1.11; código compartido en beijing_data.py) --------
daily = load_daily()
STATIONS = sorted(daily.station.unique())


def sample_dates(rng, stations, n=N_SUB):
    return _sample_dates(daily, rng, stations, n)


def skel_vec(skel):
    return skel[iu]


def task(r):
    rng = np.random.default_rng(30_000 + r)
    sub = sample_dates(rng, STATIONS)
    X, K = sub[VARS].to_numpy(float), sub.K.to_numpy()
    Z = standardize(X)
    ref_skel, _ = run_oracle(Z, K, ALPHA_CI)
    rv = skel_vec(ref_skel)
    nmi, nmi_null, k_hat = estimated_partition_nmi(Z, K, SEED + r, rng)
    out = []
    for m in METHODS + [REF]:
        if m == REF:
            sk = ref_skel
        else:
            sk, _, _ = run_method(m, X, alpha=ALPHA_CI, seed=SEED + r)
        v = skel_vec(sk)
        pc = {f"pos_{a}-{b}": int(sk[VARS.index(a), VARS.index(b)]) for a, b in POS_CONTROLS}
        out.append({"replicate": r, "model": m, "SHD_to_ref": int((v != rv).sum()), "n_edges": int(v.sum()),
                    "pos_recall": np.mean(list(pc.values())), **pc, "nmi": nmi, "nmi_null": nmi_null,
                    "k_hat": k_hat, **{f"e_{VARS[i]}|{VARS[j]}": bool(sk[i, j]) for i, j in zip(*iu)}})
    return out


def transport_task(r):
    """E5: dos mitades disjuntas de estaciones; SHD entre los grafos aprendidos en cada mitad."""
    rng = np.random.default_rng(31_000 + r)
    perm = rng.permutation(STATIONS)
    halves = [list(perm[:6]), list(perm[6:])]
    subs = [sample_dates(rng, h) for h in halves]
    out = []
    for m in METHODS + [REF]:
        sks = []
        for s_ in subs:
            X, K = s_[VARS].to_numpy(float), s_.K.to_numpy()
            if m == REF:
                sk, _ = run_oracle(standardize(X), K, ALPHA_CI)
            else:
                sk, _, _ = run_method(m, X, alpha=ALPHA_CI, seed=SEED + r)
            sks.append(skel_vec(sk))
        out.append({"replicate": r, "model": m, "SHD_between_halves": int((sks[0] != sks[1]).sum())})
    return out


sc = pd.DataFrame([x for o in Parallel(n_jobs=N_JOBS)(delayed(task)(r) for r in range(N_REPLICATES)) for x in o])
sc.to_csv(RESULTS_DIR / "real_scores.csv", index=False)
tr = pd.DataFrame([x for o in Parallel(n_jobs=N_JOBS)(delayed(transport_task)(r) for r in range(N_REPLICATES))
                   for x in o])
tr.to_csv(RESULTS_DIR / "real_transport.csv", index=False)
pd.DataFrame({"n_station_days": [len(daily)], "n_dates": [daily.date.nunique()], "n_stations": [len(STATIONS)],
              **{f"share_{k}": [(daily.K == i).mean()] for i, k in enumerate(K_NAMES)}}).to_csv(
    RESULTS_DIR / "real_data_summary.csv", index=False)

# ---------------- table_07: resultados por método ----------------
g = sc[sc.model != REF].groupby("model")
mk = {"E2": mark_best_two(g.SHD_to_ref.mean().reindex(METHODS_MAIN).to_dict(), True),
      "E3": mark_best_two(g.pos_recall.mean().reindex(METHODS_MAIN).to_dict(), False),
      "E5": mark_best_two(tr[tr.model.isin(METHODS_MAIN)].groupby("model").SHD_between_halves.mean().to_dict(), True)}
rows, tex = [], []
for m in METHODS + [REF]:
    dd, tt = sc[sc.model == m], tr[tr.model == m]
    lo, hi = bootstrap_ci(dd.SHD_to_ref)
    row = {"model": m, "E2_SHD_to_ref_mean": dd.SHD_to_ref.mean(), "E2_SHD_to_ref_sd": dd.SHD_to_ref.std(ddof=1),
           "E2_ci_low": lo, "E2_ci_high": hi, "E3_pos_recall": dd.pos_recall.mean(),
           **{c: dd[c].mean() for c in dd.columns if c.startswith("pos_") and c != "pos_recall"},
           "E4_n_edges_mean": dd.n_edges.mean(), "E5_SHD_halves_mean": tt.SHD_between_halves.mean(),
           "E5_SHD_halves_sd": tt.SHD_between_halves.std(ddof=1)}
    rows.append(row)
    f = (lambda key: None) if m in (REF, METHOD_V1) else (lambda key: mk[key].get(m))
    tex.append({"Method": m + (r"$^\dagger$" if m == REF else r"$^\ddagger$" if m == METHOD_V1 else ""),
                r"E2: SHD to Oracle-K $\downarrow$": "reference" if m == REF else
                apply_mark(f"{row['E2_SHD_to_ref_mean']:.2f} $\\pm$ {row['E2_SHD_to_ref_sd']:.2f}", f("E2")),
                "E2 95\\% CI": "n/a" if m == REF else f"[{lo:.2f}, {hi:.2f}]",
                r"E3: pos.\ recall $\uparrow$": apply_mark(f"{row['E3_pos_recall']:.3f}", f("E3")),
                r"E4: edges": f"{row['E4_n_edges_mean']:.1f}",
                r"E5: SHD halves $\downarrow$": apply_mark(f"{row['E5_SHD_halves_mean']:.2f} $\\pm$ "
                                                          f"{row['E5_SHD_halves_sd']:.2f}", f("E5"))})
t7 = pd.DataFrame(rows)
save_table(t7, "table_07_real_data_validation", pd.DataFrame(tex), column_format="lccccc")

# ---------------- table_08: pruebas (E2 primaria, E1) ----------------
wide = sc.pivot(index="replicate", columns="model", values="SHD_to_ref")
tests, ps = [], []
for b in BASELINES + [METHOD_V1]:
    x, y = wide[PROPOSED].to_numpy(), wide[b].to_numpy()
    p = 1.0 if np.all(x == y) else wilcoxon(x, y).pvalue
    lo, hi = bootstrap_ci(x - y)
    tests.append({"criterion": "E2 SHD to Oracle-K", "comparison": f"AMCD vs {b}", "delta_mean": (x - y).mean(),
                  "delta_ci_low": lo, "delta_ci_high": hi, "p": p, "rank_biserial": rank_biserial_wilcoxon(x, y),
                  "cliffs_delta": cliffs_delta(x, y), "primary": b == "PC"})
    ps.append(p)
ph = multipletests(ps, method="holm")[1]
for t, q in zip(tests, ph):
    t["p_holm"] = q
e1 = sc[sc.model == PROPOSED].drop_duplicates("replicate")
diff = e1.nmi - e1.nmi_null
p_e1 = 1.0 if np.all(diff == 0) else wilcoxon(diff, alternative="greater").pvalue
lo, hi = bootstrap_ci(diff)
tests.append({"criterion": "E1 context recovery", "comparison": "NMI(AMCD partition, K) − permutation null",
              "delta_mean": diff.mean(), "delta_ci_low": lo, "delta_ci_high": hi, "p": p_e1, "p_holm": p_e1,
              "rank_biserial": rank_biserial_wilcoxon(e1.nmi, e1.nmi_null), "cliffs_delta": cliffs_delta(e1.nmi, e1.nmi_null),
              "primary": False})
t8 = pd.DataFrame(tests)
for t in tests:
    t["verdict"] = ("n.s." if t["p_holm"] >= ALPHA_TEST else
                    ("AMCD closer to reference" if t["delta_mean"] < 0 else "AMCD farther from reference")
                    if t["criterion"].startswith("E2") else "context recovered above chance")
t8 = pd.DataFrame(tests)
tex8 = [{"Criterion": t["criterion"].split(" ")[0], "Comparison": t["comparison"] + (" (primary)" if t["primary"] else ""),
         r"$\Delta$ [95\% CI]": f"{t['delta_mean']:+.2f} [{t['delta_ci_low']:+.2f}, {t['delta_ci_high']:+.2f}]",
         "$p$": fmt_p(t["p"]), "$p_{\\mathrm{Holm}}$": fmt_p(t["p_holm"]), "$r_{rb}$": f"{t['rank_biserial']:+.3f}",
         "Verdict": t["verdict"]} for t in tests]
save_table(t8, "table_08_real_data_significance", pd.DataFrame(tex8), column_format="llccccl")

# ---------------- fig_07 ----------------
fig = plt.figure(figsize=(DOUBLE_COL, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.7], wspace=0.85)
ax = fig.add_subplot(gs[0])
order = METHODS_MAIN + [METHOD_V1]
sns.boxplot(data=sc[sc.model.isin(order)], x="model", y="SHD_to_ref", order=order, hue="model", legend=False,
            palette=[METHOD_COLORS[m] for m in order], width=0.65, fliersize=0, linewidth=0.7, ax=ax)
sns.stripplot(data=sc[sc.model.isin(order)], x="model", y="SHD_to_ref", order=order, color="black", size=1.5,
              alpha=0.45, jitter=0.2, ax=ax)
ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=7.5)
ax.set_xlabel("")
ax.set_ylabel("SHD to Oracle-K (E2)")
add_panel_label(ax, "a", x=-0.2)

ax = fig.add_subplot(gs[1])
pos_cols = [f"pos_{a}-{b}" for a, b in POS_CONTROLS]
H = sc[sc.model.isin(order)].groupby("model")[pos_cols].mean().reindex(order)
im = ax.imshow(H.to_numpy(), cmap="cividis", vmin=0, vmax=1, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="black" if H.iat[i, j] > 0.6 else "white")
ax.set_xticks(range(3), ["PM2.5–PM10", "T–Td", "NO2–O3"], rotation=30, ha="right", fontsize=7.5)
ax.set_yticks(range(len(order)), order, fontsize=7.5)
ax.grid(False)
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

cb = fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="7%", pad=0.05))
cb.set_label("E3 freq.", fontsize=8, labelpad=2)
cb.ax.grid(False)
cb.ax.tick_params(labelsize=7)
add_panel_label(ax, "b", x=-0.55)

ax = fig.add_subplot(gs[2])
# E1 se reporta en table_08; cuando k*=1 en todas las réplicas el NMI es 0 por definición, por lo que el panel
# muestra E5 (estabilidad entre mitades disjuntas de estaciones), que sí es informativo.
sel = ["PC", "GES", "NOTEARS", PROPOSED, REF]
sns.boxplot(data=tr[tr.model.isin(sel)], x="model", y="SHD_between_halves", order=sel, hue="model", legend=False,
            palette=[METHOD_COLORS.get(m, "0.4") for m in sel], width=0.6, fliersize=0, linewidth=0.7, ax=ax)
sns.stripplot(data=tr[tr.model.isin(sel)], x="model", y="SHD_between_halves", order=sel, color="black", size=1.5,
              alpha=0.45, jitter=0.18, ax=ax)
ax.set_xticks(range(len(sel)), sel, rotation=60, ha="right", fontsize=7.5)
ax.set_xlabel("")
ax.set_ylabel("SHD between station halves (E5)")
add_panel_label(ax, "c", x=-0.45)
save_figure(fig, "fig_07_real_data_validation")

print(t7[["model", "E2_SHD_to_ref_mean", "E2_SHD_to_ref_sd", "E3_pos_recall", "E4_n_edges_mean",
          "E5_SHD_halves_mean"]].round(3).to_string())
print(t8[["criterion", "comparison", "delta_mean", "delta_ci_low", "delta_ci_high", "p", "p_holm",
          "rank_biserial", "verdict"]].to_string())
print("k_hat distribution:", e1.k_hat.value_counts().sort_index().to_dict())
print("07 OK")
