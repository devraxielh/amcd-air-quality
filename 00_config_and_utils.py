"""00_config_and_utils.py — configuración global, estilo Q1, utilidades estadísticas y
biblioteca de métodos de descubrimiento causal (incluido el método propuesto AMCD).

Paper: Autonomic (MAPE-K) causal discovery under latent emission-source confounding.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # determinismo + sin sobre-suscripción
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import contextlib
import io
import random
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
import seaborn as sns
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Reproducibilidad y constantes del experimento
# ----------------------------------------------------------------------------
SEED = 42
N_BOOTSTRAP = 2000
N_REPLICATES = 50          # submuestreos repetidos (análogo a CV para aprendizaje de estructura)
N_SUB = 400                # tamaño de submuestra principal (80 % de 500)
ALPHA_CI = 0.01            # nivel de las pruebas de independencia condicional
ALPHA_TEST = 0.05          # nivel de las pruebas de hipótesis del paper
N_JOBS = -1


def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


set_global_seed()

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
DATA_PATH = PROJECT_DIR / "dataset_ml_ready.csv"
RESULTS_DIR = SCRIPTS_DIR / "results"
TABLES_DIR = SCRIPTS_DIR / "tables"
FIGURES_DIR = SCRIPTS_DIR / "figures"
for _d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BASE_VARS = ["pm2_5", "pm10", "nox", "no2", "so2", "vocs", "co", "co2", "ch4", "temperature"]
VAR_LABELS = {"pm2_5": r"PM$_{2.5}$", "pm10": r"PM$_{10}$", "nox": r"NO$_x$", "no2": r"NO$_2$",
              "so2": r"SO$_2$", "vocs": "VOCs", "co": "CO", "co2": r"CO$_2$", "ch4": r"CH$_4$",
              "temperature": "T"}
# Aristas plantadas dentro de cada fuente (padre, hijo); nox->no2<-co forma una v-estructura.
PLANTED_EDGES = [("nox", "no2"), ("co", "no2"), ("pm2_5", "pm10"), ("so2", "ch4")]
PLANTED_BETA = 0.6
VSTRUCT = (("nox", "no2"), ("co", "no2"))
SCENARIOS = ["D_semi", "D_real", "D_null"]

METHODS_MAIN = ["Empty", "MargCorr", "PC", "GES", "FCI", "NOTEARS", "JCI-PC", "CD-NOD", "AMCD"]
METHOD_ORACLE = "Oracle-S"
METHOD_V1 = "AMCD-v1"            # ablación: AMCD original (§1.5)
PROPOSED = "AMCD"

# ----------------------------------------------------------------------------
# Estilo gráfico Q1 (sin títulos)
# ----------------------------------------------------------------------------
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
METHOD_COLORS = {"Empty": "#999999", "MargCorr": OKABE_ITO[3], "PC": OKABE_ITO[1], "GES": OKABE_ITO[0],
                 "FCI": OKABE_ITO[6], "NOTEARS": OKABE_ITO[2], "JCI-PC": OKABE_ITO[4], "CD-NOD": "#882255",
                 "AMCD": OKABE_ITO[5], "AMCD-v1": "#E8A07A", "Oracle-S": OKABE_ITO[7]}
METHOD_MARKERS = {"Empty": "x", "MargCorr": "v", "PC": "o", "GES": "s", "FCI": "D", "NOTEARS": "^",
                  "JCI-PC": "h", "CD-NOD": "p", "AMCD": "*", "AMCD-v1": "X", "Oracle-S": "P"}
SINGLE_COL = 3.5
DOUBLE_COL = 7.0


def set_publication_style(family: str = "serif") -> None:
    sns.set_theme(context="paper", style="ticks", palette=OKABE_ITO)
    plt.rcParams.update({
        "font.family": family,
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix" if family == "serif" else "dejavusans",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


set_publication_style()


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.05) -> None:
    """Etiqueta de subfigura (a), (b)… en la esquina superior izquierda."""
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


def _strip_titles(fig) -> None:
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    for ax in fig.get_axes():
        ax.set_title("")
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")


def save_figure(fig, name: str, formats=("pdf", "png"), dpi: int = 600, despine: bool = True) -> list[Path]:
    """Guarda en vectorial + PNG 600 DPI, eliminando cualquier título residual."""
    _strip_titles(fig)
    if despine:
        for ax in fig.get_axes():
            sns.despine(ax=ax)
    paths = []
    for fmt in formats:
        p = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------
# Tablas
# ----------------------------------------------------------------------------
def _latex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def save_table(df: pd.DataFrame, name: str, df_tex: pd.DataFrame | None = None,
               index: bool = False, column_format: str | None = None) -> None:
    """CSV con valores completos + .tex booktabs (df_tex permite una versión formateada)."""
    # Celdas no aplicables explícitas (sin NaN ni vacíos) — R3.1
    df.replace("", np.nan).to_csv(TABLES_DIR / f"{name}.csv", index=index, na_rep="n/a")
    tex_df = (df_tex if df_tex is not None else df).copy()
    # Los encabezados que ya traen LaTeX (math o comandos) se respetan; el resto se escapa.
    tex_df.columns = [c if ("\\" in str(c) or "$" in str(c)) else _latex_escape(c) for c in tex_df.columns]
    for c in tex_df.select_dtypes(include="object").columns:
        tex_df[c] = tex_df[c].map(lambda v: v if "\\" in str(v) else _latex_escape(v))
    tex = tex_df.to_latex(index=index, escape=False, column_format=column_format,
                          float_format="%.3f")
    (TABLES_DIR / f"{name}.tex").write_text(tex, encoding="utf-8")


def fmt_mean_sd(values, decimals: int = 3) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.{decimals}f} $\\pm$ {v.std(ddof=1):.{decimals}f}"


def mark_best_two(values: dict, lower_is_better: bool = True, decimals: int = 3) -> dict:
    """Devuelve {clave: 'best' | 'second' | None} comparando valores redondeados (empates comparten marca)."""
    r = {k: round(float(v), decimals) for k, v in values.items() if v is not None and np.isfinite(v)}
    uniq = sorted(set(r.values()), reverse=not lower_is_better)
    return {k: ("best" if uniq and r.get(k) == uniq[0] else "second" if len(uniq) > 1 and r.get(k) == uniq[1]
                else None) for k in values}


def apply_mark(text: str, mark) -> str:
    return f"\\textbf{{{text}}}" if mark == "best" else f"\\underline{{{text}}}" if mark == "second" else text


def fmt_p(p: float) -> str:
    return "$<10^{-300}$" if p == 0 else f"{p:.2e}"


# ----------------------------------------------------------------------------
# Estadística
# ----------------------------------------------------------------------------
def bootstrap_ci(values, stat=np.mean, n_boot: int = N_BOOTSTRAP,
                 alpha: float = 0.05, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.apply_along_axis(stat, 1, v[idx])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def rank_biserial_wilcoxon(x, y) -> float:
    """Tamaño del efecto r de rango-biserial para Wilcoxon pareado (x - y)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cliffs_delta(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (x.size * y.size))


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def load_scenarios() -> dict[str, np.ndarray]:
    df = pd.read_parquet(RESULTS_DIR / "scenarios.parquet")
    return {s: df[df.scenario == s][BASE_VARS].to_numpy(float) for s in SCENARIOS}


def load_reference_labels() -> np.ndarray:
    return np.load(RESULTS_DIR / "reference_components.npy")


def truth_matrix(scenario: str) -> np.ndarray:
    """Matriz dirigida verdadera de G_{X|S}: T[i, j] = 1 si i -> j."""
    T = np.zeros((len(BASE_VARS), len(BASE_VARS)), dtype=int)
    if scenario in ("D_semi", "D_null"):
        for a, b in PLANTED_EDGES:
            T[BASE_VARS.index(a), BASE_VARS.index(b)] = 1
    return T


def standardize(X: np.ndarray) -> np.ndarray:
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - X.mean(0)) / sd


# ----------------------------------------------------------------------------
# Representación de grafos y métricas
# ----------------------------------------------------------------------------
def cl_graph_to_masks(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """causal-learn: g[i,j] = -1 y g[j,i] = 1  <=>  i -> j (cola en i, punta en j)."""
    skel = (g != 0) | (g.T != 0)
    np.fill_diagonal(skel, False)
    directed = (g == -1) & (g.T == 1)
    return skel, directed


def weighted_to_masks(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = W != 0
    np.fill_diagonal(A, False)
    return A | A.T, A & ~A.T


def evaluate(skel: np.ndarray, directed: np.ndarray, scenario: str) -> dict:
    T = truth_matrix(scenario)
    tskel = (T | T.T).astype(bool)
    iu = np.triu_indices(len(BASE_VARS), 1)
    e, t = skel[iu], tskel[iu]
    tp, fp, fn = int((e & t).sum()), int((e & ~t).sum()), int((~e & t).sum())
    prec = tp / (tp + fp) if tp + fp else (1.0 if t.sum() == 0 else 0.0)
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {"SHD": fp + fn, "TP": tp, "FP": fp, "FN": fn, "precision": prec, "recall": rec, "F1": f1,
           "n_edges": int(e.sum())}
    if scenario in ("D_semi", "D_null"):
        (a, b), (c, d) = VSTRUCT
        ia, ib, ic = BASE_VARS.index(a), BASE_VARS.index(b), BASE_VARS.index(c)
        out["vstruct"] = int(bool(directed[ia, ib] and directed[ic, ib]))
    else:
        out["vstruct"] = np.nan
    return out


# ----------------------------------------------------------------------------
# Métodos de descubrimiento causal
# ----------------------------------------------------------------------------
def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def run_pc(Z: np.ndarray, alpha: float = ALPHA_CI):
    from causallearn.search.ConstraintBased.PC import pc
    cg = _quiet(pc, Z, alpha, "fisherz", stable=True, show_progress=False)
    return cl_graph_to_masks(cg.G.graph)


def run_fci(Z: np.ndarray, alpha: float = ALPHA_CI):
    from causallearn.search.ConstraintBased.FCI import fci
    G, _ = _quiet(fci, Z, "fisherz", alpha, show_progress=False, verbose=False)
    return cl_graph_to_masks(G.graph)


def run_ges(Z: np.ndarray, alpha: float = ALPHA_CI):
    from causallearn.search.ScoreBased.GES import ges
    r = _quiet(ges, Z, "local_score_BIC")
    return cl_graph_to_masks(r["G"].graph)


def run_margcorr(Z: np.ndarray, alpha: float = ALPHA_CI):
    n, d = Z.shape
    r = np.corrcoef(Z, rowvar=False)
    z = 0.5 * np.log((1 + r) / (1 - np.clip(r, -0.999999, 0.999999))) * np.sqrt(n - 3)
    p = 2 * stats.norm.sf(np.abs(z))
    skel = p < alpha
    np.fill_diagonal(skel, False)
    return skel, np.zeros_like(skel)


def run_empty(Z: np.ndarray, alpha: float = ALPHA_CI):
    d = Z.shape[1]
    return np.zeros((d, d), bool), np.zeros((d, d), bool)


def notears_linear(X: np.ndarray, lambda1: float = 0.1, max_iter: int = 100, h_tol: float = 1e-8,
                   rho_max: float = 1e16, w_threshold: float = 0.3) -> np.ndarray:
    """NOTEARS lineal con pérdida cuadrática (Zheng et al., NeurIPS 2018). W[i,j] != 0  <=>  i -> j."""
    n, d = X.shape
    X = X - X.mean(0)

    def _adj(w):
        return (w[: d * d] - w[d * d:]).reshape(d, d)

    def _loss(W):
        R = X - X @ W
        return 0.5 / n * (R ** 2).sum(), -1.0 / n * X.T @ R

    def _h(W):
        E = slin.expm(W * W)
        return np.trace(E) - d, E.T * W * 2

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        return obj, np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W = _adj(w_est)
    W[np.abs(W) < w_threshold] = 0
    return W


def run_notears(Z: np.ndarray, alpha: float = ALPHA_CI):
    return weighted_to_masks(notears_linear(Z))


# --------------------------- AMCD (método propuesto) ---------------------------
FULL_K_MAX_V2 = 1   # AMCD v2 (§1.10, enmienda): mezclas diagonales k=1..6 + una sola gaussiana completa (k=1)
FULL_K_MAX_V1 = 6   # AMCD v1 original (§1.5): diagonal y completa con k=1..6; se conserva como ablación


def analyze_mixture(R: np.ndarray, k_max: int = 6, seed: int = SEED, full_k_max: int = FULL_K_MAX_V2) -> dict:
    """ANALYZE: selección por BIC entre mezclas diagonales (k=1..k_max) y gaussianas de covarianza completa
    con k=1..full_k_max. En v2, la única alternativa completa es una sola gaussiana: distingue 'una fuente con
    variables correlacionadas' de 'mezcla de fuentes' sin que mezclas completas absorban la confusión."""
    rows = []
    for cov, kmax in (("diag", k_max), ("full", full_k_max)):
        for k in range(1, kmax + 1):
            g = GaussianMixture(k, covariance_type=cov, n_init=3, random_state=seed, reg_covar=1e-4).fit(R)
            rows.append((cov, k, g.bic(R)))
    best = min(rows, key=lambda t: t[2])
    return {"cov": best[0], "k": best[1], "bic": rows}


def residualize_mixture(R: np.ndarray, k: int, cov: str, seed: int, crossfit: bool = True) -> np.ndarray:
    """PLAN/EXECUTE: resta la media del componente asignado (cross-fitted en 2 folds)."""
    if not crossfit:
        g = GaussianMixture(k, covariance_type=cov, n_init=3, random_state=seed, reg_covar=1e-4).fit(R)
        return R - g.means_[g.predict(R)]
    out = np.empty_like(R)
    for tr, te in KFold(2, shuffle=True, random_state=seed).split(R):
        g = GaussianMixture(k, covariance_type=cov, n_init=3, random_state=seed, reg_covar=1e-4).fit(R[tr])
        out[te] = R[te] - g.means_[g.predict(R[te])]
    return out


def amcd(X: np.ndarray, alpha: float = ALPHA_CI, k_max: int = 6, max_iter: int = 3,
         crossfit: bool = True, k_fixed: int | None = None, replan: bool = True,
         seed: int = SEED, return_trace: bool = False, full_k_max: int = FULL_K_MAX_V2):
    """Autonomic MAPE-K Causal Discovery.

    MONITOR  : estandariza el estado actual (datos o residuos) y lo entrega a ANALYZE.
    ANALYZE  : GMM+BIC detecta si existe heterogeneidad latente (k*>1) y su tamaño.
    PLAN     : si k*>1, planifica condicionar en la fuente latente (residualización cross-fitted);
               si k*=1, el lazo se detiene (no se interviene sobre datos homogéneos).
    EXECUTE  : aplica el plan y re-monitorea los residuos (re-planificación) hasta k*=1 o max_iter;
               finalmente ejecuta PC sobre el estado condicionado.
    KNOWLEDGE: traza con la curva BIC, k*, covarianza elegida y decisión de cada iteración.
    """
    R = standardize(X)
    trace = []
    for it in range(max_iter):
        if k_fixed is not None:
            k, cov, bic = k_fixed, "diag", None
        else:
            a = analyze_mixture(R, k_max, seed + it, full_k_max)
            k, cov, bic = a["k"], a["cov"], a["bic"]
        trace.append({"iter": it, "k": k, "cov": cov, "bic": bic, "condition": k > 1})
        if k == 1:
            break
        R = standardize(residualize_mixture(R, k, cov, seed + it, crossfit))
        if k_fixed is not None or not replan:
            break
    skel, directed = run_pc(R, alpha)
    return (skel, directed, trace) if return_trace else (skel, directed)


def run_amcd(Z: np.ndarray, alpha: float = ALPHA_CI, seed: int = SEED, **kw):
    return amcd(Z, alpha=alpha, seed=seed, **kw)


# ------------------ Baselines conscientes de heterogeneidad (R1.1) ------------------
def estimate_context(Z: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Contexto latente estimado con el MISMO Analyze de AMCD (GMM, k y covarianza por BIC), en muestra,
    como en el uso estándar de JCI/CD-NOD con un índice de dominio sustituto. Las etiquetas se re-codifican
    en orden ascendente de la media del primer componente principal de cada grupo (índice ordinal informativo
    para pruebas Fisher-z lineales)."""
    a = analyze_mixture(Z, seed=seed)
    if a["k"] == 1:
        return np.zeros(len(Z), dtype=int)
    g = GaussianMixture(a["k"], covariance_type=a["cov"], n_init=3, random_state=seed, reg_covar=1e-4).fit(Z)
    lab = g.predict(Z)
    u, s, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pc1 = Z @ vt[0]
    order = np.argsort([pc1[lab == c].mean() if (lab == c).any() else np.inf for c in range(a["k"])])
    remap = np.empty(a["k"], dtype=int)
    remap[order] = np.arange(a["k"])
    return remap[lab]


def run_jci_pc(Z: np.ndarray, alpha: float = ALPHA_CI, seed: int = SEED):
    """JCI-PC (Mooij, Magliacane & Claassen, JMLR 2020): variables de contexto (k-1 dummies del contexto
    estimado) añadidas como nodos exógenos con conocimiento de fondo (prohibido X -> C)."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.graph.GraphNode import GraphNode
    ctx = estimate_context(Z, seed)
    k = ctx.max() + 1
    d = Z.shape[1]
    if k == 1:
        return run_pc(Z, alpha)
    C = np.eye(k)[ctx][:, 1:]
    data = np.hstack([Z, C])
    nodes = [GraphNode(f"X{i + 1}") for i in range(data.shape[1])]
    bk = BackgroundKnowledge()
    for i in range(d):
        for j in range(d, data.shape[1]):
            bk.add_forbidden_by_node(nodes[i], nodes[j])
    cg = _quiet(pc, data, alpha, "fisherz", stable=True, show_progress=False, background_knowledge=bk)
    return cl_graph_to_masks(cg.G.graph[:d, :d])


def run_cdnod(Z: np.ndarray, alpha: float = ALPHA_CI, seed: int = SEED):
    """CD-NOD (Huang et al., JMLR 2020) con el contexto estimado como índice de dominio (Fisher-z)."""
    from causallearn.search.ConstraintBased.CDNOD import cdnod
    ctx = estimate_context(Z, seed)
    d = Z.shape[1]
    if ctx.max() == 0:
        return run_pc(Z, alpha)
    cg = _quiet(cdnod, Z, ctx.reshape(-1, 1).astype(float), alpha, "fisherz", stable=True, show_progress=False)
    return cl_graph_to_masks(cg.G.graph[:d, :d])


def run_oracle(Z: np.ndarray, labels: np.ndarray, alpha: float = ALPHA_CI):
    R = Z.copy()
    for g in np.unique(labels):
        R[labels == g] -= R[labels == g].mean(0)
    return run_pc(standardize(R), alpha)


METHOD_FUNCS = {"Empty": run_empty, "MargCorr": run_margcorr, "PC": run_pc, "GES": run_ges,
                "FCI": run_fci, "NOTEARS": run_notears, "AMCD": run_amcd}


def run_method(name: str, X: np.ndarray, alpha: float = ALPHA_CI, seed: int = SEED,
               labels: np.ndarray | None = None, **kw) -> tuple[np.ndarray, np.ndarray, float]:
    Z = standardize(X)
    t0 = time.perf_counter()
    if name == METHOD_ORACLE:
        skel, directed = run_oracle(Z, labels, alpha)
    elif name == "AMCD":
        skel, directed = run_amcd(Z, alpha=alpha, seed=seed, **kw)
    elif name == "AMCD-v1":
        skel, directed = run_amcd(Z, alpha=alpha, seed=seed, full_k_max=FULL_K_MAX_V1, **kw)
    elif name in ("JCI-PC", "CD-NOD"):
        skel, directed = (run_jci_pc if name == "JCI-PC" else run_cdnod)(Z, alpha=alpha, seed=seed)
    else:
        skel, directed = METHOD_FUNCS[name](Z, alpha)
    return skel, directed, time.perf_counter() - t0


def subsample_indices(n_total: int, n: int, r: int) -> np.ndarray:
    return np.sort(np.random.default_rng(SEED + r).choice(n_total, size=n, replace=False))
