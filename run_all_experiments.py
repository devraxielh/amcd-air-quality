"""Ejecuta 01 → 05 desde cero dos veces y verifica los Exit Criteria:
archivos, 600 DPI, NO TITLES, ausencia de ajustes globales previos a la evaluación y reproducibilidad."""
import hashlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
STEPS = ["01_data_splitting_and_prep.py", "02_baseline_and_sota_models.py", "03_ablation_and_sensitivity.py",
         "04_statistical_tests.py", "05_explainability_and_error.py", "06_robustness_generators.py",
         "07_real_data_validation.py", "08_real_latent_regime.py"]
TABLES = ["table_01_cohort_summary", "table_02_main_results_cv", "table_03_ablation_study",
          "table_04_statistical_significance", "table_05_robustness_results",
          "table_06_robustness_significance",
          "table_07_real_data_validation", "table_08_real_data_significance",
          "table_09_latent_regime_validation", "table_10_latent_regime_significance"]
FIGURES = ["fig_01_data_distribution_or_manifold", "fig_02_main_performance_comparison",
           "fig_03_ablation_and_sensitivity", "fig_04_explainability_shap_or_latent",
           "fig_05_error_calibration_or_case_study", "fig_06_robustness_generators", "fig_07_real_data_validation",
           "fig_08_latent_regime_validation", "fig_S1_critical_difference"]
# Ajustes sobre el dataset completo permitidos y justificados: construcción de la verdad de la simulación
# (01) y el diagnóstico de error post-hoc frente a esa verdad (05). Ningún método evaluado los consume.
ALLOWED_GLOBAL_FITS = {"01_data_splitting_and_prep.py"}


def clean():
    for d in ("results", "tables", "figures"):
        shutil.rmtree(HERE / d, ignore_errors=True)


def run_pipeline() -> float:
    t0 = time.time()
    for s in STEPS:
        print(f"[run] {s}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit(f"FAILED: {s}")
    return time.time() - t0


def digest() -> dict:
    """Hash de cada CSV excluyendo columnas de tiempo de reloj (runtime), que varían por naturaleza."""
    import pandas as pd

    files = sorted((HERE / "tables").glob("*.csv")) + sorted((HERE / "results").glob("*.csv"))
    out = {}
    for f in files:
        df = pd.read_csv(f)
        df = df.drop(columns=[c for c in df.columns if "runtime" in c.lower()])
        out[f.name] = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    return out


checks = {}
clean()
t1 = run_pipeline()
d1 = digest()

checks["tables_exist"] = all((HERE / "tables" / f"{t}.{e}").exists() for t in TABLES for e in ("csv", "tex"))
checks["figures_exist"] = all((HERE / "figures" / f"{f}.{e}").exists() for f in FIGURES for e in ("pdf", "png"))
dpis = {f: Image.open(HERE / "figures" / f"{f}.png").info.get("dpi", (0, 0)) for f in FIGURES}
checks["png_600dpi"] = all(round(min(v)) >= 600 for v in dpis.values())

title_re = re.compile(r"(set_title|plt\.title|suptitle)\(\s*(?!\"\"|'')[^)]")
offenders = [f"{p.name}:{i}" for p in HERE.glob("*.py") if p.name != Path(__file__).name
             for i, line in enumerate(p.read_text().splitlines(), 1)
             if title_re.search(line) and not line.lstrip().startswith("#")]
checks["no_titles"] = not offenders

# Todos los resultados provienen de 50 réplicas
import pandas as pd  # noqa: E402

fs = pd.read_csv(HERE / "results" / "fold_scores.csv")
checks["all_results_replicated"] = bool(fs.groupby(["scenario", "model"]).replicate.nunique().min() == 50)

# Sin ajustes de preprocesamiento sobre el X completo en los scripts de evaluación (02-04)
global_fit = re.compile(r"(fit|fit_transform)\(\s*(Zfull|X_real|df)\b")
gf = [f"{p.name}:{i}" for p in HERE.glob("0[2-8]_*.py") for i, l in enumerate(p.read_text().splitlines(), 1)
      if global_fit.search(l)]
checks["no_global_fit_in_evaluation"] = not gf

clean()
t2 = run_pipeline()
d2 = digest()
checks["reproducible_second_run"] = d1 == d2
diff = [k for k in d1 if d1.get(k) != d2.get(k)]

print("\n================ EXIT CRITERIA ================")
for k, v in checks.items():
    print(f"[{'PASS' if v else 'FAIL'}] {k}")
print("DPI:", {k: tuple(round(x) for x in v) for k, v in dpis.items()})
if offenders:
    print("Title offenders:", offenders)
if gf:
    print("Global fits:", gf)
if diff:
    print("Non-reproducible files:", diff)
print(f"Runtime run1={t1:.1f}s run2={t2:.1f}s")
(HERE / "results" / "exit_criteria.txt").write_text(
    "\n".join(f"{k}: {v}" for k, v in checks.items()) + f"\nruntime_run1_s: {t1:.1f}\nruntime_run2_s: {t2:.1f}\n")
sys.exit(0 if all(checks.values()) else 1)
