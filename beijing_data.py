"""Carga, preprocesamiento y muestreo compartidos de Beijing Multi-Site (UCI 501) para 07 y 08 (§1.11, §1.12)."""
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

from utils import N_SUB, PROJECT_DIR, analyze_mixture

RAW_DIR = PROJECT_DIR / "data_external" / "beijing" / "raw" / "PRSA_Data_20130301-20170228"
VARS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]
POLL = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
K_NAMES = ["Heating", "Spring", "Summer", "Autumn"]


def load_daily() -> pd.DataFrame:
    """Medias diarias por estación (≥ 18 h válidas), sin faltantes, log de contaminantes, régimen K y día índice."""
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW_DIR / "*.csv")))], ignore_index=True)
    raw["date"] = pd.to_datetime(raw[["year", "month", "day"]])
    daily = raw.groupby(["station", "date"])[VARS].agg(lambda s: s.mean() if s.notna().sum() >= 18 else np.nan)
    daily = daily.dropna().reset_index()
    daily[POLL] = np.log(daily[POLL])
    md = daily.date.dt.month * 100 + daily.date.dt.day
    daily["K"] = np.select([(md >= 1115) | (md <= 315), md <= 531, md <= 831], [0, 1, 2], default=3)
    daily["dayidx"] = (daily.date - daily.date.min()).dt.days
    return daily


def sample_dates(daily, rng, stations, n=N_SUB):
    """Una estación al azar por fecha; fechas en una grilla cada 3 días (offset aleatorio)."""
    off = rng.integers(3)
    pool = daily[daily.station.isin(stations) & ((daily.dayidx - off) % 3 == 0)]
    dates = pool.date.unique()
    chosen = rng.choice(dates, size=n, replace=False)
    sub = pool[pool.date.isin(chosen)]
    sub = sub.groupby("date", group_keys=False).apply(lambda g: g.sample(1, random_state=int(rng.integers(1e9))))
    return sub.sort_values("date")


def estimated_partition_nmi(Z, K, seed, rng):
    """NMI (por fold cross-fitted) entre la partición de la iteración 1 de AMCD y K; y nulo permutado."""
    a = analyze_mixture(Z, seed=seed)
    if a["k"] == 1:
        return 0.0, 0.0, a["k"]
    nmi, null = [], []
    for tr, te in KFold(2, shuffle=True, random_state=seed).split(Z):
        g = GaussianMixture(a["k"], covariance_type=a["cov"], n_init=3, random_state=seed, reg_covar=1e-4).fit(Z[tr])
        lab = g.predict(Z[te])
        nmi.append(normalized_mutual_info_score(K[te], lab))
        null.append(normalized_mutual_info_score(rng.permutation(K[te]), lab))
    return float(np.mean(nmi)), float(np.mean(null)), a["k"]
