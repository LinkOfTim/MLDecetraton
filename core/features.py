"""
features.py — Feature Engineering
Строит признаки для модели из сырых данных заявки.
"""

import pandas as pd
import numpy as np
from core.config import DIRECTION_MAP, POSITIVE_STATUSES


def extract_subsidy_type(name: str) -> str:
    """Классифицирует наименование субсидии в тип."""
    if pd.isna(name):
        return "unknown"
    n = name.lower()
    if "приобретен" in n and "маточн" in n:  return "purchase_maternal"
    if "приобретен" in n and "бык"    in n:  return "purchase_bull"
    if "племенной"  in n and ("работ" in n or "маточн" in n): return "breeding_work"
    if "товарн"     in n:                     return "commodity_herd"
    if "удешевлени" in n and "молок"  in n:   return "milk_subsidy"
    if "удешевлени" in n and ("мяс" in n or "птиц" in n or "говяд" in n): return "meat_subsidy"
    if "удешевлени" in n and "корм"   in n:   return "feed_subsidy"
    if "осемен"     in n:                     return "insemination"
    if "улей"       in n or "пчел"    in n:   return "bees"
    return "other"


def target_encode(df_in: pd.DataFrame, col: str, target_col: str,
                  ref_df: pd.DataFrame, smoothing: int = 20) -> np.ndarray:
    """
    Target encoding с байесовским сглаживанием.
    ref_df — только train часть фолда (нет утечки данных).
    """
    global_mean = ref_df[target_col].mean()
    stats = ref_df.groupby(col)[target_col].agg(["mean", "count"]).reset_index()
    stats.columns = [col, "te_mean", "te_count"]
    stats["te_smooth"] = (
        (stats["te_mean"] * stats["te_count"] + global_mean * smoothing)
        / (stats["te_count"] + smoothing)
    )
    result = df_in[[col]].merge(stats[[col, "te_smooth"]], on=col, how="left")
    return result["te_smooth"].fillna(global_mean).values


def build_features(df_input: pd.DataFrame, train_df, ext_lkp=None) -> pd.DataFrame:
    """
    Строит полный набор признаков для модели.

    Параметры:
        df_input  — входные данные заявки
        train_df  — обучающий датасет (None → используются встроенные словари)
        ext_lkp   — внешние агрегаты по областям (опционально)
    """
    from core.config import (
        GLOBAL_MEAN, TE_OBLAST, TE_AKIMAT, TE_DISTRICT,
        TE_SUBSIDY_TYPE, TE_DIRECTION, TE_OBLAST_DIR,
        FREQ_OBLAST, FREQ_SUBSIDY_TYPE, MED_OBLAST, MED_SUBSIDY_TYPE,
    )

    X   = df_input.copy()

    # ── Производные колонки ──────────────────────────────────────────────────
    if "direction_en" not in X.columns:
        X["direction_en"] = X["direction"].map(DIRECTION_MAP).fillna("unknown")
    if "subsidy_type" not in X.columns:
        X["subsidy_type"] = X["subsidy_name"].apply(extract_subsidy_type)
    X["oblast_dir"] = X["oblast"].astype(str) + "_" + X["direction_en"].astype(str)

    # ── Временные признаки ───────────────────────────────────────────────────
    dt = pd.to_datetime(X["date"], dayfirst=True, errors="coerce")
    X["month"]          = dt.dt.month.fillna(1).astype(int)
    X["day_of_week"]    = dt.dt.dayofweek.fillna(0).astype(int)
    X["day_of_year"]    = dt.dt.dayofyear.fillna(1).astype(int)
    X["hour"]           = dt.dt.hour.fillna(9).astype(int)
    X["quarter"]        = dt.dt.quarter.fillna(1).astype(int)
    X["is_peak_season"] = X["month"].isin([1, 2, 3]).astype(int)

    # ── Финансовые признаки ──────────────────────────────────────────────────
    safe = X["normative"].replace(0, np.nan)
    X["sum_log"]            = np.log1p(X["sum"].fillna(0).clip(lower=0))
    X["normative_log"]      = np.log1p(X["normative"].fillna(0))
    X["head_count_log"]     = np.log1p((X["sum"] / safe).fillna(0))
    X["sum_norm_ratio"]     = (X["sum"] / safe).fillna(0)
    X["is_exact_multiple"]  = (X["sum"] % X["normative"].replace(0, 1) == 0).astype(int)
    X["sum_norm_remainder"] = X["sum"] % X["normative"].replace(0, 1)
    X["sum_decile"]         = pd.cut(
        X["sum"].fillna(0).clip(lower=0), bins=10, labels=False
    ).fillna(0)

    # ── Ранг заявки ──────────────────────────────────────────────────────────
    X["date_day_str"]   = dt.dt.strftime("%Y-%m-%d").fillna("2025-01-01")
    X["daily_app_rank"] = X.groupby(["akimat", "date_day_str"])["sum"].rank(
        method="first"
    ).fillna(1)
    X["log_daily_rank"] = np.log1p(X["daily_app_rank"])

    # ── Target encoding ──────────────────────────────────────────────────────
    if train_df is not None and "target" in train_df.columns:
        # Считаем по реальным данным — точнее
        ref = train_df.copy()
        for d in [ref]:
            if "direction_en" not in d.columns:
                d["direction_en"] = d["direction"].map(DIRECTION_MAP).fillna("unknown")
            if "subsidy_type" not in d.columns:
                d["subsidy_type"] = d["subsidy_name"].apply(extract_subsidy_type)
            d["oblast_dir"] = d["oblast"].astype(str) + "_" + d["direction_en"].astype(str)

        for col, new_col in [
            ("oblast",       "te_oblast"),
            ("akimat",       "te_akimat"),
            ("district",     "te_district"),
            ("subsidy_type", "te_subsidy_type"),
            ("direction_en", "te_direction"),
            ("oblast_dir",   "te_oblast_dir"),
        ]:
            if col in ref.columns and col in X.columns:
                X[new_col] = target_encode(X, col, "target", ref)

        for col, nc in [("oblast", "freq_oblast"), ("subsidy_type", "freq_subsidy_type")]:
            freq = ref[col].value_counts().rename(nc).reset_index()
            freq.columns = [col, nc]
            X = X.merge(freq, on=col, how="left")
            X[nc] = X[nc].fillna(1)

        for grp, mc in [("oblast", "oblast_sum_med"), ("subsidy_type", "type_sum_med")]:
            med = ref.groupby(grp)["sum"].median().rename(mc).reset_index()
            med.columns = [grp, mc]
            X = X.merge(med, on=grp, how="left")

        X["sum_vs_oblast_med"] = X["sum"] / X["oblast_sum_med"].replace(0, np.nan)
        X["sum_vs_type_med"]   = X["sum"] / X["type_sum_med"].replace(0, np.nan)

    else:
        # Используем встроенные предвычисленные значения (без Excel файла)
        X["te_oblast"]       = X["oblast"].map(TE_OBLAST).fillna(GLOBAL_MEAN)
        X["te_akimat"]       = X["akimat"].map(TE_AKIMAT).fillna(GLOBAL_MEAN)
        X["te_district"]     = X["district"].map(TE_DISTRICT).fillna(GLOBAL_MEAN)
        X["te_subsidy_type"] = X["subsidy_type"].map(TE_SUBSIDY_TYPE).fillna(GLOBAL_MEAN)
        X["te_direction"]    = X["direction_en"].map(TE_DIRECTION).fillna(GLOBAL_MEAN)
        X["te_oblast_dir"]   = X["oblast_dir"].map(TE_OBLAST_DIR).fillna(GLOBAL_MEAN)
        X["freq_oblast"]     = X["oblast"].map(FREQ_OBLAST).fillna(1)
        X["freq_subsidy_type"] = X["subsidy_type"].map(FREQ_SUBSIDY_TYPE).fillna(1)
        X["oblast_sum_med"]  = X["oblast"].map(MED_OBLAST).fillna(X["sum"].median())
        X["type_sum_med"]    = X["subsidy_type"].map(MED_SUBSIDY_TYPE).fillna(X["sum"].median())
        X["sum_vs_oblast_med"] = X["sum"] / X["oblast_sum_med"].replace(0, np.nan)
        X["sum_vs_type_med"]   = X["sum"] / X["type_sum_med"].replace(0, np.nan)

    # ── Внешние агрегаты ─────────────────────────────────────────────────────
    if ext_lkp is not None:
        X = X.merge(ext_lkp, on="oblast", how="left")
        X["ext_oblast_avg_log"]   = np.log1p(X.get("ext_oblast_avg_sum", pd.Series(0)).fillna(0))
        X["ext_oblast_total_log"] = np.log1p(X.get("ext_oblast_total_sum", pd.Series(0)).fillna(0))
        X["sum_vs_ext_oblast"]    = (
            X["sum"] / X.get("ext_oblast_avg_sum", pd.Series(np.nan)).replace(0, np.nan)
        ).fillna(1.0)
        for c in ["ext_oblast_success_rate", "ext_oblast_n_apps",
                  "ext_oblast_avg_log", "ext_oblast_total_log", "sum_vs_ext_oblast"]:
            if c in X.columns:
                X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)

    # Заполнение пропусков
    enc_cols = [c for c in X.columns if c.startswith("te_") or c.startswith("freq_")
                or c.endswith("_med") or "vs_" in c]
    for col in enc_cols:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

    return X