"""
model.py — Загрузка модели и предсказание
"""

import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
from pathlib import Path

from core.config import (
    MODEL_PATH, TRAIN_PATH, EXT_PATH,
    HIGH_T, LOW_T, POSITIVE_STATUSES,
)
from core.features import build_features, extract_subsidy_type
from core.config import DIRECTION_MAP


@st.cache_resource(show_spinner="⚙️ Загрузка модели...")
def load_bundle():
    """Загружает ансамбль из pkl файла."""
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="📊 Загрузка данных...")
def load_train():
    """
    Загружает обучающий датасет для target encoding.
    Если файл недоступен — возвращает None (используются встроенные значения).
    """
    import os
    if not os.path.exists(TRAIN_PATH):
        return None   # features.py использует встроенные словари из config.py
    try:
        df = pd.read_excel(TRAIN_PATH, skiprows=4)
        df.columns = ["num", "date", "c3", "c4", "oblast", "akimat", "app_num",
                      "direction", "subsidy_name", "status", "normative", "sum", "district"]
        df = df.dropna(subset=["status"]).reset_index(drop=True)
        df["target"]       = df["status"].isin(POSITIVE_STATUSES).astype(int)
        df["direction_en"] = df["direction"].map(DIRECTION_MAP).fillna("unknown")
        df["subsidy_type"] = df["subsidy_name"].apply(extract_subsidy_type)
        df["oblast_dir"]   = df["oblast"].astype(str) + "_" + df["direction_en"].astype(str)
        return df
    except Exception:
        return None


@st.cache_resource(show_spinner="🌍 Загрузка внешних данных...")
def load_ext_lookup():
    """
    Загружает внешние агрегаты по областям.
    Если файл недоступен — возвращает None (признаки заполнятся глобальным средним).
    """
    import os
    if not os.path.exists(EXT_PATH):
        return None
    try:
        df_e = pd.read_excel(EXT_PATH)
        df_e.columns = ["app_num", "date", "applicant", "status", "sum_ext",
                        "date_reject", "reason_reject", "date_withdraw",
                        "reason_withdraw", "category", "region", "source_file"]
        EXT_POS = {"Оплачен", "Утверждено", "Сформирован платеж"}
        EXT_NEG = {"Отказано", "Отозван", "Отклонено рабочим органом",
                   "Отклонено финансовым институтом", "Аннулировано", "Возвращены субсидии"}
        df_e = df_e[df_e["status"].isin(EXT_POS | EXT_NEG)].copy()
        df_e["target_ext"] = df_e["status"].isin(EXT_POS).astype(int)
        df_e["oblast"]     = df_e["region"]
        df_e["sum_ext"]    = pd.to_numeric(df_e["sum_ext"], errors="coerce").fillna(0).clip(lower=0)
        lkp = df_e.groupby("oblast").agg(
            ext_oblast_success_rate=("target_ext", "mean"),
            ext_oblast_n_apps=("target_ext", "count"),
            ext_oblast_avg_sum=("sum_ext", "mean"),
            ext_oblast_total_sum=("sum_ext", "sum"),
        ).reset_index()
        return lkp
    except Exception:
        return None


@st.cache_resource(show_spinner="🔬 Инициализация SHAP...")
def load_explainer(_bundle):
    """Создаёт SHAP explainer для модели."""
    return shap.TreeExplainer(_bundle["lgbm"])


def predict(df_row: pd.DataFrame, bundle: dict, train_df: pd.DataFrame,
            ext_lkp, features: list) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Предсказывает вероятность одобрения для датафрейма заявок.

    Возвращает:
        (probs, X_feat) — вероятности и матрица признаков
    """
    df_feat = build_features(df_row, train_df, ext_lkp)

    # Гарантируем наличие всех признаков
    for f in features:
        if f not in df_feat.columns:
            df_feat[f] = 0.0

    X = df_feat[features].fillna(-1)

    lgbm_p = bundle["lgbm"].predict_proba(X)[:, 1]
    if bundle.get("use_cat") and bundle["catboost"] is not None:
        cat_p  = bundle["catboost"].predict_proba(X)[:, 1]
        meta_X = np.column_stack([lgbm_p, cat_p])
    else:
        meta_X = lgbm_p.reshape(-1, 1)

    probs = bundle["meta"].predict_proba(meta_X)[:, 1]
    return probs, X


def get_shap_values(explainer, X_feat: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Вычисляет SHAP значения для матрицы признаков."""
    sv = explainer.shap_values(X_feat)
    if isinstance(sv, list):
        sv1 = sv[1] if len(sv) > 1 else sv[0]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        sv1 = sv[:, :, 1]
    else:
        sv1 = sv

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev1 = float(ev[1] if len(ev) > 1 else ev[0])
    else:
        ev1 = float(ev)

    return sv1, ev1


def score_badge(p: float, high: float = HIGH_T,
                low: float = LOW_T) -> tuple[str, str, str]:
    """
    Возвращает (label, цвет, фон) для решения по порогам.

    Зоны:
        p >= high → Авто-одобрение  (зелёный)
        p <= low  → Авто-отклонение (красный)
        иначе     → Ручная проверка (жёлтый)
    """
    if p >= high:
        return "✅ Авто-одобрение",  "#16a34a", "#dcfce7"
    if p <= low:
        return "❌ Авто-отклонение", "#dc2626", "#fee2e2"
    return "⚠️ Ручная проверка", "#d97706", "#fef3c7"