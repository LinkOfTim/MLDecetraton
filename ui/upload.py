"""
upload.py — Загрузка Excel с маппингом столбцов и пакетным скорингом
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from core.config import HIGH_T, LOW_T
from core.model import predict, get_shap_values, score_badge

# Известные названия столбцов → стандартные имена
KNOWN_COL_ALIASES = {
    "Направление водства":         "direction",
    "Направление":                 "direction",
    "direction":                   "direction",
    "Наименование субсидирования": "subsidy_name",
    "subsidy_name":                "subsidy_name",
    "Область":                     "oblast",
    "oblast":                      "oblast",
    "Акимат":                      "akimat",
    "akimat":                      "akimat",
    "Район хозяйства":             "district",
    "Район":                       "district",
    "district":                    "district",
    "Норматив":                    "normative",
    "normative":                   "normative",
    "Причитающая сумма":           "sum",
    "Сумма":                       "sum",
    "sum":                         "sum",
    "Дата поступления":            "date",
    "Дата подачи заявки":          "date",
    "date":                        "date",
    "Статус заявки":               "status",
    "status":                      "status",
    "№ п/п":                       "num",
    "Номер заявки":                "app_num",
}

REQUIRED_COLS  = ["date", "oblast", "direction", "subsidy_name", "normative", "sum"]
OPTIONAL_COLS  = ["akimat", "district", "status"]

ORIGINAL_MARKERS = [
    "Реестр заявок", "subsidy.plem.kz", "ИСС", "Данные были получены"
]


def _auto_read(uploaded_file) -> pd.DataFrame:
    """Читает Excel с автодетектом служебных строк."""
    probe = pd.read_excel(uploaded_file, nrows=5, header=None)
    probe_txt = " ".join(str(v) for v in probe.values.flatten() if str(v) != "nan")
    uploaded_file.seek(0)

    if any(m in probe_txt for m in ORIGINAL_MARKERS):
        df = pd.read_excel(uploaded_file, skiprows=4)
        st.info("✅ Оригинальный формат датасета — служебные строки пропущены.")
    else:
        df = pd.read_excel(uploaded_file)

    return df.dropna(how="all").reset_index(drop=True)


def _auto_map(df: pd.DataFrame) -> dict[str, str]:
    """
    Автоматически предлагает маппинг столбцов.
    Возвращает {колонка_файла: стандартное_имя}.
    """
    mapping = {}
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in KNOWN_COL_ALIASES:
            mapping[col_str] = KNOWN_COL_ALIASES[col_str]
    return mapping


def column_mapper_ui(df: pd.DataFrame) -> dict[str, str]:
    """
    UI для сопоставления столбцов файла со стандартными полями модели.
    Показывает только если не все обязательные поля нашлись автоматически.
    """
    auto = _auto_map(df)
    file_cols = ["— не выбрано —"] + list(df.columns)
    all_targets = REQUIRED_COLS + OPTIONAL_COLS

    # Проверяем что автоматически нашли
    auto_found  = {v for v in auto.values() if v in REQUIRED_COLS}
    auto_miss   = [c for c in REQUIRED_COLS if c not in auto_found]

    if not auto_miss:
        return auto   # всё нашлось автоматически

    # Показываем UI маппинга
    st.warning(f"Не удалось автоматически определить {len(auto_miss)} столбца(-ов). "
               "Укажи вручную:")

    LABELS = {
        "date":         "📅 Дата подачи заявки",
        "oblast":       "🗺️ Область",
        "akimat":       "🏛️ Акимат",
        "direction":    "🐄 Направление субсидирования",
        "subsidy_name": "📋 Наименование субсидирования",
        "normative":    "💰 Норматив (тенге)",
        "sum":          "💵 Сумма заявки (тенге)",
        "district":     "📍 Район",
        "status":       "📊 Статус заявки",
    }

    mapping = dict(auto)

    # Инвертируем auto для обратного поиска
    rev_auto = {v: k for k, v in auto.items()}

    cols_grid = st.columns(3)
    for i, target in enumerate(all_targets):
        label    = LABELS.get(target, target)
        required = target in REQUIRED_COLS
        default  = rev_auto.get(target, "— не выбрано —")
        default_idx = file_cols.index(default) if default in file_cols else 0

        with cols_grid[i % 3]:
            chosen = st.selectbox(
                f"{label} {'*' if required else ''}",
                options=file_cols,
                index=default_idx,
                key=f"col_map_{target}",
            )
            if chosen != "— не выбрано —":
                mapping[chosen] = target

    return mapping


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Применяет маппинг и добавляет дефолты для отсутствующих колонок."""
    df = df.rename(columns=mapping)

    # Дефолты
    defaults = {
        "district": "Район не указан",
        "normative": 15000,
        "status":    "Принята",
        "akimat":    "",
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    if df["akimat"].eq("").all() and "oblast" in df.columns:
        from core.config import AKIMAT_MAP
        df["akimat"] = df["oblast"].map(AKIMAT_MAP).fillna("")

    return df


def batch_scoring_tab(bundle, train_df, ext_lkp, explainer, features):
    """Полный UI вкладки пакетного скоринга."""
    st.markdown("#### Пакетный скоринг из Excel файла")

    # ── Шаблон ───────────────────────────────────────────────────────────────
    from core.config import AKIMAT_MAP
    tpl = pd.DataFrame([{
        "date":         "21.01.2025 11:15:40",
        "oblast":       "область Абай",
        "akimat":       AKIMAT_MAP["область Абай"],
        "direction":    "Субсидирование в скотоводстве",
        "subsidy_name": "Заявка на получение субсидий за приобретение племенных быков-производителей мясных и мясо-молочных пород",
        "normative":    150000,
        "sum":          4500000,
        "district":     "Жарминский район",
    }])
    buf = io.BytesIO()
    tpl.to_excel(buf, index=False)
    st.download_button(
        "⬇️ Скачать шаблон",
        data=buf.getvalue(),
        file_name="template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.caption("Поддерживается: шаблон выше, оригинальный датасет субсидий, "
               "любой Excel — столбцы можно сопоставить вручную.")

    # ── Загрузка ─────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Загрузи Excel файл", type=["xlsx", "xls"])
    if not uploaded:
        return

    raw_df = _auto_read(uploaded)
    st.write(f"Загружено **{len(raw_df):,}** строк, **{len(raw_df.columns)}** столбцов")

    with st.expander("👁 Предпросмотр первых 3 строк"):
        st.dataframe(raw_df.head(3), use_container_width=True)

    # ── Маппинг столбцов ─────────────────────────────────────────────────────
    st.markdown("##### Сопоставление столбцов")
    mapping = column_mapper_ui(raw_df)

    # Проверяем обязательные
    mapped_targets = set(mapping.values())
    missing = [c for c in REQUIRED_COLS if c not in mapped_targets]
    if missing:
        st.error(f"Не хватает обязательных столбцов: {missing}")
        return

    df_mapped = apply_mapping(raw_df.copy(), mapping)

    # ── Скоринг ──────────────────────────────────────────────────────────────
    if not st.button("🚀 Запустить скоринг", type="primary", use_container_width=True):
        return

    bar = st.progress(0, "Вычисление скоров...")
    probs, _ = predict(df_mapped, bundle, train_df, ext_lkp, features)
    bar.progress(70, "Формирование таблицы...")

    df_res = df_mapped.copy()
    df_res["score"]       = (probs * 100).round(1)
    df_res["probability"] = probs.round(4)
    df_res["decision"]    = [score_badge(p, HIGH_T, LOW_T)[0] for p in probs]
    df_res["rank"]        = df_res["score"].rank(ascending=False, method="first").astype(int)
    df_res = df_res.sort_values("score", ascending=False)
    bar.progress(100, "Готово!")

    # ── Сводка по зонам ───────────────────────────────────────────────────────
    n = len(df_res)
    z = df_res["decision"].value_counts()
    auto_ok  = z.get("✅ Авто-одобрение",  0)
    manual   = z.get("⚠️ Ручная проверка", 0)
    auto_rej = z.get("❌ Авто-отклонение", 0)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#16a34a">{auto_ok:,}</div><div class="metric-lbl">✅ Авто-одобрение ({auto_ok/n:.1%})</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#d97706">{manual:,}</div><div class="metric-lbl">⚠️ Ручная проверка ({manual/n:.1%})</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#dc2626">{auto_rej:,}</div><div class="metric-lbl">❌ Авто-отклонение ({auto_rej/n:.1%})</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Таблица ───────────────────────────────────────────────────────────────
    def _color(row):
        d = str(row.get("decision", ""))
        if "Авто-одобрение"  in d: return ["background-color:#f0fdf4"] * len(row)
        if "Авто-отклонение" in d: return ["background-color:#fef2f2"] * len(row)
        if "Ручная"          in d: return ["background-color:#fffbeb"] * len(row)
        return [""] * len(row)

    show = ["rank"] + [c for c in ["oblast", "direction", "subsidy_name", "sum",
                                    "score", "decision"] if c in df_res.columns]
    st.dataframe(
        df_res[show].style.apply(_color, axis=1),
        use_container_width=True, height=400,
    )

    # ── Скачать ───────────────────────────────────────────────────────────────
    out = io.BytesIO()
    df_res.to_excel(out, index=False)
    st.download_button(
        "⬇️ Скачать результаты",
        data=out.getvalue(),
        file_name="scoring_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    # ── SHAP топ-5 ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 SHAP — объяснения топ-5 заявок")

    top5_idx = df_res.head(5).index.tolist()
    df_top5  = df_mapped.loc[top5_idx].reset_index(drop=True)
    _, X_top5 = predict(df_top5, bundle, train_df, ext_lkp, features)
    sv1_all, ev1 = get_shap_values(explainer, X_top5)

    for i in range(min(5, len(df_top5))):
        row_sv  = sv1_all[i] if sv1_all.ndim > 1 else sv1_all
        score_i = float(probs[top5_idx.index(df_res.head(5).index[i])] * 100)
        oblast_i = df_top5.loc[i, "oblast"] if "oblast" in df_top5.columns else ""

        with st.expander(f"🏆 #{i+1} — {oblast_i} | score {score_i:.1f}"):
            try:
                expl = shap.Explanation(
                    values=row_sv, base_values=ev1,
                    data=X_top5.iloc[i].values, feature_names=list(features),
                )
                fig, ax = plt.subplots(figsize=(9, 4))
                plt.sca(ax)
                shap.plots.waterfall(expl, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception:
                for f, v in sorted(zip(features, row_sv),
                                   key=lambda x: abs(x[1]), reverse=True)[:8]:
                    col = "#16a34a" if v > 0 else "#dc2626"
                    st.markdown(
                        f'<span style="color:{col}">{"▲" if v>0 else "▼"} {v:+.3f}</span> `{f}`',
                        unsafe_allow_html=True,
                    )
