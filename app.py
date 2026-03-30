"""
app.py — Точка входа Streamlit приложения
==========================================
Запуск:
    streamlit run app.py

Структура проекта:
    app.py              — точка входа
    core/
        config.py       — константы, пути, справочники
        features.py     — feature engineering
        model.py        — загрузка модели и предсказание
    ui/
        manual.py       — вкладка ручного ввода
        upload.py       — вкладка загрузки Excel с маппингом
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import streamlit as st
from pathlib import Path

# Добавляем папку проекта в sys.path чтобы импорты core.* и ui.* работали
_APP_DIR = Path(__file__).parent.resolve()
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from core.model import (
    load_bundle, load_train, load_ext_lookup, load_explainer,
)
from core.config import HIGH_T, LOW_T
from ui.manual import manual_input_tab
from ui.upload import batch_scoring_tab

# ── Конфиг страницы ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Скоринг субсидий АПК",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Golos Text', sans-serif; }

.main-title { font-size:28px; font-weight:700; color:#4ade80; margin-bottom:4px; }
.sub-title  { font-size:15px; color:#9ca3af; margin-bottom:24px; }

.card {
    background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.12);
    border-radius:14px; padding:20px; margin-bottom:12px;
}
.score-big { font-size:72px; font-weight:700; line-height:1; }
.badge {
    display:inline-block; padding:8px 18px; border-radius:8px;
    font-size:16px; font-weight:600;
}
.metric-box {
    text-align:center; padding:14px;
    background:rgba(255,255,255,0.05);
    border-radius:10px; border:1px solid rgba(255,255,255,0.1);
}
.metric-val { font-size:28px; font-weight:700; color:#4ade80; }
.metric-lbl { font-size:12px; color:#9ca3af; margin-top:2px; }
.factor-pos {
    background:rgba(22,163,74,0.15); border-left:3px solid #16a34a;
    padding:8px 12px; border-radius:0 6px 6px 0; margin:4px 0;
    font-size:14px; color:#4ade80;
}
.factor-neg {
    background:rgba(220,38,38,0.15); border-left:3px solid #dc2626;
    padding:8px 12px; border-radius:0 6px 6px 0; margin:4px 0;
    font-size:14px; color:#f87171;
}
.stButton > button[kind="primary"] {
    background:#16a34a !important; color:#fff !important; border:none !important;
    border-radius:10px; font-size:16px; font-weight:600; padding:12px 0;
}
.stButton > button[kind="primary"]:hover { background:#15803d !important; }
</style>
""", unsafe_allow_html=True)

# ── Загрузка ресурсов ────────────────────────────────────────────────────────
try:
    bundle    = load_bundle()
    train_df  = load_train()
    ext_lkp   = load_ext_lookup()
    explainer = load_explainer(bundle)
    features  = bundle["features"]
    HIGH_T    = bundle.get("high_threshold", HIGH_T)
    LOW_T     = bundle.get("low_threshold",  LOW_T)
except FileNotFoundError as e:
    st.error(f"**Файл не найден:** `{e}`")
    st.code("""
Структура папки:
    project/
    ├── app.py
    ├── core/  ui/
    ├── scoring_output/
    │   └── ensemble_v5.pkl
    ├── 2025  ().xlsx
    └── merged_subsidii_full.xlsx
""")
    st.stop()
except Exception as e:
    st.error(f"Ошибка загрузки: {e}")
    st.stop()

# ── Заголовок ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌾 Скоринг субсидий АПК</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Система оценки заявок — Decentrathon 5.0 · Кейс 2</div>',
    unsafe_allow_html=True,
)

# ── Метрики в шапке ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-box"><div class="metric-val">91.4%</div><div class="metric-lbl">Precision одобрения</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-box"><div class="metric-val">0.775</div><div class="metric-lbl">ROC-AUC</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-box"><div class="metric-val">33</div><div class="metric-lbl">Признаков модели</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-box"><div class="metric-val">36 651</div><div class="metric-lbl">Обучающих заявок</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Вкладки ──────────────────────────────────────────────────────────────────
tab_manual, tab_batch = st.tabs(["✏️  Ввод вручную", "📂  Загрузка Excel"])

with tab_manual:
    manual_input_tab(bundle, train_df, ext_lkp, explainer, features)

with tab_batch:
    batch_scoring_tab(bundle, train_df, ext_lkp, explainer, features)

# ── Футер ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Система предоставляет рекомендацию — финальное решение за комиссией. "
    f"Пороги: авто-одобрение p≥{HIGH_T:.2f} | авто-отклонение p≤{LOW_T:.2f} | "
    "Decentrathon 5.0"
)