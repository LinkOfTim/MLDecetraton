"""
manual.py — Вкладка ручного ввода заявки
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from datetime import datetime

from core.config import (
    HIGH_T, LOW_T,
    AKIMAT_MAP, DISTRICT_MAP, SUBSIDY_MAP, NORMATIVE_MAP, DIRECTION_MAP,
)
from core.model import predict, get_shap_values, score_badge


def manual_input_tab(bundle, train_df, ext_lkp, explainer, features):
    """Полный UI вкладки ручного ввода заявки."""

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("#### Данные заявки")

        oblast = st.selectbox(
            "Область *",
            options=sorted(AKIMAT_MAP.keys()),
            index=list(sorted(AKIMAT_MAP.keys())).index("область Абай"),
        )
        akimat = AKIMAT_MAP.get(oblast, "")
        st.caption(f"🏛 Акимат: {akimat[:60]}...")

        districts = DISTRICT_MAP.get(oblast, ["Район не указан"])
        district  = st.selectbox("Район *", options=districts)

        st.markdown("---")

        direction = st.selectbox(
            "Направление субсидирования *",
            options=sorted(SUBSIDY_MAP.keys()),
        )
        subsidy_names = SUBSIDY_MAP.get(direction, [])
        subsidy_name  = st.selectbox("Наименование субсидирования *", options=subsidy_names)

        norm_key  = f"{direction}||{subsidy_name}"
        norm_auto = NORMATIVE_MAP.get(norm_key, 15000)
        normative = st.number_input(
            "Норматив (тенге) *",
            min_value=1, value=norm_auto, step=100,
            help="Заполняется автоматически по типу субсидии",
        )

        st.markdown("---")

        c_sum, c_date = st.columns(2)
        with c_sum:
            amount = st.number_input(
                "Сумма заявки (тенге) *",
                min_value=0, value=int(norm_auto * 100),
                step=10000, format="%d",
            )
        with c_date:
            date_val = st.date_input("Дата подачи *", value=datetime.today())

        if normative > 0:
            heads = int(amount / normative)
            st.caption(f"🐄 Расчётное поголовье: **{heads:,}** голов")

        calc = st.button("🔍 Рассчитать скор", type="primary", use_container_width=True)

    # ── Результат ────────────────────────────────────────────────────────────
    with col_result:
        st.markdown("#### Результат")

        if not calc:
            st.markdown("""
            <div class="card" style="text-align:center;padding:48px;color:#9ca3af">
                <div style="font-size:52px">🌾</div>
                <div style="margin-top:14px;font-size:15px">
                    Заполните поля слева<br>и нажмите «Рассчитать скор»
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        date_str = date_val.strftime("%d.%m.%Y") + " 09:00:00"
        df_row = pd.DataFrame([{
            "date": date_str, "oblast": oblast, "akimat": akimat,
            "direction": direction, "subsidy_name": subsidy_name,
            "status": "Принята", "normative": normative,
            "sum": amount, "district": district,
        }])

        with st.spinner("Вычисление..."):
            probs, X_feat = predict(df_row, bundle, train_df, ext_lkp, features)
            p     = float(probs[0])
            score = p * 100
            label, color, bg = score_badge(p, HIGH_T, LOW_T)

        # Score карточка
        st.markdown(f"""
        <div class="card" style="text-align:center;border-color:{color}40">
            <div style="color:#6b7280;font-size:13px;margin-bottom:4px">Скор заявки</div>
            <div class="score-big" style="color:{color}">{score:.1f}</div>
            <div style="color:#9ca3af;font-size:13px">из 100</div>
            <div class="badge" style="background:{bg};color:{color};border:1.5px solid {color}40;margin-top:10px">
                {label}
            </div>
            <div style="color:#9ca3af;font-size:12px;margin-top:10px">
                Вероятность одобрения: <b>{p:.1%}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Зоны
        st.markdown(f"""
        <div class="card" style="font-size:13px">
            <b>Трёхзонная система:</b><br><br>
            <span style="color:{'#16a34a' if p>=HIGH_T else '#9ca3af'}">
                ✅ Авто-одобрение &nbsp; p ≥ {HIGH_T:.2f}</span><br>
            <span style="color:{'#d97706' if LOW_T<p<HIGH_T else '#9ca3af'}">
                ⚠️ Ручная проверка &nbsp; {LOW_T:.2f} &lt; p &lt; {HIGH_T:.2f}</span><br>
            <span style="color:{'#dc2626' if p<=LOW_T else '#9ca3af'}">
                ❌ Авто-отклонение &nbsp; p ≤ {LOW_T:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

        # Факторы SHAP
        st.markdown("**Объяснение решения:**")
        sv1, ev1 = get_shap_values(explainer, X_feat)
        row_sv   = sv1[0] if sv1.ndim > 1 else sv1
        feat_vals = list(zip(features, row_sv))

        top_pos = sorted(feat_vals, key=lambda x: x[1], reverse=True)[:4]
        top_neg = sorted(feat_vals, key=lambda x: x[1])[:4]

        c_pos, c_neg = st.columns(2)
        with c_pos:
            st.markdown("**За ✅**")
            for f, v in top_pos:
                if v > 0.01:
                    st.markdown(
                        f'<div class="factor-pos">+{v:.2f} &nbsp; <code>{f}</code></div>',
                        unsafe_allow_html=True,
                    )
        with c_neg:
            st.markdown("**Против ❌**")
            for f, v in top_neg:
                if v < -0.01:
                    st.markdown(
                        f'<div class="factor-neg">{v:.2f} &nbsp; <code>{f}</code></div>',
                        unsafe_allow_html=True,
                    )

        # Waterfall
        with st.expander("📊 Подробный SHAP waterfall"):
            try:
                expl = shap.Explanation(
                    values=row_sv, base_values=ev1,
                    data=X_feat.iloc[0].values, feature_names=list(features),
                )
                fig, ax = plt.subplots(figsize=(9, 5))
                plt.sca(ax)
                shap.plots.waterfall(expl, max_display=12, show=False)
                plt.title(f"Скор {score:.1f}/100", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Waterfall недоступен: {e}")
