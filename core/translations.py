"""
translations.py — Человекочитаемые названия признаков для SHAP графиков
"""

FEATURE_NAMES_RU = {
    # Target encoding — региональные
    "te_district":          "Успешность района",
    "te_oblast":            "Успешность области",
    "te_akimat":            "Успешность акимата",
    "te_oblast_dir":        "Успешность (область × направление)",
    "te_subsidy_type":      "Успешность типа субсидии",
    "te_direction":         "Успешность направления",

    # Frequency encoding
    "freq_oblast":          "Активность области",
    "freq_subsidy_type":    "Частота типа субсидии",

    # Финансовые признаки
    "sum_log":              "Сумма заявки (лог)",
    "normative_log":        "Норматив (лог)",
    "head_count_log":       "Поголовье (лог)",
    "sum_norm_ratio":       "Сумма / Норматив",
    "sum_decile":           "Децильный ранг суммы",
    "is_exact_multiple":    "Кратность нормативу",
    "sum_norm_remainder":   "Остаток от норматива",

    # Относительные суммы
    "sum_vs_oblast_med":    "Сумма vs медиана области",
    "sum_vs_type_med":      "Сумма vs медиана типа",
    "sum_vs_ext_oblast":    "Сумма vs средняя выплата области",
    "sum_vs_ext_ds":        "Сумма vs средняя (область × тип)",

    # Временные признаки
    "month":                "Месяц подачи",
    "day_of_year":          "День года",
    "day_of_week":          "День недели",
    "hour":                 "Час подачи",
    "quarter":              "Квартал",
    "is_peak_season":       "Пиковый сезон (янв–март)",

    # Ранг заявки
    "log_daily_rank":       "Порядок подачи в день",

    # Внешние агрегаты
    "ext_oblast_success_rate":  "Рейтинг области (внешний)",
    "ext_oblast_n_apps":        "Активность области (внешняя)",
    "ext_oblast_avg_log":       "Средняя выплата области (лог)",
    "ext_oblast_total_log":     "Суммарные выплаты области (лог)",
    "ext_ds_success_rate":      "Рейтинг (область × тип, внешний)",
    "ext_ds_count":             "Кол-во выплат (область × тип)",
    "ext_ds_avg_log":           "Средняя выплата (область × тип, лог)",
}


def translate_features(feature_names: list) -> list:
    """Переводит список названий признаков на русский."""
    return [FEATURE_NAMES_RU.get(f, f) for f in feature_names]
