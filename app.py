# app.py — Milk Digitalization v2.1 (Product page: dynamic steps + safety/quality norms, no big images)

import json
import io
import zipfile
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import sklearn (optional but recommended)
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    SKLEARN = True
except Exception:
    SKLEARN = False

# ---------------------------
# --- Настройки путей ---
# ---------------------------
DATA_DIR = Path(__file__).parent
# fallback path used previously (можешь поменять под себя)
fallback = Path(r"C:\Users\aidar\OneDrive\Desktop\МАДИНА\Milk_Digitalization")
if any(fallback.glob("*.csv")) and not any(DATA_DIR.glob("*.csv")):
    DATA_DIR = fallback

PRODUCTS_CSV = DATA_DIR / "Products.csv"
SAMPLES_CSV = DATA_DIR / "Samples.csv"
MEASUREMENTS_CSV = DATA_DIR / "Measurements.csv"
VITAMINS_CSV = DATA_DIR / "Vitamins_AminoAcids.csv"
STORAGE_CSV = DATA_DIR / "Storage_Conditions.csv"
NORMS_JSON = DATA_DIR / "process_norms.json"

# ---------------------------
# --- Утилиты ---
# ---------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, encoding="latin1")
    if not df.empty:
        df.columns = [str(c).strip() for c in df.columns]
    return df

def append_row_csv(path: Path, row: dict, cols_order=None):
    df_new = pd.DataFrame([row])
    write_header = not path.exists() or path.stat().st_size == 0
    if cols_order:
        for c in cols_order:
            if c not in df_new.columns:
                df_new[c] = ""
        df_new = df_new[cols_order]
    df_new.to_csv(path, mode='a', index=False, header=write_header, encoding='utf-8-sig')

def parse_numeric(val):
    """Аккуратно парсим числа: поддержка запятых, ±, x10^, etc."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    if s == "" or "не обнаруж" in s.lower():
        return np.nan
    s = s.replace(' ', '').replace(',', '.')
    s = s.replace('×10^', 'e').replace('x10^', 'e')
    s = s.replace('×10', 'e').replace('x10', 'e').replace('×', '')
    if '±' in s:
        s = s.split('±')[0]
    cleaned = ''
    for ch in s:
        if ch.isdigit() or ch in '.-+eE':
            cleaned += ch
        else:
            break
    try:
        return float(cleaned)
    except Exception:
        return np.nan

def download_zip(paths, filename="Milk_Digitalization_all_csv.zip"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in paths:
            if Path(p).exists():
                z.write(p, arcname=Path(p).name)
    buf.seek(0)
    st.download_button("Скачать ZIP", data=buf, file_name=filename, mime="application/zip")

def embed_pdf(path: Path):
    if not path.exists():
        st.warning("PDF файл не найден.")
        return
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode('utf-8')
    html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600"></iframe>'
    st.components.v1.html(html, height=600, scrolling=True)

# ---------------------------
# --- Автогенерация демо CSV (если нет файлов) ---
# ---------------------------
def _ensure_demo_csvs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PRODUCTS_CSV.exists():
        pd.DataFrame([
            {"product_id":1,"name":"Молоко (коровье)","type":"молоко","source":"коровье","description":"Свежее молоко"},
            {"product_id":2,"name":"Молоко (козье)","type":"молоко","source":"козье","description":"Свежее молоко"},
            {"product_id":3,"name":"Сары ірімшік (коровье)","type":"сыр","source":"коровье","description":"Твёрдый сыр"},
            {"product_id":4,"name":"Сары ірімшік (козье)","type":"сыр","source":"козье","description":"Твёрдый сыр"},
            {"product_id":5,"name":"Айран","type":"кисломолочный","source":"коровье","description":"Кисломолочный продукт"}
        ]).to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")

    if not SAMPLES_CSV.exists():
        pd.DataFrame([
            {"sample_id":1,"product_id":5,"reg_number":"A-001","date_received":datetime.now().strftime("%Y-%m-%d"),
             "storage_days":0,"conditions":"21°C, 64%","notes":"демо"},
        ]).to_csv(SAMPLES_CSV, index=False, encoding="utf-8-sig")

    if not MEASUREMENTS_CSV.exists():
        pd.DataFrame([
            {"id":1,"sample_id":1,"parameter":"Температура","unit":"°C","actual_value":"42","method":"демо"},
            {"id":2,"sample_id":1,"parameter":"pH","unit":"","actual_value":"4.3","method":"демо"},
        ]).to_csv(MEASUREMENTS_CSV, index=False, encoding="utf-8-sig")

    if not VITAMINS_CSV.exists():
        pd.DataFrame([
            {"name":"VitC","unit":"мг/100г","value":"0.90"}
        ]).to_csv(VITAMINS_CSV, index=False, encoding="utf-8-sig")

    if not STORAGE_CSV.exists():
        pd.DataFrame([
            {"sample_id":1,"temperature_C":4,"humidity_pct":70,"duration_days":3}
        ]).to_csv(STORAGE_CSV, index=False, encoding="utf-8-sig")

_ensure_demo_csvs()

# ---------------------------
# --- Кеш загрузки данных ---
# ---------------------------
@st.cache_data
def load_csvs():
    products = safe_read_csv(PRODUCTS_CSV)
    samples = safe_read_csv(SAMPLES_CSV)
    measurements = safe_read_csv(MEASUREMENTS_CSV)
    vitamins = safe_read_csv(VITAMINS_CSV)
    storage = safe_read_csv(STORAGE_CSV)
    return products, samples, measurements, vitamins, storage

products, samples, measurements, vitamins, storage = load_csvs()

# helper to normalize column names
def ensure_col(df, candidates, new_name):
    if df.empty:
        return df, None
    for col in df.columns:
        for cand in candidates:
            if str(col).strip().lower() == str(cand).strip().lower():
                return df.rename(columns={col: new_name}), new_name
    return df, None

# normalize product columns
products, _ = ensure_col(products, ["product_id","id"], "product_id")
products, _ = ensure_col(products, ["name","product_name","title"], "name")
products, _ = ensure_col(products, ["type","category"], "type")
products, _ = ensure_col(products, ["source"], "source")
products, _ = ensure_col(products, ["description"], "description")

# normalize samples columns
samples, _ = ensure_col(samples, ["sample_id","id"], "sample_id")
samples, _ = ensure_col(samples, ["product_id","product"], "product_id")
samples, _ = ensure_col(samples, ["reg_number"], "reg_number")
samples, _ = ensure_col(samples, ["date_received","date"], "date_received")
samples, _ = ensure_col(samples, ["storage_days","duration_days"], "storage_days")
samples, _ = ensure_col(samples, ["conditions"], "conditions")
samples, _ = ensure_col(samples, ["notes"], "notes")

# normalize measurement columns
measurements, _ = ensure_col(measurements, ["id"], "id")
measurements, _ = ensure_col(measurements, ["sample_id","sample"], "sample_id")
measurements, _ = ensure_col(measurements, ["parameter","param","indicator"], "parameter")
measurements, _ = ensure_col(measurements, ["actual_value","value","measurement"], "actual_value")
measurements, _ = ensure_col(measurements, ["unit"], "unit")
measurements, _ = ensure_col(measurements, ["method"], "method")

# storage
storage, _ = ensure_col(storage, ["sample_id"], "sample_id")
storage, _ = ensure_col(storage, ["temperature_C","temperature_c","temp"], "temperature_C")
storage, _ = ensure_col(storage, ["humidity_pct","humidity"], "humidity_pct")
storage, _ = ensure_col(storage, ["duration_days"], "duration_days")

# to int-like
def to_intlike(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
    return df

products = to_intlike(products, "product_id")
samples = to_intlike(samples, "sample_id")
samples = to_intlike(samples, "product_id")
measurements = to_intlike(measurements, "sample_id")
storage = to_intlike(storage, "sample_id")

# numeric measurements
if 'actual_value' in measurements.columns:
    measurements['actual_numeric'] = measurements['actual_value'].apply(parse_numeric)
else:
    measurements['actual_numeric'] = np.nan

# parse dates
if 'date_received' in samples.columns:
    samples['date_received'] = pd.to_datetime(samples['date_received'], errors='coerce')

# ---------------------------
# --- Нормы (process_norms.json) ---
# ---------------------------
default_norms = {
    "Пастеризация": {"min":72.0, "max":75.0, "unit":"°C", "note":"Типовая пастеризация (72–75°C) — см. протокол."},
    "Охлаждение": {"min":2.0, "max":6.0, "unit":"°C", "note":"Хранение/охлаждение."},
    "Ферментация": {"min":18.0, "max":42.0, "unit":"°C", "note":"Температуры ферментации — зависят от рецептуры."}
}
if NORMS_JSON.exists():
    try:
        norms = json.loads(NORMS_JSON.read_text(encoding='utf-8'))
    except Exception:
        norms = default_norms
else:
    norms = default_norms

# ---------------------------
# --- UI стили ---
# ---------------------------
st.set_page_config(page_title="Milk Digitalization", layout="wide")
st.markdown("""
<style>
.card{background:#fff;padding:12px;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.06);margin-bottom:12px}
.prod-title{font-weight:700;color:#0b4c86}
.step-card{background:#fff;padding:16px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.1);margin:8px 0;border-left:5px solid #0b4c86;transition:all 0.3s ease}
.step-card:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,0.15)}
.step-title{font-weight:600;color:#0b4c86;margin-bottom:8px}
.step-desc{color:#666;font-size:14px}
.arrow{text-align:center;font-size:20px;margin:4px 0;color:#0b4c86}
.step-small{font-size:13px;color:#333}
.small-muted{color:#666;font-size:13px}
.footer{color:#888;font-size:12px;margin-top:18px}
.product-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:20px;border-radius:15px;margin:10px 0;box-shadow:0 8px 25px rgba(0,0,0,0.15)}
.product-card:hover{transform:translateY(-3px);box-shadow:0 12px 35px rgba(0,0,0,0.2)}
.bad{background:#fff0f0}
.ok{background:#e6fff2}
</style>
""", unsafe_allow_html=True)

STEP_COLORS = {
    "pasteurization":"#d9534f",
    "cooling":"#0275d8",
    "fermentation":"#5cb85c",
    "accept":"#5bc0de",
    "normalization":"#f0ad4e",
    "homogenization":"#6f42c1",
    "inoculation":"#20c997",
    "coagulation":"#fd7e14",
    "pressing":"#6c757d",
    "filtration":"#007bff",
    "storage":"#17a2b8",
    "final":"#343a40"
}

PRODUCT_COLORS = {
    1: "linear-gradient(135deg,#667eea 0%,#764ba2 100%)",
    2: "linear-gradient(135deg,#f093fb 0%,#f5576c 100%)",
    3: "linear-gradient(135deg,#4facfe 0%,#00f2fe 100%)",
    4: "linear-gradient(135deg,#43e97b 0%,#38f9d7 100%)",
    5: "linear-gradient(135deg,#fa709a 0%,#fee140 100%)"
}

def color_for_step(step_id):
    sid = str(step_id).lower()
    for k,v in STEP_COLORS.items():
        if k in sid:
            return v
    return "#0b4c86"

def color_for_product(product_id):
    return PRODUCT_COLORS.get(product_id, "linear-gradient(135deg,#667eea 0%,#764ba2 100%)")

# ---------------------------
# --- State & Navigation ---
# ---------------------------
if 'page' not in st.session_state:
    st.session_state['page'] = 'Главная'
if 'selected_product' not in st.session_state:
    st.session_state['selected_product'] = None
if 'selected_step' not in st.session_state:
    st.session_state['selected_step'] = None
if 'selected_step_label' not in st.session_state:
    st.session_state['selected_step_label'] = None

st.sidebar.title("Навигация")
nav_choice = st.sidebar.radio(
    "",
    ["Главная", "Продукт", "Модели и аналитика"],
    index=["Главная","Продукт","Модели и аналитика"].index(st.session_state['page'])
    if st.session_state['page'] in ["Главная","Продукт","Модели и аналитика"] else 0
)

if nav_choice != st.session_state['page']:
    st.session_state['page'] = nav_choice
    st.session_state['selected_step'] = None
    st.session_state['selected_step_label'] = None
    st.rerun()

# Загрузка CSV
st.sidebar.markdown("---")
st.sidebar.write("Загрузить CSV (опционально)")
u = st.sidebar.file_uploader(
    "Выбери CSV (Products/Samples/Measurements/Vitamins/Storage). Можно по одному.",
    type=["csv"]
)
if u is not None:
    fname = u.name.lower()
    content = u.read()
    if "product" in fname:
        dest = PRODUCTS_CSV
    elif "sample" in fname:
        dest = SAMPLES_CSV
    elif "measure" in fname or "measurement" in fname:
        dest = MEASUREMENTS_CSV
    elif "vitamin" in fname or "amino" in fname:
        dest = VITAMINS_CSV
    elif "storage" in fname:
        dest = STORAGE_CSV
    else:
        dest = None

    if dest:
        try:
            Path(dest).write_bytes(content)
            st.sidebar.success(f"Сохранён {dest.name}")
            st.cache_data.clear()
            products, samples, measurements, vitamins, storage = load_csvs()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Ошибка записи: {e}")
    else:
        st.sidebar.info("Не удалось определить тип файла по имени. Переименуй файл и загрузи снова.")

st.sidebar.markdown("---")
st.sidebar.caption(f"📂 DATA_DIR: {DATA_DIR}")
missing = [p.name for p in [PRODUCTS_CSV,SAMPLES_CSV,MEASUREMENTS_CSV,VITAMINS_CSV,STORAGE_CSV] if not p.exists()]
if missing:
    st.sidebar.warning("Не найдены файлы: " + ", ".join(missing))
else:
    st.sidebar.success("Все CSV найдены ✅")

st.sidebar.markdown("---")
st.sidebar.markdown("Версия: 2.1 — dynamic Product page")

def goto_product(pid: int):
    st.session_state['selected_product'] = int(pid)
    st.session_state['page'] = 'Продукт'
    st.session_state['selected_step'] = None
    st.session_state['selected_step_label'] = None
    st.rerun()

# ---------------------------
# --- MAIN: Главная ---
# ---------------------------
if st.session_state['page'] == 'Главная':
    st.title("🥛 Milk Digitalization — демо платформа")
    st.markdown("Платформа для мониторинга партий, визуализации показателей и прототипирования моделей для молокопереработки.")
    st.markdown("---")

    # fixed five products; prefer CSV values if present
    fixed_products = [
        {"product_id":1,"name":"Молоко (коровье)","type":"молоко","source":"коровье","description":"Свежее молоко"},
        {"product_id":2,"name":"Молоко (козье)","type":"молоко","source":"козье","description":"Свежее молоко"},
        {"product_id":3,"name":"Сары ірімшік (коровье)","type":"сыр","source":"коровье","description":"Твёрдый сыр"},
        {"product_id":4,"name":"Сары ірімшік (козье)","type":"сыр","source":"козье","description":"Твёрдый сыр"},
        {"product_id":5,"name":"Айран","type":"кисломолочный","source":"коровье","description":"Кисломолочный продукт"}
    ]

    display_products = []
    for fp in fixed_products:
        chosen = None
        if not products.empty and 'product_id' in products.columns:
            try:
                match = products[products['product_id'] == fp['product_id']]
                if not match.empty:
                    chosen = match.iloc[0].to_dict()
            except Exception:
                chosen = None
        display_products.append(chosen if chosen is not None else fp)

    st.subheader("Наши продукты")
    cols = st.columns(2)
    for i, p in enumerate(display_products):
        with cols[i % 2]:
            product_color = color_for_product(p['product_id'])
            st.markdown(f"""
            <div class="product-card" style="background:{product_color}">
                <div style="font-size:18px;font-weight:bold;margin-bottom:8px">{p['name']}</div>
                <div style="font-size:14px;opacity:0.9">Тип: {p.get('type','-')}</div>
                <div style="font-size:14px;opacity:0.9">Источник: {p.get('source','-')}</div>
                <div style="font-size:13px;margin-top:10px;opacity:0.8">{p.get('description','')}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("📊 Открыть детали", key=f"btn_product_{p['product_id']}",
                         use_container_width=True, help="Перейти к журналу партий и измерениям"):
                goto_product(p['product_id'])

    st.markdown("---")
    st.subheader("Быстрые действия")
    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("📋 Журнал партий", use_container_width=True):
        st.session_state['page'] = 'Продукт'; st.rerun()
    if c2.button("📈 Аналитика", use_container_width=True):
        st.session_state['page'] = 'Модели и аналитика'; st.rerun()
    if c3.button("💾 Скачать CSV ZIP", use_container_width=True):
        download_zip([PRODUCTS_CSV, SAMPLES_CSV, MEASUREMENTS_CSV, VITAMINS_CSV, STORAGE_CSV])

    st.markdown("---")
    with st.expander("📂 Файлы в рабочей папке"):
        files_list = [p.name for p in DATA_DIR.glob("*")
                      if p.suffix.lower() in ['.csv', '.json', '.pdf', '.png', '.jpg', '.jpeg']]
        st.write(files_list)
























# ---------------------------
# --- PRODUCT PAGE (динамическая, без больших картинок) ---
# ---------------------------
elif st.session_state['page'] == 'Продукт':

    # --- Генератор этапа ---
    def _stage(id, label, icon, desc="", norm=None):
        return (id, label, icon, desc, norm or {})

    # --- Правильная логика этапов по продукту (с учётом source) ---
    def _product_steps(prod: dict):
        """
        Таблица-правило для гомогенизации:
        | Продукт                      | Гомогенизация |
        | ---------------------------- | ------------- |
        | Молоко (коровье)            | есть          |
        | Молоко (козье)              | нет           |
        | Сары ірімшік (коровье)      | есть          |
        | Сары ірімшік (козье)        | нет           |
        | Айран                       | есть          |
        """
        name = str(prod.get('name', '')).strip()
        source = str(prod.get('source', '')).strip()

        nlow = name.lower()
        slow = source.lower()

        # тип продукта
        is_ayran  = "айран" in nlow
        is_cheese = ("ірімшік" in nlow) or ("сыр" in nlow)
        is_milk   = ("молоко" in nlow)

        # признак козьего сырья (учитываем и name, и source)
        goat = (
            ("козье" in nlow) or ("козий" in nlow) or ("goat" in nlow) or ("ешкі" in nlow) or
            ("козье" in slow) or ("козий" in slow) or ("goat" in slow) or ("ешкі" in slow)
        )

        common = [
            _stage("accept", "Приёмка сырья", "📥",
                   "Осмотр тары, органолептика, экспресс-анализ состава/обсеменённости."),
            _stage("clarify", "Очистка и сортировка (4–6 °C)", "🧽",
                   "Фильтрация/сепараторы. Оценка чистоты, кислотности (°Т), определение сорта.",
                   {"min": 4, "max": 6, "unit": "°C", "note": "Охлаждение до 4–6 °C замедляет рост бактерий"}),
            _stage("normalization", "Нормализация состава", "⚖️",
                   "Приведение к нормам по жирности/белку/витаминам/минералам."),
        ]

        # Нужна ли гомогенизация по правилу
        need_homogenization = (is_ayran or (is_milk and not goat) or (is_cheese and not goat))
        if need_homogenization:
            common.append(_stage("homogenization", "Гомогенизация", "🌀",
                                 "Дробление жировых шариков → однородность, отсутствие отстоя."))

        # Общая пастеризация
        common.append(_stage(
            "pasteurization", "Пастеризация (65–69 °C)", "🔥",
            "Термообработка для снижения микрофлоры.",
            {"min": 65, "max": 69, "unit": "°C", "note": "Пастеризация согласно рецептуре/ГОСТ"}
        ))

        # Хвост процесса по типам продукта
        if is_ayran:
            tail = [
                _stage("cool_to_inoc", "Охлаждение до заквашивания (35–45 °C)", "🌡️", "Перед внесением закваски.",
                       {"min": 35, "max": 45, "unit": "°C"}),
                _stage("inoculation", "Внесение закваски", "🧫",
                       "Культуры: стрептококк, болгарская палочка, дрожжи."),
                _stage("fermentation", "Сквашивание (20–25 °C)", "⏱️", "Выдержка при заданной температуре.",
                       {"min": 20, "max": 25, "unit": "°C"}),
                _stage("salt", "Добавление соли (1.5–2%)", "🧂", "Перемешать до однородности."),
                _stage("mix_water", "Смешивание с водой / газирование", "💧", "Смешивание с кипячёной водой, газирование."),
                _stage("mature", "Созревание в бутылках (хол.)", "🥶", "Холодильное созревание."),
                _stage("label", "Розлив/упаковка/маркировка", "📦", "Готовый продукт.")
            ]
        elif is_cheese:
            tail = [
                _stage("prep_cheese", "Подготовка к выработке", "🧰", "Коррекция состава/кальций/закваски."),
                _stage("rennet", "Сычужное свертывание", "🧀", "Внесение фермента → образование сгустка."),
                _stage("curd", "Обработка сгустка", "🔪", "Резка/нагрев/перемешивание → выделение сыворотки."),
                _stage("form", "Формование", "🧱", "Выкладка в формы."),
                _stage("press", "Самопрессование/прессование", "🗜️", "Осушка и уплотнение структуры."),
                _stage("salt_dry", "Посолка/обсушка", "🧂", "Рассол/сухая посолка; обсушка 2–3 суток (10–12 °C)."),
                _stage("ripen", "Созревание", "⏳", "Камеры с контролем T/влажности."),
                _stage("label", "Упаковка/хранение/реализация", "📦", "Контроль качества и выпуск.")
            ]
        else:
            tail = [
                _stage("cooling", "Охлаждение (2–6 °C)", "❄️", "Быстрое охлаждение после пастеризации.",
                       {"min": 2, "max": 6, "unit": "°C"}),
                _stage("steril", "Стерилизация / UHT", "🧪", "Безопасность и длительный срок хранения."),
                _stage("label", "Розлив/упаковка/маркировка", "📦", "Готовый продукт.")
            ]

        return common + tail

    # --- Рендер карточки этапа ---
    def render_step_card(sid, label, icon, desc, color):
        active = (st.session_state.get('selected_step') == sid)
        # Цвет активного этапа — мягкий светло-голубой (не тёмный)
        bg = "#E8F0FE" if active else "white"
        st.markdown(
            f"""
            <div class="step-card" style="border-left:5px solid {color}; background:{bg}">
              <div class="step-title">{icon} {label}</div>
              <div class="step-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return st.button(("✅ " if active else "") + f"Выбрать этап: {label}", key=f"btn_{sid}", use_container_width=True)

    # --- Поля форм по этапам ---
    STEP_FIELDS = {
        "clarify": [
            {"name":"Температура очищения", "key":"t_clean", "unit":"°C", "type":"number", "default":5.0},
            {"name":"Кислотность", "key":"acid_T", "unit":"°Т", "type":"number"},
            {"name":"Сорт молока", "key":"grade", "type":"select", "options":["Высший","1","2","3"], "default":"Высший"},
        ],
        "pasteurization": [
            {"name":"Фактическая T пастеризации", "key":"t_past", "unit":"°C", "type":"number"},
            {"name":"Время выдержки", "key":"time_hold", "unit":"мин", "type":"number"},
        ],
        "cool_to_inoc": [
            {"name":"T заквашивания", "key":"t_inoc", "unit":"°C", "type":"number"},
        ],
        "inoculation": [
            {"name":"Доза закваски", "key":"dose_culture", "unit":"%", "type":"number"},
        ],
        "fermentation": [
            {"name":"T сквашивания", "key":"t_ferm", "unit":"°C", "type":"number"},
            {"name":"Время сквашивания", "key":"time_ferm", "unit":"ч", "type":"number"},
        ],
        "salt": [
            {"name":"Соль", "key":"salt_pct", "unit":"%", "type":"number", "default":1.8},
        ],
        "mix_water": [
            {"name":"Доля воды", "key":"water_pct", "unit":"%", "type":"number"},
        ],
        "cooling": [
            {"name":"Температура охлаждения", "key":"t_cool", "unit":"°C", "type":"number"},
        ],
        "rennet": [
            {"name":"Количество фермента", "key":"rennet_ml", "unit":"мл/100л", "type":"number"},
        ],
        "press": [
            {"name":"Давление/время", "key":"press_params", "unit":"", "type":"text"},
        ],
    }

    # --- Основная логика страницы ---
    pid = st.session_state.get('selected_product', None)
    if pid is None:
        st.info("Выберите продукт на главной странице.")
        if st.button("← На главную"):
            st.session_state['page'] = 'Главная'; st.rerun()
    else:
        # Найти продукт
        prod = None
        if not products.empty and 'product_id' in products.columns:
            m = products[products['product_id'] == int(pid)]
            if not m.empty:
                prod = m.iloc[0].to_dict()
        if prod is None:
            names = {1:"Молоко (коровье)",2:"Молоко (козье)",3:"Сары ірімшік (коровье)",4:"Сары ірімшік (козье)",5:"Айран"}
            prod = {"product_id":pid,"name":names.get(pid,f"Продукт {pid}"),"type":"-","source":"-","description":""}

        # Заголовок
        col1, col2 = st.columns([3,1])
        with col1:
            st.header(prod['name'])
        with col2:
            if st.button("← Назад к продуктам", use_container_width=True):
                st.session_state['page'] = 'Главная'; st.rerun()

        st.write(f"**Тип:** {prod.get('type','-')}  •  **Источник:** {prod.get('source','-')}")
        if prod.get('description'):
            st.caption(prod.get('description'))

        # -------- НОРМАТИВЫ --------
        st.markdown("---")
        st.subheader("🧾 Нормативы качества и безопасности (для молока-сырья)")
        pname = str(prod['name']).lower()
        if "молоко" in pname:
            st.markdown(
                "- **Соматические клетки**: 400–1000 тыс/мл (по сорту)\n"
                "- **Патогенные микроорганизмы**: отсутствуют (в т.ч. сальмонеллы)\n"
                "- **КМАФАнМ**: 1·10⁵ – 4·10⁶ КОЕ/г (не более 1·10⁶)\n"
                "- **Класс по редуктазной пробе**: I–II\n"
                "- **Кислотность**: до 19 °Т\n"
                "- **Плотность**: ≥ 1027 кг/м³; **СОМО** ≥ 8,2%; **ингибирующие вещества** — отсутствуют"
            )
            df_phys = pd.DataFrame({
                "Показатель": ["Кислотность, °Т", "Группа чистоты", "Плотность, кг/м³ (не менее)", "Температура замерзания, °C"],
                "Высший сорт": ["16–18", "I", "1028,0", "не выше −0,520"],
                "Первый сорт": ["16–18", "I", "1027,0", "не выше −0,520"],
                "Второй сорт": ["16–20,99", "II", "1027,0", "не выше −0,520"],
                "Несортовое": ["<15,99 или >21,00", "III", "<1026,9", "выше −0,520"]
            })
            st.dataframe(df_phys, use_container_width=True)
        elif "айран" in pname:
            st.caption("Айран: требования к исходному молоку — как для питьевого молока (см. нормы выше).")
        elif ("сыр" in pname) or ("ірімшік" in pname):
            st.caption("Сыры (в т.ч. сары ірімшік): исходное молоко по ветеринарным/санитарным требованиям; КМАФАнМ ≤ 1×10⁶ КОЕ/г, патогены — отсутствуют.")

        # -------- Процесс (кликабельные этапы) --------
        st.markdown("---")
        st.subheader("💡 Процесс изготовления (кликабельные этапы)")

        steps = _product_steps(prod)  # <— важно: передаём весь prod
        for idx, (sid, label, icon, desc, norm) in enumerate(steps):
            color = color_for_step(sid)
            if render_step_card(sid, label, icon, desc, color):
                st.session_state['selected_step'] = sid
                st.session_state['selected_step_label'] = label
                st.rerun()
            if idx < len(steps) - 1:
                st.markdown('<div class="arrow">⬇️</div>', unsafe_allow_html=True)

        # --- Детали выбранного этапа ---
        if st.session_state.get('selected_step'):
            st.markdown("---")
            sel = st.session_state['selected_step']
            sel_label = st.session_state.get('selected_step_label', sel)
            st.subheader(f"📋 Данные этапа: {sel_label}")

            # Норма этапа (если есть)
            norm = None
            for sid, label, icon, desc, n in steps:
                if sid == sel:
                    norm = n; break
            if norm:
                st.success(f"Норма: {norm.get('min','-')} — {norm.get('max','-')} {norm.get('unit','')}")
                if norm.get('note'):
                    st.caption(norm['note'])

            # Журнал партий
            st.write("**📊 Журнал партий для продукта:**")
            if 'product_id' in samples.columns:
                prod_samples = samples[samples['product_id'] == int(pid)].copy()
            else:
                prod_samples = pd.DataFrame()
            if prod_samples.empty:
                st.info("Партии для этого продукта отсутствуют. Добавьте партию ниже.")
            else:
                st.dataframe(prod_samples.sort_values(by='date_received', ascending=False).reset_index(drop=True))

            # Измерения по партиям
            st.write("**📈 Измерения (Measurements):**")
            if 'sample_id' in measurements.columns and not prod_samples.empty:
                rel = measurements[measurements['sample_id'].isin(prod_samples['sample_id'])].copy()
            else:
                rel = pd.DataFrame()
            if rel.empty:
                st.info("Нет измерений для этих партий.")
            else:
                if 'actual_numeric' not in rel.columns and 'actual_value' in rel.columns:
                    rel['actual_numeric'] = rel['actual_value'].apply(parse_numeric)
                st.dataframe(rel[['sample_id','parameter','unit','actual_value','actual_numeric']].reset_index(drop=True))

            # Форма параметров этапа
            st.markdown("### ➕ Сохранить параметры этапа")
            with st.form(f"form_stage_params_{pid}_{sel}", clear_on_submit=True):
                sample_opts = prod_samples['sample_id'].tolist() if not prod_samples.empty else []
                sample_choice = st.selectbox("Sample ID", options=sample_opts) if sample_opts else None
                vals = {}
                fields = STEP_FIELDS.get(sel, [])
                c1, c2 = st.columns(2)
                for i, f in enumerate(fields):
                    with (c1 if i % 2 == 0 else c2):
                        t = f.get("type","text")
                        label_f = f["name"]
                        key = f["key"]
                        if t == "number":
                            vals[key] = st.number_input(f"{label_f} ({f.get('unit','')})", value=float(f.get("default", 0.0)))
                        elif t == "select":
                            opts = f.get("options", [])
                            default = f.get("default", opts[0] if opts else "")
                            idx = opts.index(default) if (opts and default in opts) else 0
                            vals[key] = st.selectbox(label_f, options=opts, index=idx)
                        else:
                            vals[key] = st.text_input(label_f, value=str(f.get("default","")))
                save_params = st.form_submit_button("💾 Сохранить параметры")

            if save_params:
                if sample_choice is None:
                    st.error("Сначала добавьте партию.")
                else:
                    try:
                        base_id = int(datetime.now().timestamp())
                        rows = []
                        for j, f in enumerate(fields):
                            par_name = f"{sel_label}: {f['name']}"
                            rows.append({
                                "id": base_id + j,
                                "sample_id": int(sample_choice),
                                "parameter": par_name,
                                "unit": f.get("unit",""),
                                "actual_value": str(vals.get(f['key'],"")),
                                "method": "этап/форма"
                            })
                        if rows:
                            df_append = pd.DataFrame(rows)
                            write_header = not MEASUREMENTS_CSV.exists() or MEASUREMENTS_CSV.stat().st_size == 0
                            df_append.to_csv(MEASUREMENTS_CSV, mode='a', index=False, header=write_header, encoding='utf-8-sig')
                            st.cache_data.clear()
                            products, samples, measurements, vitamins, storage = load_csvs()
                            st.success("✅ Параметры этапа сохранены.")
                    except Exception as e:
                        st.error(f"Ошибка сохранения: {e}")

            # Форма быстрой партии
            st.markdown("### ➕ Добавить новую партию (Sample)")
            with st.form(f"form_add_sample_{pid}", clear_on_submit=True):
                try:
                    existing = pd.to_numeric(samples.get('sample_id', pd.Series(dtype='Int64')), errors='coerce').dropna()
                    new_sid = int(existing.max()) + 1 if not existing.empty else 1
                except Exception:
                    new_sid = 1
                c1, c2 = st.columns(2)
                with c1:
                    reg_number = st.text_input("Рег. номер", value=f"A-{new_sid:03d}")
                    date_received = st.date_input("Дата поступления", value=datetime.now().date())
                    storage_days = st.number_input("Срок хранения, дни", min_value=0, value=0)
                with c2:
                    temp_input = st.number_input("Температура (°C)", value=21.0, format="%.2f")
                    humidity = st.number_input("Влажность (%)", value=64)
                    notes = st.text_area("Примечания", value=st.session_state.get('selected_step_label', ''))
                save_sample = st.form_submit_button("💾 Сохранить партию")

            if save_sample:
                row = {
                    "sample_id": int(new_sid),
                    "product_id": int(pid),
                    "reg_number": reg_number,
                    "date_received": date_received.strftime("%Y-%m-%d"),
                    "storage_days": int(storage_days),
                    "conditions": f"{temp_input}°C, {humidity}%",
                    "notes": notes
                }
                try:
                    append_row_csv(SAMPLES_CSV, row, cols_order=["sample_id","product_id","reg_number","date_received","storage_days","conditions","notes"])
                    st.cache_data.clear()
                    products, samples, measurements, vitamins, storage = load_csvs()
                    st.success("✅ Партия добавлена.")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

        # Сводка измерений по продукту
        st.markdown("---")
        st.subheader("📈 Измерения по продукту")
        if 'product_id' in samples.columns and 'sample_id' in measurements.columns:
            prod_samples = samples[samples['product_id'] == int(pid)]
            rel = measurements[measurements['sample_id'].isin(prod_samples['sample_id'])] if not prod_samples.empty else pd.DataFrame()
            if rel.empty:
                st.info("Измерений пока нет.")
            else:
                if 'actual_numeric' not in rel.columns and 'actual_value' in rel.columns:
                    rel['actual_numeric'] = rel['actual_value'].apply(parse_numeric)
                st.dataframe(rel.sort_values(by='sample_id', ascending=False).reset_index(drop=True), use_container_width=True)
        else:
            st.info("Данных пока нет.")

        # Кнопки быстрого доступа
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗜️ Скачать все CSV (ZIP)", use_container_width=True):
                download_zip([PRODUCTS_CSV, SAMPLES_CSV, MEASUREMENTS_CSV, VITAMINS_CSV, STORAGE_CSV])
        with c2:
            if st.button("🔄 Обновить страницу", use_container_width=True):
                st.cache_data.clear()
                products, samples, measurements, vitamins, storage = load_csvs()
                st.rerun()















# ---------------------------
# --- MODELS & ANALYTICS ---
# ---------------------------
elif st.session_state['page'] == 'Модели и аналитика':
    st.title("📊 Модели и аналитика — Опыт D1 и D2 (Айран)")
    st.write("Витрина регрессионных подходов и итоговых графиков (пример).")

    # =========================
    # 1) Вводные таблицы
    # =========================
    st.subheader("📄 Вводные данные")
    c1, c2 = st.columns(2)

    data_D1 = {
        "Группа": ["Контроль", "Опыт 1 (добавка 1)", "Опыт 2 (добавка 2)"],
        "pH": [3.69, 3.65, 3.51],
        "°T": [91, 92, 97],
        "LAB (КОЕ/см³)": [1.2e6, 1.6e6, 2.1e6],
    }
    df_D1 = pd.DataFrame(data_D1)
    df_D1["log10(LAB)"] = np.log10(df_D1["LAB (КОЕ/см³)"].astype(float))
    with c1:
        st.markdown("**Таблица 4. D1 — Айран (7 суток)**")
        st.dataframe(df_D1, use_container_width=True)

    data_D2 = {
        "Группа": ["Контроль", "Опыт 1", "Опыт 2"],
        "Белок %": [1.96, 2.05, 2.23],
        "Углеводы %": [2.73, 3.06, 3.85],
        "Жир %": [2.05, 1.93, 2.71],
        "Влага %": [92.56, 92.26, 90.40],
        "АОА вод. (мг/г)": [0.10, 0.15, 0.12],
        "АОА жир (мг/г)": [0.031, 0.043, 0.041],
        "VitC (мг/100г)": [0.880, 0.904, 0.897],
    }
    df_D2 = pd.DataFrame(data_D2)
    with c2:
        st.markdown("**Таблица 5. D2 — Айран (14 суток)**")
        st.dataframe(df_D2, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Итоговые графики")

    tab1, tab2, tab3 = st.tabs(["D1: кислотность и LAB", "D2: состав и свойства", "Моделирование pH"])

    with tab1:
        fig, ax1 = plt.subplots(figsize=(8,5))
        ax1.bar(df_D1["Группа"], df_D1["pH"])
        ax1.set_ylabel("pH"); ax1.set_title("D1 (7 суток): кислотность и рост LAB")
        ax2 = ax1.twinx(); ax2.plot(df_D1["Группа"], df_D1["log10(LAB)"], marker="o", linewidth=2)
        ax2.set_ylabel("log10(LAB)")
        st.pyplot(fig, use_container_width=True)

    with tab2:
        df_comp = df_D2.melt(id_vars="Группа",
                             value_vars=["Белок %", "Углеводы %", "Жир %"],
                             var_name="Показатель", value_name="Значение")
        fig1, ax = plt.subplots(figsize=(8,5))
        groups = df_comp["Группа"].unique()
        cats = df_comp["Показатель"].unique()
        x = np.arange(len(groups))
        width = 0.8 / len(cats)
        for i, cat in enumerate(cats):
            vals = df_comp[df_comp["Показатель"] == cat]["Значение"].values
            ax.bar(x + i*width - (len(cats)-1)*width/2, vals, width=width, label=cat)
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel("Процент содержания (%)"); ax.set_title("D2 (14 суток): состав айрана"); ax.legend()
        st.pyplot(fig1, use_container_width=True)

        fig2, axes = plt.subplots(1, 2, figsize=(12,5))
        axes[0].bar(df_D2["Группа"], df_D2["АОА вод. (мг/г)"])
        axes[0].set_title("АОА (водная фаза)"); axes[0].set_ylabel("АОА, мг/г")
        axes[1].bar(df_D2["Группа"], df_D2["VitC (мг/100г)"])
        axes[1].set_title("Витамин C"); axes[1].set_ylabel("VitC, мг/100г")
        plt.suptitle("D2: функциональные свойства", fontsize=14)
        st.pyplot(fig2, use_container_width=True)

    with tab3:
        time = np.array([2, 4, 6, 8, 10])
        ph_control = np.array([4.515, 4.433, 4.386, 4.352, 4.325])
        ph_exp1 = np.array([4.464, 4.394, 4.352, 4.323, 4.300])
        ph_exp2 = np.array([4.419, 4.333, 4.282, 4.246, 4.218])

        st.markdown("**Динамика pH (контроль, опыт 1, опыт 2)**")
        fig0, ax0 = plt.subplots(figsize=(8,5))
        ax0.plot(time, ph_control, 'o-', label='Контроль')
        ax0.plot(time, ph_exp1, 's-', label='Опыт 1')
        ax0.plot(time, ph_exp2, '^-', label='Опыт 2')
        ax0.set_xlabel('Время ферментации, ч'); ax0.set_ylabel('pH')
        ax0.set_title('Сравнение динамики pH (2–10 ч)')
        ax0.grid(True, alpha=0.3); ax0.legend()
        st.pyplot(fig0, use_container_width=True)

        st.markdown("**Модели для pH(t): логарифмическая и гиперболическая (без SciPy)**")
        t_fit = np.array([1, 2, 3, 4, 5, 6, 8, 10], dtype=float)
        pH_exp = np.array([4.65, 4.50, 4.33, 4.20, 4.05, 3.90, 3.78, 3.70], dtype=float)

        ln_t = np.log(t_fit)
        c1, c0 = np.polyfit(ln_t, pH_exp, 1)  # y = c1*ln(t) + c0
        alpha = c0; beta = -c1

        inv_t = 1.0 / t_fit
        m, a_intercept = np.polyfit(inv_t, pH_exp, 1)  # y = m*(1/t) + a
        a = a_intercept; b = m

        t_pred = np.linspace(1, 10, 100)
        pH_log_pred = alpha - beta * np.log(t_pred)
        pH_inv_pred = a + b / t_pred

        fig1, ax1 = plt.subplots(figsize=(8,5))
        ax1.scatter(t_fit, pH_exp, color='black', label='Экспериментальные точки')
        ax1.plot(t_pred, pH_log_pred, label='Логарифмическая  pH = α - β ln(t)')
        ax1.plot(t_pred, pH_inv_pred, linestyle='--', label='Гиперболическая  pH = a + b/t')
        ax1.set_xlabel('Время, ч'); ax1.set_ylabel('pH'); ax1.grid(True, alpha=0.3)
        ax1.set_title('Моделирование динамики pH при ферментации айрана')
        ax1.legend()
        st.pyplot(fig1, use_container_width=True)

# ---------------------------
# --- Footer ---
# ---------------------------
st.markdown("---")
st.markdown("""
<div class='footer'>
    <div style='text-align: center; padding: 20px;'>
        <h3>🥛 Milk Digitalization Platform</h3>
        <p>Версия 2.1 | Обновлена страница «Продукт»: динамические этапы + нормы (без больших изображений)</p>
        <p>📧 Поддержка: demo@milk-digitalization.kz | 📞 +7 (777) 123-45-67</p>
        <div style='margin-top: 15px;'>
            <small>Мониторинг партий, контроль качества, аналитика, прогнозирование</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Статистика данных")
if not products.empty: st.sidebar.write(f"Продукты: {len(products)}")
if not samples.empty: st.sidebar.write(f"Партии: {len(samples)}")
if not measurements.empty: st.sidebar.write(f"Измерения: {len(measurements)}")

st.sidebar.markdown("---")
st.sidebar.info("""
**Использование:**
1) Выберите продукт на главной странице
2) Сверьте с нормативами качества
3) Перейдите по этапам процесса (карточки)
4) Сохраняйте параметры и/или добавляйте партии
5) Смотрите измерения и аналитику
""")

# Кнопка сброса
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Сбросить состояние приложения"):
    st.session_state.clear()
    st.rerun()
