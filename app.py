# app.py — Milk Digitalization v2.1 (Clean design, no emojis)
# Требует: pandas, numpy, streamlit, matplotlib, seaborn  (sklearn — опционально)

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
# --- UI стили (спокойная палитра) ---
# ---------------------------
st.set_page_config(page_title="Milk Digitalization", layout="wide")
st.markdown(
    """
<style>
:root{
  --bg:#f7f8fa;
  --card:#ffffff;
  --text:#1f2937;
  --muted:#6b7280;
  --accent:#0b4c86;
  --accent-2:#2c5282;
  --border:#e5e7eb;
  --shadow:0 6px 16px rgba(0,0,0,0.06);
}

body{background:var(--bg);}
.card{background:var(--card);padding:14px;border-radius:12px;box-shadow:var(--shadow);margin-bottom:12px;border:1px solid var(--border)}
.prod-title{font-weight:700;color:var(--accent)}
.step-card{background:var(--card);padding:16px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.08);margin:8px 0;border-left:4px solid var(--accent);transition:transform .2s ease;border-top:1px solid var(--border);}
.step-card:hover{transform:translateY(-2px)}
.step-title{font-weight:600;color:var(--accent);margin-bottom:6px}
.step-desc{color:var(--muted);font-size:14px}
.arrow{text-align:center;font-size:18px;margin:2px 0;color:var(--accent)}
.step-small{font-size:13px;color:#333}
.small-muted{color:var(--muted);font-size:13px}
.footer{color:#8a8f98;font-size:12px;margin-top:18px}

.product-card{background:var(--card);color:var(--text);padding:18px;border-radius:14px;margin:10px 0;box-shadow:var(--shadow);border:1px solid var(--border)}
.product-card:hover{transform:translateY(-3px);box-shadow:0 10px 24px rgba(0,0,0,0.10)}

/* Полная кликабельность: убираем стили у кнопки и растягиваем на карточку */
form.card-form{margin:0}
button.card-btn{all:unset;display:block;width:100%;cursor:pointer}
button.card-btn:focus{outline:2px solid var(--accent-2);outline-offset:4px;border-radius:12px}

.bad{background:#fff5f5}
.ok{background:#f0fbf4}

/* Typography tweaks */
h1,h2,h3{color:var(--text)}
</style>
""",
    unsafe_allow_html=True,
)

# Мягкие цвета для этапов
STEP_COLORS = {
    "pasteurization":"#b91c1c1a",
    "cooling":"#1d4ed81a",
    "fermentation":"#1665341a",
    "accept":"#0369a11a",
    "normalization":"#92400e1a",
    "homogenization":"#6d28d91a",
    "inoculation":"#0f766e1a",
    "coagulation":"#9a34121a",
    "pressing":"#3341551a",
    "filtration":"#1e40af1a",
    "storage":"#0e74901a",
    "final":"#1118271a"
}

# Спокойные градиенты для продуктов (едва заметные)
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
    return "#e5e7eb"

def color_for_product(product_id):
    return PRODUCT_COLORS.get(product_id, "linear-gradient(135deg,#f9fafb 0%,#eef2f7 100%)")

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
    st.sidebar.success("Все CSV найдены")

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
    st.title("Milk Digitalization — демо платформа")
    st.markdown(
        """
        Добро пожаловать в платформу **Milk Digitalization**.
        Здесь вы можете мониторить партии, анализировать качество
        и визуализировать ключевые показатели производства.
        """
    )
    st.markdown("---")

    # ---------- Исходные продукты ----------
    fixed_products = [
        {"product_id": 1, "name": "Молоко (коровье)",       "type": "Молоко",        "source": "Коровье", "description": "Свежее пастеризованное молоко"},
        {"product_id": 2, "name": "Молоко (козье)",         "type": "Молоко",        "source": "Козье",   "description": "Натуральное фермерское козье молоко"},
        {"product_id": 3, "name": "Сары ірімшік (коровье)", "type": "Сыр",           "source": "Коровье", "description": "Твёрдый сыр традиционной выработки"},
        {"product_id": 4, "name": "Сары ірімшік (козье)",   "type": "Сыр",           "source": "Козье",   "description": "Твёрдый сыр из козьего молока"},
        {"product_id": 5, "name": "Айран",                  "type": "Кисломолочный", "source": "Коровье", "description": "Освежающий кисломолочный продукт"},
    ]

    # ---------- Подготовка данных ----------
    display_products = []
    if not products.empty and 'product_id' in products.columns:
        for fp in fixed_products:
            match = products[products['product_id'] == fp['product_id']]
            display_products.append(match.iloc[0].to_dict() if not match.empty else fp)
    else:
        display_products = fixed_products

    # ---------- Стили карточек + кнопки-обёртки ----------
    st.markdown(
        """
        <style>
          .product-card {
              background: linear-gradient(135deg, #fafafa, #f3f4f6);
              border-radius: 16px;
              padding: 18px;
              box-shadow: 0 6px 16px rgba(0,0,0,0.08);
              transition: all 0.2s ease;
              color: #111827;
              height: 100%;
              cursor: pointer;
              border: 1px solid #e5e7eb;
          }
          .product-card:hover {
              transform: translateY(-4px);
              box-shadow: 0 12px 28px rgba(0,0,0,0.12);
              filter: brightness(1.02);
          }
          .product-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 4px; }
          .product-meta { font-size: 0.9rem; color: #374151; margin-bottom: 8px; }
          .product-desc { font-size: 0.92rem; color: #4b5563; margin-bottom: 0; }

          form.card-form { margin: 0; }
          button.card-btn { all: unset; display: block; width: 100%; cursor: pointer; }
          button.card-btn:focus { outline: 2px solid #2563eb; outline-offset: 4px; border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Отрисовка карточек (вся карточка = submit) ----------
    st.subheader("Наши продукты")
    cols = st.columns(2)

    def gradient_for(pid):
        gradients = [
            "linear-gradient(135deg, #fafafa, #f3f4f6)",
            "linear-gradient(135deg, #fafafa, #f3f4f6)",
            "linear-gradient(135deg, #fafafa, #f3f4f6)",
            "linear-gradient(135deg, #fafafa, #f3f4f6)",
            "linear-gradient(135deg, #fafafa, #f3f4f6)"
        ]
        return gradients[(pid - 1) % len(gradients)]

    for i, p in enumerate(display_products):
        pid = int(p["product_id"]) 
        grad = gradient_for(pid)

        card_html = f"""
        <form class="card-form" method="get">
          <input type="hidden" name="goto" value="product"/>
          <input type="hidden" name="pid" value="{pid}"/>
          <button class="card-btn" type="submit" aria-label="Открыть {p["name"]}">
            <div class="product-card" style="background: {grad};">
                <div class="product-title">{p["name"]}</div>
                <div class="product-meta">Тип: {p["type"]} • Источник: {p["source"]}</div>
                <div class="product-desc">{p["description"]}</div>
            </div>
          </button>
        </form>
        """
        with cols[i % 2]:
            st.markdown(card_html, unsafe_allow_html=True)

    # ---------- Обработка параметров (в этой же странице) ----------
    try:
        params = st.query_params
        if params.get("goto") == "product" and params.get("pid"):
            st.session_state['selected_product'] = int(params.get("pid"))
            st.session_state['page'] = 'Продукт'
            st.query_params.clear()
            st.rerun()
    except Exception:
        pass

    # ---------- Быстрые действия ----------
    st.markdown("---")
    st.subheader("Быстрые действия")
    c1, c2, c3 = st.columns(3)
    if c1.button("Журнал партий", use_container_width=True):
        st.session_state['page'] = 'Продукт'; st.rerun()
    if c2.button("Аналитика", use_container_width=True):
        st.session_state['page'] = 'Модели и аналитика'; st.rerun()
    if c3.button("Скачать CSV ZIP", use_container_width=True):
        download_zip([PRODUCTS_CSV, SAMPLES_CSV, MEASUREMENTS_CSV, VITAMINS_CSV, STORAGE_CSV])


# ---------------------------
# --- PRODUCT PAGE (динамическая, без больших картинок) ---
# ---------------------------
elif st.session_state['page'] == 'Продукт':

    # --- Генератор этапа ---
    def _stage(id, label, desc="", norm=None):
        return (id, label, desc, norm or {})

    # --- Правильная логика этапов по продукту (с учётом source) ---
    def _product_steps(prod: dict):
        name = str(prod.get('name', '')).strip()
        source = str(prod.get('source', '')).strip()

        nlow = name.lower()
        slow = source.lower()

        is_ayran  = "айран" in nlow
        is_cheese = ("ірімшік" in nlow) or ("сыр" in nlow)
        is_milk   = ("молоко" in nlow)

        goat = (
            ("козье" in nlow) or ("козий" in nlow) or ("goat" in nlow) or ("ешкі" in nlow) or
            ("козье" in slow) or ("козий" in slow) or ("goat" in slow) or ("ешкі" in slow)
        )

        common = [
            _stage("accept", "Приёмка сырья", "Осмотр тары, органолептика, экспресс-анализ состава/обсеменённости."),
            _stage("clarify", "Очистка и сортировка (4–6 °C)", "Фильтрация/сепараторы. Оценка чистоты, кислотности (°Т), определение сорта.",
                   {"min": 4, "max": 6, "unit": "°C", "note": "Охлаждение до 4–6 °C замедляет рост бактерий"}),
            _stage("normalization", "Нормализация состава", "Приведение к нормам по жирности/белку/витаминам/минералам."),
        ]

        need_homogenization = (is_ayran or (is_milk and not goat) or (is_cheese and not goat))
        if need_homogenization:
            common.append(_stage("homogenization", "Гомогенизация", "Дробление жировых шариков → однородность, отсутствие отстоя."))

        common.append(_stage(
            "pasteurization", "Пастеризация (65–69 °C)", "Термообработка для снижения микрофлоры.",
            {"min": 65, "max": 69, "unit": "°C", "note": "Пастеризация согласно рецептуре/ГОСТ"}
        ))

        if is_ayran:
            tail = [
                _stage("cool_to_inoc", "Охлаждение до заквашивания (35–45 °C)", "Перед внесением закваски.", {"min": 35, "max": 45, "unit": "°C"}),
                _stage("inoculation", "Внесение закваски", "Культуры: стрептококк, болгарская палочка, дрожжи."),
                _stage("fermentation", "Сквашивание (20–25 °C)", "Выдержка при заданной температуре.", {"min": 20, "max": 25, "unit": "°C"}),
                _stage("salt", "Добавление соли (1.5–2%)", "Перемешать до однородности."),
                _stage("mix_water", "Смешивание с водой / газирование", "Смешивание с кипячёной водой, газирование."),
                _stage("mature", "Созревание в бутылках (холод)", "Холодильное созревание."),
                _stage("label", "Розлив/упаковка/маркировка", "Готовый продукт."),
            ]
        elif is_cheese:
            tail = [
                _stage("prep_cheese", "Подготовка к выработке", "Коррекция состава/кальций/закваски."),
                _stage("rennet", "Сычужное свертывание", "Внесение фермента → образование сгустка."),
                _stage("curd", "Обработка сгустка", "Резка/нагрев/перемешивание → выделение сыворотки."),
                _stage("form", "Формование", "Выкладка в формы."),
                _stage("press", "Самопрессование/прессование", "Осушка и уплотнение структуры."),
                _stage("salt_dry", "Посолка/обсушка", "Рассол/сухая посолка; обсушка 2–3 суток (10–12 °C)."),
                _stage("ripen", "Созревание", "Камеры с контролем температуры и влажности."),
                _stage("label", "Упаковка/хранение/реализация", "Контроль качества и выпуск."),
            ]
        else:
            tail = [
                _stage("cooling", "Охлаждение (2–6 °C)", "Быстрое охлаждение после пастеризации.", {"min": 2, "max": 6, "unit": "°C"}),
                _stage("steril", "Стерилизация / UHT", "Безопасность и длительный срок хранения."),
                _stage("label", "Розлив/упаковка/маркировка", "Готовый продукт."),
            ]

        return common + tail

    # --- Рендер карточки этапа ---
    def render_step_card(sid, label, desc, color):
        active = (st.session_state.get('selected_step') == sid)
        bg = "#EEF2F7" if active else "white"
        st.markdown(
            f"""
            <div class="step-card" style="border-left:4px solid {color}; background:{bg}">
              <div class="step-title">{label}</div>
              <div class="step-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return st.button(("Выбран: " if active else "Выбрать этап: ") + f"{label}", key=f"btn_{sid}", use_container_width=True)

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

    pid = st.session_state.get('selected_product', None)
    if pid is None:
        st.info("Выберите продукт на главной странице.")
        if st.button("На главную"):
            st.session_state['page'] = 'Главная'; st.rerun()
    else:
        prod = None
        if not products.empty and 'product_id' in products.columns:
            m = products[products['product_id'] == int(pid)]
            if not m.empty:
                prod = m.iloc[0].to_dict()
        if prod is None:
            names = {1:"Молоко (коровье)",2:"Молоко (козье)",3:"Сары ірімшік (коровье)",4:"Сары ірімшік (козье)",5:"Айран"}
            prod = {"product_id":pid,"name":names.get(pid,f"Продукт {pid}"),"type":"-","source":"-","description":""}

        col1, col2 = st.columns([3,1])
        with col1:
            st.header(prod['name'])
        with col2:
            if st.button("Назад к продуктам", use_container_width=True):
                st.session_state['page'] = 'Главная'; st.rerun()

        st.write(f"**Тип:** {prod.get('type','-')}  •  **Источник:** {prod.get('source','-')}")
        if prod.get('description'):
            st.caption(prod.get('description'))

        # -------- НОРМАТИВЫ --------
        st.markdown("---")
        st.subheader("Нормативы качества и безопасности (для молока-сырья)")
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
        st.subheader("Процесс изготовления (кликабельные этапы)")

        steps = _product_steps(prod)
        for idx, (sid, label, desc, norm) in enumerate(steps):
            color = color_for_step(sid)
            if render_step_card(sid, label, desc, color):
                st.session_state['selected_step'] = sid
                st.session_state['selected_step_label'] = label
                st.rerun()
            if idx < len(steps) - 1:
                st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)

        # --- Детали выбранного этапа ---
        if st.session_state.get('selected_step'):
            st.markdown("---")
            sel = st.session_state['selected_step']
            sel_label = st.session_state.get('selected_step_label', sel)
            st.subheader(f"Данные этапа: {sel_label}")

            norm = None
            for sid, label, desc, n in steps:
                if sid == sel:
                    norm = n; break
            if norm:
                st.success(f"Норма: {norm.get('min','-')} — {norm.get('max','-')} {norm.get('unit','')}")
                if norm.get('note'):
                    st.caption(norm['note'])

            st.write("**Журнал партий для продукта:**")
            if 'product_id' in samples.columns:
                prod_samples = samples[samples['product_id'] == int(pid)].copy()
            else:
                prod_samples = pd.DataFrame()
            if prod_samples.empty:
                st.info("Партии для этого продукта отсутствуют. Добавьте партию ниже.")
            else:
                st.dataframe(prod_samples.sort_values(by='date_received', ascending=False).reset_index(drop=True))

            st.write("**Измерения (Measurements):**")
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

            st.markdown("### Сохранить параметры этапа")
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
                save_params = st.form_submit_button("Сохранить параметры")

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
                            st.success("Параметры этапа сохранены.")
                    except Exception as e:
                        st.error(f"Ошибка сохранения: {e}")

            st.markdown("### Добавить новую партию (Sample)")
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
                save_sample = st.form_submit_button("Сохранить партию")

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
                    st.success("Партия добавлена.")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

        st.markdown("---")
        st.subheader("Измерения по продукту")
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

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Скачать все CSV (ZIP)", use_container_width=True):
                download_zip([PRODUCTS_CSV, SAMPLES_CSV, MEASUREMENTS_CSV, VITAMINS_CSV, STORAGE_CSV])
        with c2:
            if st.button("Обновить страницу", use_container_width=True):
                st.cache_data.clear()
                products, samples, measurements, vitamins, storage = load_csvs()
                st.rerun()


# ---------------------------
# --- MODELS & ANALYTICS (v3: карточки без GET, остаёмся в разделе) ---
# ---------------------------
elif st.session_state['page'] == 'Модели и аналитика':
    st.title("Модели и аналитика")
    st.caption("Выберите продукт, чтобы открыть витрину аналитики и моделей.")

    # состояние выбора внутри раздела аналитики
    if 'analytics_selected_product' not in st.session_state:
        st.session_state['analytics_selected_product'] = None

    # список продуктов (как на Главной)
    fixed_products = [
        {"product_id": 1, "name": "Молоко (коровье)",       "type": "Молоко",        "source": "Коровье", "description": "Свежее пастеризованное молоко"},
        {"product_id": 2, "name": "Молоко (козье)",         "type": "Молоко",        "source": "Козье",   "description": "Натуральное фермерское козье молоко"},
        {"product_id": 3, "name": "Сары ірімшік (коровье)", "type": "Сыр",           "source": "Коровье", "description": "Твёрдый сыр традиционной выработки"},
        {"product_id": 4, "name": "Сары ірімшік (козье)",   "type": "Сыр",           "source": "Козье",   "description": "Твёрдый сыр из козьего молока"},
        {"product_id": 5, "name": "Айран",                  "type": "Кисломолочный", "source": "Коровье", "description": "Освежающий кисломолочный продукт"},
    ]
    display_products = []
    if not products.empty and 'product_id' in products.columns:
        for fp in fixed_products:
            match = products[products['product_id'] == fp['product_id']]
            display_products.append(match.iloc[0].to_dict() if not match.empty else fp)
    else:
        display_products = fixed_products

    # стили карточек
    st.markdown(
        """
        <style>
          .product-card {
              background: linear-gradient(135deg, #fafafa, #f3f4f6);
              border-radius: 16px;
              padding: 18px;
              box-shadow: 0 6px 16px rgba(0,0,0,0.08);
              transition: all 0.2s ease;
              color: #111827;
              border: 1px solid #e5e7eb;
              margin-bottom: 10px;
          }
          .product-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 4px; }
          .product-meta { font-size: 0.9rem; color: #374151; margin-bottom: 8px; }
          .product-desc { font-size: 0.92rem; color: #4b5563; margin-bottom: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # карточки (кнопка под карточкой -> без GET)
    st.subheader("Продукты")
    cols = st.columns(2)
    for i, p in enumerate(display_products):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="product-card">
                  <div class="product-title">{p["name"]}</div>
                  <div class="product-meta">Тип: {p["type"]} • Источник: {p["source"]}</div>
                  <div class="product-desc">{p["description"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Открыть аналитику", key=f"open_analytics_{int(p['product_id'])}", use_container_width=True):
                st.session_state['analytics_selected_product'] = int(p["product_id"])
                st.session_state['page'] = 'Модели и аналитика'   # жёстко остаёмся в разделе
                st.rerun()

    # ===== отрисовщики аналитики =====
    def render_airan_analytics():
        st.markdown("---")
        st.header("Айран — аналитика и модели")

        st.subheader("Вводные данные")
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
        st.subheader("Итоговые графики")
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

            fig0, ax0 = plt.subplots(figsize=(8,5))
            ax0.plot(time, ph_control, 'o-', label='Контроль')
            ax0.plot(time, ph_exp1, 's-', label='Опыт 1')
            ax0.plot(time, ph_exp2, '^-', label='Опыт 2')
            ax0.set_xlabel('Время ферментации, ч'); ax0.set_ylabel('pH')
            ax0.set_title('Сравнение динамики pH (2–10 ч)')
            ax0.grid(True, alpha=0.3); ax0.legend()
            st.pyplot(fig0, use_container_width=True)

            t_fit = np.array([1, 2, 3, 4, 5, 6, 8, 10], dtype=float)
            pH_exp = np.array([4.65, 4.50, 4.33, 4.20, 4.05, 3.90, 3.78, 3.70], dtype=float)
            ln_t = np.log(t_fit)
            c1_, c0_ = np.polyfit(ln_t, pH_exp, 1)  # y = c1*ln(t) + c0
            alpha = c0_; beta = -c1_
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

    def render_cheese_analytics():
        st.markdown("---")
        st.header("Сары ірімшік — аналитика и модели")

        # ==== ваш код (с минимальными правками) ====
        KPI_cow = {'M':24.63, 'Pro':11.43, 'Fat':35.06, 'Sol':75.36}
        params = {'k_coag': 0.35, 'k_drain': 0.08, 'k_dry': 0.0018, 'n_page': 1.12}
        experiments = [
            {'name': 'Контроль', 'additive': 0.05,  'color': (0, 0.45, 0.74)},
            {'name': 'Опыт-1',   'additive': 0.045, 'color': (0.85, 0.33, 0.1)}
        ]

        coag  = lambda t, k: 1 - np.exp(-k*t)
        drain = lambda t, k, M0, Mmin: Mmin + (M0 - Mmin)*np.exp(-k*t)
        # исправлена опечатка: добавлен знак умножения перед np.exp
        pagef  = lambda t, k, n, M0, Mf: Mf + (M0 - Mf)*np.exp(-(k*t)*n)

        # 1) Сравнение двух опытов — динамика влажности M(t)
        t = np.linspace(0, 8, 200)
        figA, axA = plt.subplots(figsize=(8,5))
        axA.set_title('Сравнение двух опытов — динамика влажности M(t)')
        axA.set_xlabel('Время, ч'); axA.set_ylabel('Влажность, %'); axA.grid(True)

        for expi in experiments:
            M0 = 55 - 100*expi['additive']
            Mmin = KPI_cow['M']
            M_drain = drain(t, params['k_drain']*(1+0.2*expi['additive']), M0, 35)
            M_page  = pagef(t, params['k_dry']*(1+0.3*expi['additive']), params['n_page'], 35, Mmin)
            M_total = (M_drain + M_page)/2
            axA.plot(t, M_total, linewidth=2.2, color=expi['color'],
                     label=f"{expi['name']} (добавка {100*expi['additive']:.1f}%)")
            axA.text(t[-1], M_total[-1], f" {M_total[-1]:.1f}%", color=expi['color'], weight='bold')
        axA.legend()
        st.pyplot(figA, use_container_width=True)

        # 2) Степень коагуляции α(t)
        figB, axB = plt.subplots(figsize=(8,5))
        axB.set_title('Степень коагуляции α(t)')
        axB.set_xlabel('Время, ч'); axB.set_ylabel('α'); axB.grid(True)
        for expi in experiments:
            axB.plot(t, coag(t, params['k_coag']*(1+0.1*expi['additive'])),
                     linewidth=2, color=expi['color'],
                     label=f"{expi['name']} ({100*expi['additive']:.1f}%)")
        axB.set_ylim([0,1]); axB.legend()
        st.pyplot(figB, use_container_width=True)

        # 3) Сушка (Page): M(t)
        figC, axC = plt.subplots(figsize=(8,5))
        axC.set_title('Сушка (Page): M(t)')
        axC.set_xlabel('Время, ч'); axC.set_ylabel('Влажность, %'); axC.grid(True)
        for expi in experiments:
            axC.plot(t, pagef(t, params['k_dry']*(1+0.3*expi['additive']), params['n_page'], 35, KPI_cow['M']),
                     linewidth=2, color=expi['color'],
                     label=f"{expi['name']} ({100*expi['additive']:.1f}%)")
        axC.set_ylim([34.8, 35.05]); axC.legend()
        st.pyplot(figC, use_container_width=True)

        # 4) Дренаж/пресс: влажность M(t)
        t_drain = np.linspace(0, 1.5, 180)
        M0_drain = 42.5; M_eq_press = 35.0
        figD, axD = plt.subplots(figsize=(8,5))
        axD.set_title('Дренаж/пресс: влажность M(t)')
        axD.set_xlabel('Время, ч'); axD.set_ylabel('Влажность, %'); axD.grid(True)
        for expi in experiments:
            k_drain_eff = params['k_drain']*(1+0.25*expi['additive'])
            M_drain_t = M_eq_press + (M0_drain - M_eq_press)*np.exp(-k_drain_eff*t_drain)
            axD.plot(t_drain, M_drain_t, linewidth=2, color=expi['color'],
                     label=f"{expi['name']} ({100*expi['additive']:.1f}%)")
        axD.legend()
        st.pyplot(figD, use_container_width=True)

        # 5) Дренаж: относительная потеря сыворотки
        figE, axE = plt.subplots(figsize=(8,5))
        axE.set_title('Дренаж: относительная потеря сыворотки')
        axE.set_xlabel('Время, ч'); axE.set_ylabel('Доля, отн. ед.'); axE.grid(True)
        for expi in experiments:
            k_drain_eff = params['k_drain']*(1+0.25*expi['additive'])
            M_drain_t = M_eq_press + (M0_drain - M_eq_press)*np.exp(-k_drain_eff*t_drain)
            loss_rel = (M0_drain - M_drain_t) / (M0_drain - M_eq_press)
            axE.plot(t_drain, loss_rel, linewidth=2, color=expi['color'],
                     label=f"{expi['name']} ({100*expi['additive']:.1f}%)")
        axE.set_ylim([0,1]); axE.legend()
        st.pyplot(figE, use_container_width=True)

        # 6) Посол: NaCl на поверхности и в центре
        t_salt = np.linspace(0, 2.0, 181)
        C_brine = 0.20
        k_surf_base, k_center_base = 2.2, 0.55
        figF, axF = plt.subplots(figsize=(8,5))
        axF.set_title('Посол: NaCl на поверхности и в центре')
        axF.set_xlabel('Время, ч'); axF.set_ylabel('Массовая доля NaCl'); axF.grid(True)
        for expi in experiments:
            k_surf   = k_surf_base*(1+0.10*expi['additive'])
            k_center = k_center_base*(1+0.10*expi['additive'])
            C_surf   = C_brine*(1 - np.exp(-k_surf*t_salt))
            C_center = C_brine*(1 - np.exp(-k_center*t_salt))
            axF.plot(t_salt, C_surf, '-',  linewidth=2, color=expi['color'])
            axF.plot(t_salt, C_center, '--', linewidth=2, color=expi['color'])
        axF.axhline(C_brine, color='k', linestyle=':', label='Рассол 20%')
        axF.legend()
        st.pyplot(figF, use_container_width=True)

    def render_empty_analytics(name: str):
        st.markdown("---")
        st.header(f"{name} — аналитика и модели")
        st.info("Для этого продукта аналитика пока не настроена.")

    # === если выбран продукт — показываем аналитику ===
    sel_pid = st.session_state['analytics_selected_product']
    if sel_pid is not None:
        prod_meta = next((p for p in display_products if int(p["product_id"]) == int(sel_pid)), None)
        c1, c2 = st.columns([3,1])
        with c1:
            st.subheader(f"Продукт: {prod_meta['name'] if prod_meta else f'ID {sel_pid}'}")
            if prod_meta:
                st.caption(f"Тип: {prod_meta.get('type','-')} • Источник: {prod_meta.get('source','-')}")
        with c2:
            if st.button("← Назад к продуктам", use_container_width=True):
                st.session_state['analytics_selected_product'] = None
                st.session_state['page'] = 'Модели и аналитика'
                st.rerun()

        if sel_pid == 5:
            render_airan_analytics()
        elif sel_pid in (3, 4):  # Сары ірімшік (коровье/козье)
            render_cheese_analytics()
        else:
            render_empty_analytics(prod_meta['name'] if prod_meta else f"Продукт {sel_pid}")


# ---------------------------
# --- Footer ---
# ---------------------------
st.markdown("---")
st.markdown(
    """
<div class='footer'>
    <div style='text-align:center; padding: 16px;'>
        <h3>Milk Digitalization Platform</h3>
        <p>Версия 2.1 | Обновлена страница «Продукт»: динамические этапы + нормы (без больших изображений)</p>
        <p>Поддержка: demo@milk-digitalization.kz | +7 (777) 123-45-67</p>
        <div style='margin-top: 10px;'>
            <small>Мониторинг партий, контроль качества, аналитика, прогнозирование</small>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Статистика данных")
if not products.empty: st.sidebar.write(f"Продукты: {len(products)}")
if not samples.empty: st.sidebar.write(f"Партии: {len(samples)}")
if not measurements.empty: st.sidebar.write(f"Измерения: {len(measurements)}")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Использование:**
    1) Выберите продукт на главной странице
    2) Сверьте с нормативами качества
    3) Перейдите по этапам процесса (карточки)
    4) Сохраняйте параметры и/или добавляйте партии
    5) Смотрите измерения и аналитику
    """
)

st.sidebar.markdown("---")
if st.sidebar.button("Сбросить состояние приложения"):
    st.session_state.clear()
    st.rerun()
