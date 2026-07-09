"""
FireRisk DZ — Tableau de bord analytique national.

Carte de risque incendie par wilaya (heuristique historique + météo
récente), détail par wilaya, saisonnalité, corrélations météo <-> feu.

Lancer avec : streamlit run app.py
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FireRisk DZ", page_icon="🔥", layout="wide", initial_sidebar_state="collapsed")

# ---------- Palette & styles ----------
ACCENT = "#ff7a45"
CARD_BG = "#171c22"
BG = "#0f1216"
BORDER = "rgba(255,255,255,0.07)"
RISK_COLORS = {
    "Faible": "#4caf50",
    "Modéré": "#ffca28",
    "Élevé": "#ff9800",
    "Très élevé": "#e53935",
    "Hors périmètre": "#3a3f47",
}
RISK_ICONS = {"Faible": "🟢", "Modéré": "🟡", "Élevé": "🟠", "Très élevé": "🔴", "Hors périmètre": "⚪"}

# ---------- Vue grand public : textes en langage courant ----------
RISK_ADVICE = {
    "Faible": (
        "Risque faible aujourd'hui",
        "Les conditions ne favorisent pas les départs de feu. Restez tout de même prudent avec le feu en forêt.",
    ),
    "Modéré": (
        "Risque modéré aujourd'hui",
        "Soyez vigilant : évitez de jeter des mégots ou d'allumer un feu près des zones boisées.",
    ),
    "Élevé": (
        "Risque élevé aujourd'hui",
        "Évitez tout feu en extérieur (barbecue, brûlage de déchets verts). Signalez immédiatement toute fumée suspecte.",
    ),
    "Très élevé": (
        "Danger — risque très élevé aujourd'hui",
        "N'allumez aucun feu, même à proximité des habitations. En cas de départ de feu, appelez sans attendre la Protection civile : 14.",
    ),
    "Hors périmètre": (
        "Non concernée",
        "Cette wilaya n'a pas de couverture forestière significative : le risque d'incendie de forêt n'y est pas évalué.",
    ),
}


def qualify_temp(t):
    if t >= 38:
        return "très chaud"
    if t >= 30:
        return "chaud"
    if t >= 20:
        return "doux"
    return "frais"


def qualify_humidity(h):
    if h < 25:
        return "l'air est très sec"
    if h < 45:
        return "l'air est sec"
    return "l'humidité est normale"


def qualify_wind(w):
    if w >= 35:
        return "le vent souffle très fort"
    if w >= 20:
        return "le vent souffle fort"
    if w >= 10:
        return "il y a un peu de vent"
    return "il n'y a presque pas de vent"

st.markdown(f"""
<style>
    .block-container {{
        padding-top: 0.6rem; padding-bottom: 1.5rem;
        padding-left: 1.4rem; padding-right: 1.4rem;
        max-width: 100% !important;
    }}
    #MainMenu {{visibility: hidden;}}
    header[data-testid="stHeader"] {{ height: 0; }}

    .topbar {{
        display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
        gap: 6px 18px; background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 8px 16px; margin-bottom: 10px;
    }}
    .topbar h1 {{
        font-size: 1.25rem; margin: 0; font-weight: 800; white-space: nowrap;
        background: linear-gradient(90deg, #ffffff, #ffb347);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .topbar .meta {{ color: #8b95a1; font-size: 0.72rem; line-height: 1.3; text-align: right; }}

    div[data-testid="stMetric"] {{
        background: {CARD_BG}; border: 1px solid {BORDER}; border-top: 2px solid {ACCENT};
        border-radius: 8px; padding: 8px 12px 6px 12px;
    }}
    div[data-testid="stMetric"] label {{ color: #9aa4af !important; font-size: 0.68rem !important; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.25rem !important; font-weight: 700 !important; }}

    .section-title {{
        display: flex; align-items: center; gap: 8px; margin: 2px 0 8px 0;
        font-size: 0.92rem; font-weight: 700; color: #e8ebee; text-transform: uppercase; letter-spacing: 0.03em;
    }}
    .section-title .bar {{ width: 3px; height: 14px; border-radius: 2px; background: {ACCENT}; }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 10px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        height: 34px; padding: 0 14px; background: {CARD_BG}; border-radius: 8px 8px 0 0;
        font-size: 0.82rem;
    }}
    hr {{ border-color: {BORDER} !important; margin: 10px 0 !important; }}
    div[data-testid="stCaptionContainer"] {{ font-size: 0.72rem; }}
    .stDataFrame {{ font-size: 0.78rem; }}
</style>
""", unsafe_allow_html=True)


def section_title(icon: str, text: str):
    st.markdown(f'<div class="section-title"><span class="bar"></span>{icon} {text}</div>', unsafe_allow_html=True)


def chart_layout(fig, height, **kwargs):
    layout = dict(template="plotly_dark", paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                  height=height, margin=dict(t=28, l=8, r=8, b=8), font=dict(size=11))
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


# ---------- Chargement des données ----------
DATA_REPO_RAW = "https://raw.githubusercontent.com/kenzakab16/firerisk-dz-data/master/data/processed"


@st.cache_data(ttl=6 * 3600, show_spinner="Chargement des données...")
def load_data():
    """Historique figé (2000-2025, embarqué dans le dépôt) + année en cours
    récupérée depuis le dépôt de données (mis à jour quotidiennement par
    GitHub Actions), avec repli sur la copie locale si GitHub est
    injoignable. Cache 6 h : suit les mises à jour sans redéploiement."""
    import io
    import urllib.request

    wilayas = pd.read_csv("wilayas.csv")
    with open("wilayas_simplified.geojson", encoding="utf-8") as f:
        geojson = json.load(f)

    frozen = pd.read_parquet("ml_table_daily_wilaya_2000_2025.parquet")
    try:
        with urllib.request.urlopen(f"{DATA_REPO_RAW}/ml_table_current_year.parquet", timeout=30) as r:
            current = pd.read_parquet(io.BytesIO(r.read()))
        data_source = "données feu du jour via GitHub"
    except Exception:
        current = pd.read_parquet("ml_table_current_year.parquet")
        data_source = "copie locale (GitHub injoignable)"

    ml = pd.concat([frozen, current], ignore_index=True)
    ml["date"] = pd.to_datetime(ml["date"])
    ml = ml.drop_duplicates(subset=["wilaya_id", "date"], keep="last")
    ml["month"] = ml["date"].dt.month
    ml["year"] = ml["date"].dt.year
    return wilayas, geojson, ml, data_source


wilayas, geojson, ml, DATA_SOURCE = load_data()
LAST_DATE = ml["date"].max()
MOIS_FR = {1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Juin",
           7: "Juil", 8: "Août", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc"}
JOURS_FR = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Jeu", 4: "Ven", 5: "Sam", 6: "Dim"}


# ---------- Prévisions météo 7 jours (Open-Meteo, requête groupée) ----------
@st.cache_data(ttl=3600, show_spinner="Récupération des prévisions météo...")
def fetch_forecast(wilayas: pd.DataFrame):
    """Prévisions quotidiennes à 7 jours pour toutes les wilayas en une
    seule requête groupée. Retourne None si l'API est injoignable
    (le dashboard retombe alors sur la dernière météo du jeu de données)."""
    import urllib.request
    daily_vars = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
        "precipitation_sum", "rain_sum",
        "sunshine_duration", "shortwave_radiation_sum",
        "et0_fao_evapotranspiration", "surface_pressure_mean",
    ]
    lats = ",".join(f"{v:.4f}" for v in wilayas["centroid_lat"])
    lons = ",".join(f"{v:.4f}" for v in wilayas["centroid_lon"])
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        f"&daily={','.join(daily_vars)}"
        "&forecast_days=7&timezone=Africa%2FAlgiers"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, list) or len(data) != len(wilayas):
        return None
    rows = []
    for w, loc in zip(wilayas.itertuples(), data):
        d = loc["daily"]
        for i, day in enumerate(d["time"]):
            row = {"wilaya_id": w.wilaya_id, "date": day}
            for v in daily_vars:
                row[v] = d[v][i]
            rows.append(row)
    fc = pd.DataFrame(rows)
    fc["date"] = pd.to_datetime(fc["date"])
    return fc


@st.cache_data
def compute_climatology(ml: pd.DataFrame):
    """Climatologie mensuelle par wilaya : fréquence historique de feu et
    moyenne/écart-type des variables météo, base de l'anomalie."""
    g = ml.groupby(["wilaya_id", "month"])
    clim = g.agg(
        freq_fire=("fire_detected", "mean"),
        temp_mean=("temperature_2m_max", "mean"), temp_std=("temperature_2m_max", "std"),
        hum_mean=("relative_humidity_2m_mean", "mean"), hum_std=("relative_humidity_2m_mean", "std"),
        wind_mean=("wind_speed_10m_max", "mean"), wind_std=("wind_speed_10m_max", "std"),
    ).reset_index()
    for c in ["temp_std", "hum_std", "wind_std"]:
        clim[c] = clim[c].replace(0, 1).fillna(1)
    return clim


forecast = fetch_forecast(wilayas)


# ---------- Modèle prédictif (phase IA) ----------
@st.cache_resource
def load_model():
    """Charge le modèle entraîné (HistGradientBoosting, ère VIIRS 2015-2023,
    testé sur 2024-2026). Retourne (None, None) si l'artefact est absent."""
    import joblib
    try:
        model = joblib.load("model_fire_risk_v1.joblib")
        with open("model_fire_risk_v1_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        return model, meta
    except Exception:
        return None, None


MODEL_WEATHER_FEATURES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "precipitation_sum", "rain_sum",
    "sunshine_duration", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration", "surface_pressure_mean",
]


def predict_fire_proba(model, fc: pd.DataFrame, wilayas: pd.DataFrame):
    """Applique le modèle aux prévisions : mêmes features qu'à
    l'entraînement (météo prévue + saisonnalité + statiques wilaya)."""
    forest = wilayas[wilayas["is_forest_zone"]]
    df = fc.merge(forest[["wilaya_id", "wilaya_name", "centroid_lat", "centroid_lon", "area_km2"]],
                  on="wilaya_id", how="inner")
    X = df[MODEL_WEATHER_FEATURES + ["centroid_lat", "centroid_lon", "area_km2"]].copy()
    month = df["date"].dt.month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["wilaya_id"] = pd.Categorical(df["wilaya_id"], categories=sorted(forest["wilaya_id"]))
    df["fire_proba"] = model.predict_proba(X)[:, 1]
    return df[["wilaya_id", "wilaya_name", "date", "fire_proba"]]


ml_model, ml_meta = load_model()


# ---------- Calcul du score de risque par wilaya ----------
def score_to_level(s, is_forest):
    if not is_forest:
        return "Hors périmètre"
    if s >= 75:
        return "Très élevé"
    if s >= 50:
        return "Élevé"
    if s >= 25:
        return "Modéré"
    return "Faible"


@st.cache_data
def compute_risk(ml: pd.DataFrame, wilayas: pd.DataFrame, current_month: int,
                 current_weather: pd.DataFrame | None):
    """Score de risque par wilaya. `current_weather` (wilaya_id, temp,
    humidity, wind) vient des prévisions du jour si disponibles ; sinon
    on retombe sur la dernière ligne du jeu de données historique."""
    cw = current_weather.set_index("wilaya_id") if current_weather is not None else None
    rows = []
    for wid, g in ml.groupby("wilaya_id"):
        w = wilayas.loc[wilayas["wilaya_id"] == wid].iloc[0]
        if cw is not None and wid in cw.index:
            temp, hum, wind = cw.loc[wid, ["temp", "humidity", "wind"]]
        else:
            latest = g.loc[g["date"] == g["date"].max()].iloc[0]
            temp = latest["temperature_2m_max"]
            hum = latest["relative_humidity_2m_mean"]
            wind = latest["wind_speed_10m_max"]

        month_hist = g[g["month"] == current_month]
        freq_month = month_hist["fire_detected"].mean() if len(month_hist) else 0.0

        clim = month_hist[["temperature_2m_max", "relative_humidity_2m_mean", "wind_speed_10m_max"]].mean()
        clim_std = month_hist[["temperature_2m_max", "relative_humidity_2m_mean", "wind_speed_10m_max"]].std().replace(0, 1)

        temp_z = (temp - clim["temperature_2m_max"]) / clim_std["temperature_2m_max"]
        humidity_z = -(hum - clim["relative_humidity_2m_mean"]) / clim_std["relative_humidity_2m_mean"]
        wind_z = (wind - clim["wind_speed_10m_max"]) / clim_std["wind_speed_10m_max"]
        weather_anomaly = np.clip((temp_z + humidity_z + wind_z) / 3, -2, 2)

        rows.append({
            "wilaya_id": wid, "wilaya_name": w["wilaya_name"], "is_forest_zone": w["is_forest_zone"],
            "freq_month": freq_month, "weather_anomaly": weather_anomaly,
            "latest_temp": temp, "latest_humidity": hum, "latest_wind": wind,
            "total_fire_days": g["fire_detected"].sum(), "total_detections": g["nb_detections"].sum(),
            "total_frp": g["frp_total"].sum(),
        })
    df = pd.DataFrame(rows)

    forest = df[df["is_forest_zone"]].copy()
    if len(forest):
        freq_pct = forest["freq_month"].rank(pct=True)
        anomaly_pct = forest["weather_anomaly"].rank(pct=True)
        forest["risk_score"] = (0.55 * freq_pct + 0.45 * anomaly_pct) * 100
        df.loc[forest.index, "risk_score"] = forest["risk_score"]

    df["risk_level"] = df.apply(lambda r: score_to_level(r.get("risk_score", 0), r["is_forest_zone"]), axis=1)
    return df


if forecast is not None:
    today_fc = forecast[forecast["date"] == forecast["date"].min()]
    current_weather = today_fc.rename(columns={
        "temperature_2m_max": "temp", "relative_humidity_2m_mean": "humidity",
        "wind_speed_10m_max": "wind",
    })[["wilaya_id", "temp", "humidity", "wind"]]
    RISK_DATE = forecast["date"].min()
    WEATHER_SOURCE = "prévision du jour"
else:
    current_weather = None
    RISK_DATE = LAST_DATE
    WEATHER_SOURCE = f"météo du {LAST_DATE.strftime('%d/%m')} (API injoignable)"

risk_df = compute_risk(ml, wilayas, RISK_DATE.month, current_weather)
forest_risk = risk_df[risk_df["is_forest_zone"]]
clim = compute_climatology(ml)


# ---------- Backtesting : score/proba rétro-calculés sur météo réelle ----------
@st.cache_data
def backtest_recent(ml: pd.DataFrame, wilayas: pd.DataFrame, clim: pd.DataFrame, days: int, _model):
    """Rejoue le score heuristique et le modèle IA sur les `days` derniers
    jours COMPLETS en utilisant la météo réellement observée (pas la
    prévision telle qu'elle aurait été émise à l'époque — non conservée
    pour l'instant, voir forecast_log.csv pour la collecte à partir
    d'aujourd'hui). Les 2 derniers jours sont exclus : les détections
    FIRMS les plus récentes peuvent être encore incomplètes (latence
    satellite/traitement)."""
    forest_ids = set(wilayas.loc[wilayas["is_forest_zone"], "wilaya_id"])
    max_date = ml["date"].max()
    window_end = max_date - pd.Timedelta(days=2)
    window_start = window_end - pd.Timedelta(days=days - 1)

    df = ml[(ml["wilaya_id"].isin(forest_ids)) & (ml["date"] >= window_start) & (ml["date"] <= window_end)].copy()
    if "fire_data_coverage" in df.columns:
        df = df[df["fire_data_coverage"]]
    if df.empty:
        return df, window_start, window_end

    df = df.merge(clim, on=["wilaya_id", "month"], how="left")
    temp_z = (df["temperature_2m_max"] - df["temp_mean"]) / df["temp_std"]
    hum_z = -(df["relative_humidity_2m_mean"] - df["hum_mean"]) / df["hum_std"]
    wind_z = (df["wind_speed_10m_max"] - df["wind_mean"]) / df["wind_std"]
    df["weather_anomaly"] = np.clip((temp_z + hum_z + wind_z) / 3, -2, 2)
    df["freq_pct"] = df.groupby("date")["freq_fire"].rank(pct=True)
    df["anomaly_pct"] = df.groupby("date")["weather_anomaly"].rank(pct=True)
    df["heuristic_score"] = (0.55 * df["freq_pct"] + 0.45 * df["anomaly_pct"]) * 100
    df["heuristic_rank"] = df.groupby("date")["heuristic_score"].rank(ascending=False, method="min").astype(int)

    if _model is not None:
        X = df[MODEL_WEATHER_FEATURES + ["centroid_lat", "centroid_lon", "area_km2"]].copy()
        month = df["date"].dt.month
        X["month_sin"] = np.sin(2 * np.pi * month / 12)
        X["month_cos"] = np.cos(2 * np.pi * month / 12)
        X["wilaya_id"] = pd.Categorical(df["wilaya_id"], categories=sorted(forest_ids))
        df["ai_proba"] = _model.predict_proba(X)[:, 1] * 100
        df["ai_rank"] = df.groupby("date")["ai_proba"].rank(ascending=False, method="min").astype(int)
    else:
        df["ai_proba"] = np.nan
        df["ai_rank"] = np.nan

    return df, window_start, window_end


def render_public_view():
    """Vue grand public : langage courant, gros pictos, pas de jargon
    (pas de score/percentile/corrélation/AUC). Carte + météo du jour +
    conseils de prévention par wilaya."""
    n_high = int((forest_risk["risk_level"].isin(["Très élevé", "Élevé"])).sum())
    n_watch = int((forest_risk["risk_level"] == "Modéré").sum())
    n_calm = int((forest_risk["risk_level"] == "Faible").sum())

    if n_high > 0:
        banner_color, banner_icon = "#e5393522", "🔴"
        banner_text = f"{n_high} wilaya{'s' if n_high > 1 else ''} en alerte incendie aujourd'hui"
    elif n_watch > 0:
        banner_color, banner_icon = "#ffca2822", "🟡"
        banner_text = f"{n_watch} wilaya{'s' if n_watch > 1 else ''} sous surveillance, sans alerte majeure"
    else:
        banner_color, banner_icon = "#4caf5022", "🟢"
        banner_text = "Aucune wilaya en alerte incendie aujourd'hui"

    st.markdown(f"""
    <div style="background:{banner_color}; border:1px solid {BORDER}; border-radius:12px;
                padding:18px 24px; margin-bottom:16px; display:flex; align-items:center; gap:14px;">
        <span style="font-size:2rem;">{banner_icon}</span>
        <div>
            <div style="font-size:1.15rem; font-weight:800; color:#e8ebee;">{banner_text}</div>
            <div style="font-size:0.8rem; color:#9aa4af;">
                Situation au {RISK_DATE.strftime('%d/%m/%Y')} · {n_high} en alerte,
                {n_watch} sous surveillance, {n_calm} calmes, sur les 36 wilayas à couvert forestier suivies.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI clés du jour, comparés à hier (dernière météo réellement observée
    # dans le jeu de données, avant la prévision d'aujourd'hui)
    yesterday_risk_df = compute_risk(ml, wilayas, LAST_DATE.month, None)
    yesterday_forest = yesterday_risk_df[yesterday_risk_df["is_forest_zone"]]

    n_max_today = int((forest_risk["risk_level"] == "Très élevé").sum())
    n_max_yest = int((yesterday_forest["risk_level"] == "Très élevé").sum())

    at_risk_today = forest_risk[forest_risk["risk_level"].isin(["Élevé", "Très élevé"])]
    at_risk_yest = yesterday_forest[yesterday_forest["risk_level"].isin(["Élevé", "Très élevé"])]
    temp_today = (at_risk_today["latest_temp"].mean() if len(at_risk_today) else forest_risk["latest_temp"].mean())
    temp_yest = (at_risk_yest["latest_temp"].mean() if len(at_risk_yest) else yesterday_forest["latest_temp"].mean())

    score_today = forest_risk["risk_score"].mean()
    score_yest = yesterday_forest["risk_score"].mean()
    if score_today - score_yest > 4:
        trend_label, trend_delta = "En hausse", "Le risque augmente"
    elif score_today - score_yest < -4:
        trend_label, trend_delta = "En baisse", "Le risque diminue"
    else:
        trend_label, trend_delta = "Stable", "Situation stable"

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("🔴 Zones en alerte maximale", n_max_today,
               delta=f"{n_max_today - n_max_yest:+d} vs hier" if n_max_today != n_max_yest else "= hier",
               delta_color="inverse")
    kc2.metric("🌡️ Température dans les zones à risque", f"{temp_today:.0f} °C",
               delta=f"{temp_today - temp_yest:+.0f} °C vs hier", delta_color="inverse")
    kc3.metric("📊 Tendance du risque", trend_label, delta=trend_delta, delta_color="off")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    col_map, col_search = st.columns([6, 4])
    with col_map:
        section_title("🗺️", "Carte du risque incendie en Algérie")
        level_order = ["Faible", "Modéré", "Élevé", "Très élevé", "Hors périmètre"]
        fig_map_pub = px.choropleth_mapbox(
            risk_df, geojson=geojson, locations="wilaya_id", featureidkey="properties.wilaya_id",
            color="risk_level", category_orders={"risk_level": level_order},
            color_discrete_map=RISK_COLORS,
            hover_name="wilaya_name", hover_data={"wilaya_id": False, "risk_level": True},
            mapbox_style="carto-darkmatter", zoom=4.4, center={"lat": 32.5, "lon": 3.0}, opacity=0.8,
            labels={"risk_level": "Niveau"},
        )
        fig_map_pub.update_layout(height=480, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=CARD_BG,
                                   legend=dict(orientation="h", y=-0.02, font=dict(color="#e8ebee", size=11)))
        st.plotly_chart(fig_map_pub, use_container_width=True)
        st.caption("🟢 Faible · 🟡 Modéré · 🟠 Élevé · 🔴 Très élevé · ⚪ Zone non forestière (non suivie)")

    with col_search:
        section_title("📍", "Ma wilaya")
        all_names = sorted(wilayas["wilaya_name"])
        default_idx = all_names.index("Tizi Ouzou") if "Tizi Ouzou" in all_names else 0
        chosen = st.selectbox("Choisissez votre wilaya", all_names, index=default_idx, label_visibility="collapsed")
        sel_row = risk_df[risk_df["wilaya_name"] == chosen].iloc[0]
        title, advice = RISK_ADVICE[sel_row["risk_level"]]
        icon = RISK_ICONS[sel_row["risk_level"]]
        level_color = RISK_COLORS[sel_row["risk_level"]]

        st.markdown(f"""
        <div style="background:{CARD_BG}; border:1px solid {BORDER}; border-left:5px solid {level_color};
                    border-radius:10px; padding:16px 18px; margin-bottom:12px;">
            <div style="font-size:1.05rem; font-weight:800; color:#e8ebee;">{icon} {chosen} — {title}</div>
            <div style="font-size:0.85rem; color:#c3cad2; margin-top:6px;">{advice}</div>
        </div>
        """, unsafe_allow_html=True)

        if sel_row["is_forest_zone"]:
            t, h, w = sel_row["latest_temp"], sel_row["latest_humidity"], sel_row["latest_wind"]
            st.markdown(f"""
            <div style="background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px; padding:14px 18px;">
                <div style="font-size:0.85rem; color:#c3cad2; line-height:1.7;">
                    🌡️ Il va faire <b>{t:.0f}°C</b> ({qualify_temp(t)}) &nbsp;·&nbsp;
                    💧 {qualify_humidity(h)} (<b>{h:.0f}%</b>) &nbsp;·&nbsp;
                    💨 {qualify_wind(w)} (<b>{w:.0f} km/h</b>)
                </div>
            </div>
            """, unsafe_allow_html=True)

            if forecast is not None:
                fc_sel = forecast[forecast["wilaya_id"] == sel_row["wilaya_id"]].sort_values("date")
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                section_title("📅", "Les 7 prochains jours")
                clim_w = clim[clim["wilaya_id"] == sel_row["wilaya_id"]]
                fc_sel = fc_sel.copy()
                fc_sel["month"] = fc_sel["date"].dt.month
                fc_sel = fc_sel.merge(clim_w, on="month", how="left")
                temp_z = (fc_sel["temperature_2m_max"] - fc_sel["temp_mean"]) / fc_sel["temp_std"]
                hum_z = -(fc_sel["relative_humidity_2m_mean"] - fc_sel["hum_mean"]) / fc_sel["hum_std"]
                wind_z = (fc_sel["wind_speed_10m_max"] - fc_sel["wind_mean"]) / fc_sel["wind_std"]
                anomaly = np.clip((temp_z + hum_z + wind_z) / 3, -2, 2)
                freq_pct_all = fc_sel["freq_fire"].rank(pct=True)
                anomaly_pct_all = anomaly.rank(pct=True)
                day_score = (0.55 * freq_pct_all + 0.45 * anomaly_pct_all) * 100
                day_level = [score_to_level(s, True) for s in day_score]

                chips = "".join(
                    f"""<div style="display:inline-block; text-align:center; margin-right:8px; min-width:58px;">
                        <div style="font-size:0.68rem; color:#9aa4af;">{JOURS_FR[d.weekday()]}</div>
                        <div style="font-size:1.4rem;">{RISK_ICONS[lvl]}</div>
                        <div style="font-size:0.62rem; color:#9aa4af;">{d.strftime('%d/%m')}</div>
                    </div>"""
                    for d, lvl in zip(fc_sel["date"], day_level)
                )
                st.markdown(f'<div style="white-space:nowrap; overflow-x:auto; padding:6px 0;">{chips}</div>',
                             unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.caption(
        "Ces informations sont calculées automatiquement à partir de la météo et de l'historique des incendies "
        "détectés par satellite depuis 2000. Ce n'est **pas une prévision officielle** — en cas de doute ou "
        "de départ de feu, contactez la Protection civile : **14**."
    )


# ================= EN-TÊTE COMPACT (commun aux deux vues) =================
st.markdown(f"""
<div class="topbar">
    <h1>🔥 FireRisk DZ</h1>
    <div class="meta">Risque incendie de forêt en Algérie, par wilaya · mis à jour au {RISK_DATE.strftime('%d/%m/%Y')}</div>
</div>
""", unsafe_allow_html=True)

# ================= BASCULE VUE SIMPLE / VUE EXPERTE =================
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "public"

tcol1, tcol2, _ = st.columns([1.4, 1.4, 7.2])
with tcol1:
    if st.button("👤 Vue simple", use_container_width=True,
                 type="primary" if st.session_state.view_mode == "public" else "secondary"):
        st.session_state.view_mode = "public"
        st.rerun()
with tcol2:
    if st.button("🔬 Vue experte", use_container_width=True,
                 type="primary" if st.session_state.view_mode == "expert" else "secondary"):
        st.session_state.view_mode = "expert"
        st.rerun()

if st.session_state.view_mode == "public":
    render_public_view()
    st.stop()

st.caption(
    f"Vue experte · Risque évalué le {RISK_DATE.strftime('%d/%m/%Y')} ({WEATHER_SOURCE}) · "
    f"Historique feu à jour au {LAST_DATE.strftime('%d/%m/%Y')} · Open-Meteo (ERA5) + NASA FIRMS (MODIS/VIIRS), 2000 → aujourd'hui"
)

# ================= NAVIGATION PAR ONGLETS =================
tab_overview, tab_forecast, tab_backtest, tab_wilaya, tab_trends, tab_corr = st.tabs([
    "🗺️ Vue d'ensemble", "🔮 Prévisions & IA", "🎯 Backtesting", "📍 Détail par wilaya",
    "📉 Tendances 2000-2026", "🔗 Corrélations & classement",
])

# ---------- Onglet 1 : Vue d'ensemble ----------
with tab_overview:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("🌲 Wilayas suivies", int(risk_df["is_forest_zone"].sum()))
    k2.metric("🔴 Risque très élevé", int((forest_risk["risk_level"] == "Très élevé").sum()))
    k3.metric("🟠 Risque élevé", int((forest_risk["risk_level"] == "Élevé").sum()))
    k4.metric("🟡 Risque modéré", int((forest_risk["risk_level"] == "Modéré").sum()))
    k5.metric("🟢 Risque faible", int((forest_risk["risk_level"] == "Faible").sum()))
    k6.metric("🔥 Jours-feu cumulés 2000-2026", f'{int(forest_risk["total_fire_days"].sum()):,}')

    col_map, col_rank = st.columns([7, 3])
    with col_map:
        section_title("🗺️", "Carte du risque par wilaya")
        level_order = ["Faible", "Modéré", "Élevé", "Très élevé", "Hors périmètre"]
        fig_map = px.choropleth_mapbox(
            risk_df, geojson=geojson, locations="wilaya_id", featureidkey="properties.wilaya_id",
            color="risk_level", category_orders={"risk_level": level_order},
            color_discrete_map=RISK_COLORS,
            hover_name="wilaya_name",
            hover_data={"wilaya_id": False, "risk_level": True, "freq_month": ":.1%"},
            mapbox_style="carto-darkmatter", zoom=4.4, center={"lat": 32.5, "lon": 3.0}, opacity=0.78,
        )
        fig_map.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=CARD_BG,
                               legend=dict(orientation="h", y=-0.02, font=dict(color="#e8ebee", size=10)))
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            f"Score = fréquence historique de feu du mois (55%) + anomalie météo du jour vs climatologie "
            f"(45%, température↑/humidité↓/vent↑). Météo : {WEATHER_SOURCE}. Indicateur heuristique."
        )

    with col_rank:
        section_title("🏆", "Classement du jour")
        rank_tbl = forest_risk.sort_values("risk_score", ascending=False)[
            ["wilaya_name", "risk_level", "risk_score"]
        ].copy()
        rank_tbl["risk_score"] = rank_tbl["risk_score"].round(0).astype(int)
        rank_tbl.columns = ["Wilaya", "Niveau", "Score"]
        st.dataframe(
            rank_tbl, use_container_width=True, hide_index=True, height=560,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            },
        )

# ---------- Onglet 2 : Prévisions & IA ----------
with tab_forecast:
    if forecast is not None:
        col_h, col_ai = st.columns(2)

        fc = forecast.copy()
        fc["month"] = fc["date"].dt.month
        fc = fc.merge(clim, on=["wilaya_id", "month"], how="left")
        fc = fc.merge(wilayas[["wilaya_id", "wilaya_name", "is_forest_zone"]], on="wilaya_id")
        fc = fc[fc["is_forest_zone"]].copy()

        temp_z = (fc["temperature_2m_max"] - fc["temp_mean"]) / fc["temp_std"]
        hum_z = -(fc["relative_humidity_2m_mean"] - fc["hum_mean"]) / fc["hum_std"]
        wind_z = (fc["wind_speed_10m_max"] - fc["wind_mean"]) / fc["wind_std"]
        fc["weather_anomaly"] = np.clip((temp_z + hum_z + wind_z) / 3, -2, 2)
        fc["freq_pct"] = fc.groupby("date")["freq_fire"].rank(pct=True)
        fc["anomaly_pct"] = fc.groupby("date")["weather_anomaly"].rank(pct=True)
        fc["risk_score"] = (0.55 * fc["freq_pct"] + 0.45 * fc["anomaly_pct"]) * 100

        outlook = fc.pivot_table(index="wilaya_name", columns="date", values="risk_score")
        outlook = outlook.loc[outlook.mean(axis=1).sort_values(ascending=False).index]
        day_labels = [f"{JOURS_FR[d.weekday()]} {d.strftime('%d/%m')}" for d in outlook.columns]

        with col_h:
            section_title("🔮", "Score heuristique à 7 jours")
            fig_outlook = go.Figure(go.Heatmap(
                z=outlook.values, x=day_labels, y=outlook.index,
                colorscale=[[0, "#2e7d32"], [0.25, "#4caf50"], [0.5, "#ffca28"], [0.75, "#ff9800"], [1, "#e53935"]],
                zmin=0, zmax=100, colorbar=dict(title="Score", thickness=12),
                hovertemplate="%{y} — %{x} : score %{z:.0f}/100<extra></extra>",
            ))
            chart_layout(fig_outlook, 720, yaxis=dict(autorange="reversed", tickfont=dict(size=10)))
            st.plotly_chart(fig_outlook, use_container_width=True)
            top3 = outlook.mean(axis=1).head(3)
            st.caption(f"À surveiller cette semaine : {', '.join(top3.index)}.")

        with col_ai:
            if ml_model is not None:
                section_title("🤖", "Probabilité IA à 7 jours")
                preds = predict_fire_proba(ml_model, forecast, wilayas)
                proba_matrix = preds.pivot_table(index="wilaya_name", columns="date", values="fire_proba") * 100
                proba_matrix = proba_matrix.loc[proba_matrix.mean(axis=1).sort_values(ascending=False).index]
                day_labels_ml = [f"{JOURS_FR[d.weekday()]} {d.strftime('%d/%m')}" for d in proba_matrix.columns]

                fig_ml = go.Figure(go.Heatmap(
                    z=proba_matrix.values, x=day_labels_ml, y=proba_matrix.index,
                    colorscale="YlOrRd", zmin=0, zmax=60,
                    colorbar=dict(title="P(feu) %", thickness=12),
                    text=np.round(proba_matrix.values).astype(int), texttemplate="%{text}",
                    textfont=dict(size=8),
                    hovertemplate="%{y} — %{x} : %{z:.0f}%% de probabilité<extra></extra>",
                ))
                chart_layout(fig_ml, 720, yaxis=dict(autorange="reversed", tickfont=dict(size=10)))
                st.plotly_chart(fig_ml, use_container_width=True)
                m = ml_meta["metrics"]
                top3_ml = proba_matrix.mean(axis=1).head(3)
                st.caption(
                    f"Gradient boosting, entraîné 2015-2023, testé 2024-2026 (ROC-AUC {m['roc_auc_test']:.2f}, "
                    f"taux de base {m['base_rate_test']*100:.0f}%). À risque : {', '.join(top3_ml.index)}."
                )
            else:
                st.info("Modèle prédictif non trouvé (model_fire_risk_v1.joblib).")
    else:
        st.info("⚠️ API de prévision Open-Meteo injoignable — réessayez au prochain rechargement.")

# ---------- Onglet 3 : Backtesting ----------
with tab_backtest:
    days_window = st.radio("Fenêtre", [7, 14, 30], index=1, horizontal=True, label_visibility="collapsed",
                            format_func=lambda d: f"{d} derniers jours")
    bt, w_start, w_end = backtest_recent(ml, wilayas, clim, days_window, ml_model)

    if bt.empty:
        st.info("Pas assez de données récentes pour la rétro-simulation.")
    else:
        fire_rows = bt[bt["fire_detected"] == 1]
        n_fire_days = len(fire_rows)
        TOP_K = 8

        hit_heuristic = (fire_rows["heuristic_rank"] <= TOP_K).mean() * 100 if n_fire_days else np.nan
        hit_ai = (fire_rows["ai_rank"] <= TOP_K).mean() * 100 if n_fire_days and ml_model is not None else np.nan

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📅 Fenêtre analysée", f"{w_start.strftime('%d/%m')} → {w_end.strftime('%d/%m')}")
        k2.metric("🔥 Journées-wilaya en feu", n_fire_days)
        k3.metric(f"🎯 Détection top {TOP_K} (heuristique)", f"{hit_heuristic:.0f} %" if n_fire_days else "—")
        k4.metric(f"🤖 Détection top {TOP_K} (IA)", f"{hit_ai:.0f} %" if n_fire_days and ml_model is not None else "—")

        col_calib, col_hits = st.columns([1, 1])

        with col_calib:
            section_title("📐", "Calibration IA sur la fenêtre récente")
            if ml_model is not None and n_fire_days >= 3:
                calib = bt.copy()
                calib["bin"] = pd.qcut(calib["ai_proba"], q=min(6, calib["ai_proba"].nunique()),
                                        duplicates="drop")
                calib_g = calib.groupby("bin", observed=True).agg(
                    pred=("ai_proba", "mean"), obs=("fire_detected", "mean"), n=("fire_detected", "size"),
                ).reset_index()
                fig_calib = go.Figure()
                fig_calib.add_trace(go.Scatter(x=calib_g["pred"], y=calib_g["obs"] * 100, mode="lines+markers",
                                                name="Observé", line=dict(color=ACCENT, width=2),
                                                marker=dict(size=8)))
                fig_calib.add_trace(go.Scatter(x=[0, calib_g["pred"].max()], y=[0, calib_g["pred"].max()],
                                                mode="lines", name="Calibration parfaite",
                                                line=dict(color="#4fc3f7", width=1, dash="dot")))
                chart_layout(fig_calib, 340,
                             xaxis=dict(title="Probabilité prédite (%)"), yaxis=dict(title="Taux de feu observé (%)"),
                             legend=dict(orientation="h", y=1.15, font=dict(size=10)))
                st.plotly_chart(fig_calib, use_container_width=True)
                st.caption("Proche de la diagonale = modèle bien calibré sur la période récente.")
            else:
                st.info("Trop peu de journées-feu récentes pour une courbe de calibration fiable.")

        with col_hits:
            section_title("✅", "Feux détectés vs rang prédit")
            if n_fire_days:
                hits_tbl = fire_rows.sort_values("date", ascending=False)[
                    ["date", "wilaya_name", "heuristic_rank", "ai_rank", "nb_detections"]
                ].copy()
                hits_tbl["date"] = hits_tbl["date"].dt.strftime("%d/%m")
                hits_tbl.columns = ["Date", "Wilaya", "Rang heuristique", "Rang IA", "Détections"]
                st.dataframe(
                    hits_tbl, use_container_width=True, hide_index=True, height=340,
                    column_config={
                        "Rang heuristique": st.column_config.NumberColumn(help=f"1 = risque le plus élevé ce jour-là (sur 36). ≤{TOP_K} = repéré."),
                        "Rang IA": st.column_config.NumberColumn(help=f"1 = probabilité la plus élevée ce jour-là (sur 36). ≤{TOP_K} = repéré."),
                    },
                )
                st.caption(f"Rang ≤ {TOP_K} = la wilaya était dans la liste de surveillance ce jour-là.")
            else:
                st.info("Aucun feu de végétation détecté sur cette fenêtre.")

        st.warning(
            "⚠️ **Méthodologie** : cette rétro-simulation applique le score heuristique et le modèle IA à la "
            "météo **réellement observée** des derniers jours, pas à la prévision telle qu'elle aurait été "
            "émise à l'époque (non archivée avant aujourd'hui — voir ci-dessous). C'est un test honnête de la "
            "capacité de *discrimination* du modèle (sait-il reconnaître les journées à risque une fois la "
            "météo connue ?), mais pas encore un test de la qualité des *prévisions* à 7 jours elles-mêmes. "
            "Les prévisions quotidiennes sont désormais archivées automatiquement (voir `forecast_log.csv` dans "
            "[firerisk-dz-data](https://github.com/kenzakab16/firerisk-dz-data)) : un vrai backtesting "
            "prévision-vs-réel sera possible dans quelques semaines."
        )

# ---------- Onglet 4 : Détail par wilaya ----------
with tab_wilaya:
    forest_names = sorted(risk_df.loc[risk_df["is_forest_zone"], "wilaya_name"])
    default_idx = forest_names.index("Tizi Ouzou") if "Tizi Ouzou" in forest_names else 0
    sel_col, *metric_cols = st.columns([2, 1, 1, 1, 1, 1])
    with sel_col:
        selected_name = st.selectbox("Wilaya (zone forestière)", forest_names, index=default_idx, label_visibility="collapsed")

    sel = risk_df[risk_df["wilaya_name"] == selected_name].iloc[0]
    sel_ml = ml[ml["wilaya_id"] == sel["wilaya_id"]].copy()

    icon = RISK_ICONS[sel["risk_level"]]
    meteo_suffix = "prévu" if forecast is not None else "connu"
    metric_cols[0].metric(f"🌡️ Temp. max {meteo_suffix}", f'{sel["latest_temp"]:.1f} °C')
    metric_cols[1].metric(f"💧 Humidité {meteo_suffix}e", f'{sel["latest_humidity"]:.0f} %')
    metric_cols[2].metric(f"💨 Vent {meteo_suffix}", f'{sel["latest_wind"]:.0f} km/h')
    metric_cols[3].metric("🔥 Jours-feu (2000-2026)", int(sel["total_fire_days"]))
    metric_cols[4].metric(f"{icon} Risque", sel["risk_level"])

    c1, c2, c3 = st.columns(3)
    with c1:
        section_title("📅", "Saisonnalité des feux")
        monthly = sel_ml.groupby("month")["fire_detected"].sum().reindex(range(1, 13), fill_value=0)
        fig_month = go.Figure(go.Bar(
            x=[MOIS_FR[m] for m in monthly.index], y=monthly.values,
            marker=dict(color=monthly.values, colorscale="OrRd"),
        ))
        chart_layout(fig_month, 320, yaxis=dict(title="Jours avec feu"))
        st.plotly_chart(fig_month, use_container_width=True)

    with c2:
        section_title("📈", "Température vs jours de feu")
        monthly_temp = sel_ml.groupby("month")["temperature_2m_max"].mean().reindex(range(1, 13))
        fig_tf = go.Figure()
        fig_tf.add_trace(go.Bar(x=[MOIS_FR[m] for m in monthly.index], y=monthly.values,
                                 name="Jours-feu", marker=dict(color=ACCENT), opacity=0.7))
        fig_tf.add_trace(go.Scatter(x=[MOIS_FR[m] for m in monthly_temp.index], y=monthly_temp.values,
                                     name="Temp. max (°C)", yaxis="y2", line=dict(color="#ff5c5c", width=3)))
        chart_layout(fig_tf, 320, yaxis=dict(title="Jours-feu"),
                     yaxis2=dict(title="Temp. (°C)", overlaying="y", side="right"),
                     legend=dict(orientation="h", y=1.18, font=dict(size=10)))
        st.plotly_chart(fig_tf, use_container_width=True)

    with c3:
        section_title("🕰️", "Évolution annuelle")
        sel_covered = sel_ml[sel_ml["fire_data_coverage"]] if "fire_data_coverage" in sel_ml.columns else sel_ml
        sel_annual = sel_covered.groupby("year")["fire_detected"].sum()
        sel_annual = sel_annual[sel_annual.index >= 2001]
        fig_sel_annual = go.Figure(go.Bar(
            x=sel_annual.index, y=sel_annual.values,
            marker=dict(color=sel_annual.values, colorscale="OrRd"),
        ))
        fig_sel_annual.add_vline(x=2014.5, line=dict(color="#4fc3f7", width=1, dash="dot"))
        chart_layout(fig_sel_annual, 320, yaxis=dict(title="Jours-feu"), xaxis=dict(dtick=4))
        st.plotly_chart(fig_sel_annual, use_container_width=True)
    st.caption("Trait bleu : bascule capteur MODIS → VIIRS (2015), ~5× plus sensible — comptages non comparables avant/après.")

# ---------- Onglet 5 : Tendances nationales ----------
with tab_trends:
    forest_ids = risk_df.loc[risk_df["is_forest_zone"], "wilaya_id"]
    forest_hist = ml[ml["wilaya_id"].isin(forest_ids)].copy()
    if "fire_data_coverage" in forest_hist.columns:
        forest_hist = forest_hist[forest_hist["fire_data_coverage"]]
    forest_hist = forest_hist[forest_hist["year"] >= 2001]
    current_year = LAST_DATE.year

    annual_fire = forest_hist.groupby("year")["fire_detected"].sum()
    summer = forest_hist[forest_hist["month"].isin([6, 7, 8, 9])]
    annual_summer_temp = summer.groupby("year")["temperature_2m_max"].mean()

    fig_trend = go.Figure()
    bar_colors = ["#8a5a3b" if y < 2015 else ACCENT for y in annual_fire.index]
    fig_trend.add_trace(go.Bar(x=annual_fire.index, y=annual_fire.values, name="Jours-feu (cumul wilayas)",
                                marker=dict(color=bar_colors), opacity=0.85))
    fig_trend.add_trace(go.Scatter(x=annual_summer_temp.index, y=annual_summer_temp.values,
                                    name="Temp. max moy. été (°C)", yaxis="y2",
                                    line=dict(color="#ff5c5c", width=3)))
    z = np.polyfit(annual_summer_temp.index, annual_summer_temp.values, 1)
    fig_trend.add_trace(go.Scatter(x=annual_summer_temp.index, y=np.polyval(z, annual_summer_temp.index),
                                    name=f"Tendance ({z[0]*10:+.2f} °C/décennie)", yaxis="y2",
                                    line=dict(color="#ff5c5c", width=1.5, dash="dash")))
    fig_trend.add_vline(x=2014.5, line=dict(color="#4fc3f7", width=1.5, dash="dot"))
    fig_trend.add_annotation(x=2014.5, y=1.1, yref="paper", showarrow=False,
                              text="MODIS → VIIRS (×5 plus sensible)", font=dict(size=10, color="#4fc3f7"))
    for yr, label, ay in [(2021, "Kabylie", -60), (2023, "Vague nationale", -25)]:
        if yr in annual_fire.index:
            fig_trend.add_annotation(x=yr, y=annual_fire[yr], text=f"{yr} — {label}",
                                      showarrow=True, arrowhead=2, ax=40, ay=ay,
                                      font=dict(size=9, color="#ffb347"), arrowcolor="#ffb347")
    chart_layout(fig_trend, 380,
                 yaxis=dict(title="Jours-feu/an"),
                 yaxis2=dict(title="Temp. été (°C)", overlaying="y", side="right", showgrid=False),
                 legend=dict(orientation="h", y=1.15, font=dict(size=10)), xaxis=dict(dtick=2))

    col_trend, col_heat = st.columns([1, 1])
    with col_trend:
        section_title("📉", "Jours-feu vs température estivale")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption(f"⚠️ Comptages avant/après 2015 non comparables (changement de capteur). "
                   f"Tendance température homogène (ERA5) : {z[0]*10:+.2f} °C/décennie.")

    with col_heat:
        section_title("🗓️", "Heatmap saisonnière (année × mois)")
        heat = forest_hist.pivot_table(index="year", columns="month", values="fire_detected", aggfunc="sum").fillna(0)
        heat = heat.reindex(columns=range(1, 13), fill_value=0)
        fig_heat = go.Figure(go.Heatmap(
            z=heat.values, x=[MOIS_FR[m] for m in heat.columns], y=heat.index,
            colorscale="OrRd", colorbar=dict(title="Jours-feu", thickness=12),
            hovertemplate="%{y} — %{x} : %{z} jours-feu<extra></extra>",
        ))
        chart_layout(fig_heat, 380, yaxis=dict(dtick=3, autorange="reversed"))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Saison des feux (juin-octobre, pic juillet-août) et étés 2021/2023 nettement visibles.")

# ---------- Onglet 6 : Corrélations & classement ----------
with tab_corr:
    col_corr, col_table = st.columns([1, 2])

    with col_corr:
        section_title("🔗", "Corrélations météo ↔ incendies")
        corr_vars = {
            "temperature_2m_max": "🌡️ Température max",
            "relative_humidity_2m_mean": "💧 Humidité relative",
            "wind_speed_10m_max": "💨 Vent max",
            "precipitation_sum": "🌧️ Précipitations",
            "et0_fao_evapotranspiration": "🌾 Évapotranspiration",
        }
        forest_ml = ml[ml["wilaya_id"].isin(risk_df.loc[risk_df["is_forest_zone"], "wilaya_id"])]
        corrs = {label: forest_ml[col].corr(forest_ml["fire_detected"]) for col, label in corr_vars.items()}
        corr_series = pd.Series(corrs).sort_values()
        fig_corr = go.Figure(go.Bar(
            x=corr_series.values, y=corr_series.index, orientation="h",
            marker=dict(color=["#e53935" if v > 0 else "#4fc3f7" for v in corr_series.values]),
            text=[f"{v:+.3f}" for v in corr_series.values], textposition="outside",
        ))
        chart_layout(fig_corr, 460, margin=dict(t=20, l=8, r=70, b=8),
                     xaxis=dict(title="Corrélation avec fire_detected", range=[-0.32, 0.42]))
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Pearson, 36 wilayas forestières, 2000-2026 (~348k jours×wilaya). Rouge = corrélation positive.")

    with col_table:
        section_title("🏆", "Classement complet des wilayas")
        table = forest_risk.sort_values("risk_score", ascending=False)[
            ["wilaya_name", "risk_level", "risk_score", "freq_month", "total_fire_days", "latest_temp", "latest_humidity", "latest_wind"]
        ].copy()
        table["risk_score"] = table["risk_score"].round(1)
        table["freq_month"] = (table["freq_month"] * 100).round(1)
        table.columns = ["Wilaya", "Niveau", "Score", "Fréq. mois (%)", "Jours-feu",
                          "Temp. (°C)", "Humidité (%)", "Vent (km/h)"]
        st.dataframe(table, use_container_width=True, hide_index=True, height=460)

st.caption(
    "FireRisk DZ — phase tableau de bord + IA. Score heuristique et probabilité du modèle prédictif sont deux "
    "indicateurs complémentaires, pas des prévisions officielles. Sources et méthodologie détaillées : "
    "[dépôt de données](https://github.com/kenzakab16/firerisk-dz-data)."
)
