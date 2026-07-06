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

st.set_page_config(page_title="FireRisk DZ — Tableau de bord national", page_icon="🔥", layout="wide")

# ---------- Palette & styles ----------
ACCENT = "#ff7a45"
CARD_BG = "#171c22"
BORDER = "rgba(255,255,255,0.06)"
RISK_COLORS = {
    "Faible": "#4caf50",
    "Modéré": "#ffca28",
    "Élevé": "#ff9800",
    "Très élevé": "#e53935",
    "Hors périmètre": "#3a3f47",
}
RISK_ICONS = {"Faible": "🟢", "Modéré": "🟡", "Élevé": "🟠", "Très élevé": "🔴", "Hors périmètre": "⚪"}

st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }}
    .hero {{
        background: linear-gradient(120deg, rgba(255,122,69,0.16), rgba(23,28,34,0.0) 70%), {CARD_BG};
        border: 1px solid {BORDER}; border-radius: 16px; padding: 26px 30px; margin-bottom: 24px;
    }}
    .hero h1 {{ font-size: 1.8rem; margin: 0 0 6px 0; font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #ffb347);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .hero p {{ color: #9aa4af; font-size: 0.85rem; margin: 0; }}
    div[data-testid="stMetric"] {{
        background: {CARD_BG}; border: 1px solid {BORDER}; border-top: 3px solid {ACCENT};
        border-radius: 12px; padding: 14px 16px 10px 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }}
    div[data-testid="stMetric"] label {{ color: #9aa4af !important; font-size: 0.75rem !important; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.5rem !important; font-weight: 700 !important; }}
    .section-title {{ display: flex; align-items: center; gap: 10px; margin: 6px 0 12px 0;
        font-size: 1.1rem; font-weight: 700; color: #e8ebee; }}
    .section-title .bar {{ width: 4px; height: 18px; border-radius: 2px; background: {ACCENT}; }}
    hr {{ border-color: {BORDER} !important; margin: 26px 0 !important; }}
</style>
""", unsafe_allow_html=True)


def section_title(icon: str, text: str):
    st.markdown(f'<div class="section-title"><span class="bar"></span>{icon} {text}</div>', unsafe_allow_html=True)


# ---------- Chargement des données ----------
@st.cache_data
def load_data():
    wilayas = pd.read_csv("wilayas.csv")
    with open("wilayas_simplified.geojson", encoding="utf-8") as f:
        geojson = json.load(f)
    ml = pd.read_parquet("ml_table_daily_wilaya_2000_2026.parquet")
    ml["date"] = pd.to_datetime(ml["date"])
    ml["month"] = ml["date"].dt.month
    ml["year"] = ml["date"].dt.year
    return wilayas, geojson, ml


wilayas, geojson, ml = load_data()
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
    lats = ",".join(f"{v:.4f}" for v in wilayas["centroid_lat"])
    lons = ",".join(f"{v:.4f}" for v in wilayas["centroid_lon"])
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        "&daily=temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum"
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
            rows.append({
                "wilaya_id": w.wilaya_id, "date": day,
                "temperature_2m_max": d["temperature_2m_max"][i],
                "relative_humidity_2m_mean": d["relative_humidity_2m_mean"][i],
                "wind_speed_10m_max": d["wind_speed_10m_max"][i],
                "precipitation_sum": d["precipitation_sum"][i],
            })
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
    WEATHER_SOURCE = "prévision Open-Meteo du jour"
else:
    current_weather = None
    RISK_DATE = LAST_DATE
    WEATHER_SOURCE = f"dernière météo du jeu de données ({LAST_DATE.strftime('%d/%m/%Y')}) — API de prévision injoignable"

risk_df = compute_risk(ml, wilayas, RISK_DATE.month, current_weather)

# ---------- En-tête ----------
st.markdown(f"""
<div class="hero">
    <h1>🔥 FireRisk DZ — Risque incendie de forêt par wilaya</h1>
    <p>
        Historique 2000–{LAST_DATE.year} + prévisions 7 jours &nbsp;·&nbsp;
        Météo : Open-Meteo (ERA5 + prévisions) &nbsp;·&nbsp; Incendies : NASA FIRMS (MODIS + VIIRS) &nbsp;·&nbsp;
        Risque évalué au {RISK_DATE.strftime('%d/%m/%Y')} ({WEATHER_SOURCE})
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- KPIs nationaux ----------
forest_risk = risk_df[risk_df["is_forest_zone"]]
c1, c2, c3, c4 = st.columns(4)
c1.metric("🌲 Wilayas forestières suivies", int(risk_df["is_forest_zone"].sum()))
c2.metric("🔴 Risque très élevé", int((forest_risk["risk_level"] == "Très élevé").sum()))
c3.metric("🟠 Risque élevé", int((forest_risk["risk_level"] == "Élevé").sum()))
c4.metric("🔥 Total jours-feu détectés (2000-2026, zone forestière)", f'{int(forest_risk["total_fire_days"].sum()):,}')

st.divider()

# ---------- Carte de risque ----------
section_title("🗺️", "Carte du risque incendie par wilaya")
level_order = ["Faible", "Modéré", "Élevé", "Très élevé", "Hors périmètre"]
fig_map = px.choropleth_mapbox(
    risk_df, geojson=geojson, locations="wilaya_id", featureidkey="properties.wilaya_id",
    color="risk_level", category_orders={"risk_level": level_order},
    color_discrete_map=RISK_COLORS,
    hover_name="wilaya_name",
    hover_data={"wilaya_id": False, "risk_level": True, "freq_month": ":.1%"},
    mapbox_style="carto-darkmatter", zoom=4.2, center={"lat": 32.5, "lon": 3.0}, opacity=0.75,
)
fig_map.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=CARD_BG,
                       legend=dict(orientation="h", y=-0.02, font=dict(color="#e8ebee")))
st.plotly_chart(fig_map, use_container_width=True)
st.caption(
    f"Score de risque = fréquence historique de feu pour le mois en cours (55%) + anomalie de la météo "
    f"du jour vs climatologie du mois (45%, température↑/humidité↓/vent↑). Météo du jour : {WEATHER_SOURCE}. "
    "Indicateur heuristique — un modèle prédictif entraîné (phase IA) viendra en complément."
)

st.divider()

# ---------- Perspectives 7 jours ----------
if forecast is not None:
    section_title("🔮", "Perspectives à 7 jours (prévisions Open-Meteo)")
    clim = compute_climatology(ml)
    fc = forecast.copy()
    fc["month"] = fc["date"].dt.month
    fc = fc.merge(clim, on=["wilaya_id", "month"], how="left")
    fc = fc.merge(wilayas[["wilaya_id", "wilaya_name", "is_forest_zone"]], on="wilaya_id")
    fc = fc[fc["is_forest_zone"]].copy()

    temp_z = (fc["temperature_2m_max"] - fc["temp_mean"]) / fc["temp_std"]
    hum_z = -(fc["relative_humidity_2m_mean"] - fc["hum_mean"]) / fc["hum_std"]
    wind_z = (fc["wind_speed_10m_max"] - fc["wind_mean"]) / fc["wind_std"]
    fc["weather_anomaly"] = np.clip((temp_z + hum_z + wind_z) / 3, -2, 2)

    # Score par jour : percentiles calculés jour par jour, comme pour la carte
    fc["freq_pct"] = fc.groupby("date")["freq_fire"].rank(pct=True)
    fc["anomaly_pct"] = fc.groupby("date")["weather_anomaly"].rank(pct=True)
    fc["risk_score"] = (0.55 * fc["freq_pct"] + 0.45 * fc["anomaly_pct"]) * 100

    outlook = fc.pivot_table(index="wilaya_name", columns="date", values="risk_score")
    outlook = outlook.loc[outlook.mean(axis=1).sort_values(ascending=False).index]
    day_labels = [f"{JOURS_FR[d.weekday()]} {d.strftime('%d/%m')}" for d in outlook.columns]

    fig_outlook = go.Figure(go.Heatmap(
        z=outlook.values, x=day_labels, y=outlook.index,
        colorscale=[[0, "#2e7d32"], [0.25, "#4caf50"], [0.5, "#ffca28"], [0.75, "#ff9800"], [1, "#e53935"]],
        zmin=0, zmax=100, colorbar=dict(title="Score"),
        hovertemplate="%{y} — %{x} : score %{z:.0f}/100<extra></extra>",
    ))
    fig_outlook.update_layout(
        template="plotly_dark", paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        height=760, margin=dict(t=20, l=10, r=10, b=10),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_outlook, use_container_width=True)

    top3 = outlook.mean(axis=1).head(3)
    st.caption(
        f"Score de risque quotidien (0-100) par wilaya forestière, calculé sur les prévisions météo à 7 jours "
        f"croisées avec la climatologie 2000-2026. Wilayas classées par risque moyen décroissant — "
        f"à surveiller cette semaine : {', '.join(top3.index)}. Prévisions rafraîchies toutes les heures."
    )
else:
    st.info("⚠️ API de prévision Open-Meteo injoignable — les perspectives à 7 jours seront affichées au prochain rechargement avec connexion.")

st.divider()

# ---------- Sélection d'une wilaya ----------
section_title("📍", "Détail par wilaya")
forest_names = sorted(risk_df.loc[risk_df["is_forest_zone"], "wilaya_name"])
default_idx = forest_names.index("Tizi Ouzou") if "Tizi Ouzou" in forest_names else 0
selected_name = st.selectbox("Choisir une wilaya (zone forestière)", forest_names, index=default_idx)

sel = risk_df[risk_df["wilaya_name"] == selected_name].iloc[0]
sel_ml = ml[ml["wilaya_id"] == sel["wilaya_id"]].copy()

icon = RISK_ICONS[sel["risk_level"]]
meteo_suffix = "aujourd'hui (prévision)" if forecast is not None else "connue"
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric(f"🌡️ Temp. max {meteo_suffix}", f'{sel["latest_temp"]:.1f} °C')
d2.metric(f"💧 Humidité {meteo_suffix}", f'{sel["latest_humidity"]:.0f} %')
d3.metric(f"💨 Vent max {meteo_suffix}", f'{sel["latest_wind"]:.0f} km/h')
d4.metric("🔥 Jours avec feu (2000-2026)", int(sel["total_fire_days"]))
d5.metric(f"{icon} Niveau de risque actuel", sel["risk_level"])

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    section_title("📅", f"Saisonnalité des feux — {selected_name}")
    monthly = sel_ml.groupby("month")["fire_detected"].sum().reindex(range(1, 13), fill_value=0)
    fig_month = go.Figure(go.Bar(
        x=[MOIS_FR[m] for m in monthly.index], y=monthly.values,
        marker=dict(color=monthly.values, colorscale="OrRd"),
    ))
    fig_month.update_layout(template="plotly_dark", paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                             height=340, margin=dict(t=20, l=10, r=10, b=10),
                             yaxis=dict(title="Jours avec feu (cumul 2000-2026)"))
    st.plotly_chart(fig_month, use_container_width=True)

with col_b:
    section_title("📈", "Température moyenne vs jours de feu, par mois")
    monthly_temp = sel_ml.groupby("month")["temperature_2m_max"].mean().reindex(range(1, 13))
    fig_tf = go.Figure()
    fig_tf.add_trace(go.Bar(x=[MOIS_FR[m] for m in monthly.index], y=monthly.values,
                             name="Jours avec feu", marker=dict(color=ACCENT), opacity=0.7))
    fig_tf.add_trace(go.Scatter(x=[MOIS_FR[m] for m in monthly_temp.index], y=monthly_temp.values,
                                 name="Temp. max moy. (°C)", yaxis="y2", line=dict(color="#ff5c5c", width=3)))
    fig_tf.update_layout(template="plotly_dark", paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                          height=340, margin=dict(t=20, l=10, r=10, b=10),
                          yaxis=dict(title="Jours avec feu"), yaxis2=dict(title="Temp. (°C)", overlaying="y", side="right"),
                          legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_tf, use_container_width=True)

# Évolution annuelle pour la wilaya sélectionnée
section_title("🕰️", f"Évolution annuelle des feux — {selected_name}")
sel_covered = sel_ml[sel_ml["fire_data_coverage"]] if "fire_data_coverage" in sel_ml.columns else sel_ml
sel_annual = sel_covered.groupby("year")["fire_detected"].sum()
sel_annual = sel_annual[sel_annual.index >= 2001]  # 2000 partiel (couverture satellite depuis nov.)
fig_sel_annual = go.Figure(go.Bar(
    x=sel_annual.index, y=sel_annual.values,
    marker=dict(color=sel_annual.values, colorscale="OrRd"),
))
fig_sel_annual.add_vline(x=2014.5, line=dict(color="#4fc3f7", width=1, dash="dot"),
                          annotation_text="MODIS → VIIRS", annotation_position="top",
                          annotation=dict(font=dict(size=10, color="#4fc3f7")))
fig_sel_annual.update_layout(template="plotly_dark", paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                              height=300, margin=dict(t=30, l=10, r=10, b=10),
                              yaxis=dict(title="Jours avec feu"), xaxis=dict(dtick=2))
st.plotly_chart(fig_sel_annual, use_container_width=True)

st.divider()

# ---------- Tendances nationales 2001-2026 ----------
section_title("📉", "Tendances nationales (zone forestière, 2001-2026)")
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
                                name="Temp. max moy. été juin-sept (°C)", yaxis="y2",
                                line=dict(color="#ff5c5c", width=3)))
z = np.polyfit(annual_summer_temp.index, annual_summer_temp.values, 1)
fig_trend.add_trace(go.Scatter(x=annual_summer_temp.index, y=np.polyval(z, annual_summer_temp.index),
                                name=f"Tendance temp. ({z[0]*10:+.2f} °C/décennie)", yaxis="y2",
                                line=dict(color="#ff5c5c", width=1.5, dash="dash")))
fig_trend.add_vline(x=2014.5, line=dict(color="#4fc3f7", width=1.5, dash="dot"))
fig_trend.add_annotation(x=2014.5, y=1.06, yref="paper", showarrow=False,
                          text="Changement de capteur MODIS → VIIRS (×5 plus sensible)",
                          font=dict(size=11, color="#4fc3f7"))
for yr, label, ay in [(2021, "Incendies meurtriers de Kabylie", -70), (2023, "Vague nationale juillet", -30)]:
    if yr in annual_fire.index:
        fig_trend.add_annotation(x=yr, y=annual_fire[yr], text=f"{yr} — {label}",
                                  showarrow=True, arrowhead=2, ax=40, ay=ay,
                                  font=dict(size=10, color="#ffb347"), arrowcolor="#ffb347")
fig_trend.update_layout(
    template="plotly_dark", paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    height=440, margin=dict(t=50, l=10, r=10, b=10),
    yaxis=dict(title="Jours-feu par an (cumul 36 wilayas)"),
    yaxis2=dict(title="Temp. max moy. été (°C)", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", y=1.14),
    xaxis=dict(dtick=2),
)
st.plotly_chart(fig_trend, use_container_width=True)
st.caption(
    f"⚠️ Les comptages avant/après 2015 ne sont pas directement comparables : MODIS (résolution 1 km, "
    f"2001-2014, barres brunes) détecte beaucoup moins de petits feux que VIIRS (375 m, 2015+, barres orange). "
    f"La tendance de température estivale, elle, est homogène sur toute la période (ERA5) : "
    f"{z[0]*10:+.2f} °C/décennie sur la zone forestière. {current_year} est une année partielle "
    f"(données jusqu'au {LAST_DATE.strftime('%d/%m')})."
)

# Heatmap année x mois
section_title("🗓️", "Heatmap saisonnière : jours-feu par mois et par année")
heat = forest_hist.pivot_table(index="year", columns="month", values="fire_detected", aggfunc="sum").fillna(0)
heat = heat.reindex(columns=range(1, 13), fill_value=0)
fig_heat = go.Figure(go.Heatmap(
    z=heat.values, x=[MOIS_FR[m] for m in heat.columns], y=heat.index,
    colorscale="OrRd", colorbar=dict(title="Jours-feu"),
    hovertemplate="%{y} — %{x} : %{z} jours-feu<extra></extra>",
))
fig_heat.update_layout(template="plotly_dark", paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                        height=560, margin=dict(t=20, l=10, r=10, b=10),
                        yaxis=dict(dtick=2, autorange="reversed"))
st.plotly_chart(fig_heat, use_container_width=True)
st.caption(
    "Cumul des jours avec détection sur les 36 wilayas forestières. La saison des feux (juin-octobre, "
    "pic juillet-août) ressort nettement, ainsi que les étés exceptionnels 2021 et 2023. "
    "Même réserve que ci-dessus sur la comparaison avant/après 2015."
)

st.divider()

# ---------- Corrélations ----------
section_title("🔗", "Corrélations météo ↔ incendies (zone forestière, 2000-2026)")
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
fig_corr.update_layout(template="plotly_dark", paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                        height=320, margin=dict(t=20, l=10, r=60, b=10),
                        xaxis=dict(title="Coefficient de corrélation avec fire_detected", range=[-0.3, 0.3]))
st.plotly_chart(fig_corr, use_container_width=True)
st.caption(
    "Corrélation de Pearson entre chaque variable météo et la variable binaire fire_detected, "
    "calculée sur les 36 wilayas forestières, 2000-2026 (~348k jours x wilaya). "
    "Rouge = plus la variable est élevée, plus les feux sont fréquents ; bleu = l'inverse."
)

st.divider()

# ---------- Classement national ----------
section_title("🏆", "Classement des wilayas par risque actuel")
table = forest_risk.sort_values("risk_score", ascending=False)[
    ["wilaya_name", "risk_level", "risk_score", "freq_month", "total_fire_days", "latest_temp", "latest_humidity", "latest_wind"]
].copy()
table["risk_score"] = table["risk_score"].round(1)
table["freq_month"] = (table["freq_month"] * 100).round(1)
table.columns = ["Wilaya", "Niveau", "Score (0-100)", "Fréq. feu ce mois (%, hist.)", "Jours-feu (2000-2026)",
                  "Dernière temp. (°C)", "Dernière humidité (%)", "Dernier vent (km/h)"]
st.dataframe(table, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "FireRisk DZ — phase tableau de bord analytique. Score de risque basé sur l'historique et la météo la "
    "plus récente disponible dans le jeu de données (pas de prévision temps réel à ce stade). "
    "Prochaine étape : intégration de prévisions météo et modèle prédictif (Random Forest / XGBoost)."
)
