# FireRisk DZ — Tableau de bord national

Tableau de bord Streamlit d'analyse du risque incendie de forêt en Algérie, par wilaya, basé sur 26 ans de données réelles (2000 → aujourd'hui).

C'est la **phase 3** du projet FireRisk DZ. Les données sont produites par le pipeline ETL du dépôt [firerisk-dz-data](https://github.com/kenzakab16/firerisk-dz-data) (météo Open-Meteo/ERA5 + détections NASA FIRMS MODIS/VIIRS, filtrées des torchères industrielles).

## Fonctionnalités

- 🗺️ **Carte choroplèthe du risque par wilaya** — score combinant fréquence historique de feu pour le mois en cours (55%) et anomalie météo récente vs climatologie (45%) ; 4 niveaux (faible → très élevé), wilayas sahariennes hors périmètre
- 📍 **Détail par wilaya** — météo récente, saisonnalité des feux, température vs feux par mois, évolution annuelle 2001-2026
- 📉 **Tendances nationales** — jours-feu par an croisés avec la température estivale (tendance ~+0,9 °C/décennie sur la zone forestière), années exceptionnelles 2021/2023 annotées, distinction visuelle MODIS (2001-2014) / VIIRS (2015+, capteur ~5× plus sensible : comptages non directement comparables entre les deux ères)
- 🗓️ **Heatmap saisonnière** année × mois sur 26 ans
- 🔗 **Corrélations météo ↔ incendies** sur ~348k observations jour × wilaya
- 🏆 **Classement des wilayas** par score de risque actuel

## Lancer

```bash
pip install -r requirements.txt
streamlit run app.py
```

Les données nécessaires sont incluses dans le dépôt :
- `ml_table_daily_wilaya_2000_2026.parquet` — table jour × wilaya (météo + feux), 58 wilayas
- `wilayas.csv` / `wilayas_simplified.geojson` — référentiel et géométries des wilayas

Pour rafraîchir les données, relancer le pipeline de [firerisk-dz-data](https://github.com/kenzakab16/firerisk-dz-data) et copier les trois fichiers ci-dessus.

## Avertissement méthodologique

Le score de risque est un **indicateur analytique historique**, pas une prévision : il s'appuie sur la dernière météo disponible dans le jeu de données (J-1) et la fréquence historique du mois. Un modèle prédictif entraîné (phase 4 du projet) et l'intégration des prévisions météo à 7 jours viendront le compléter.

`fire_detected` provient de la détection satellite (MODIS 1 km avant 2015, VIIRS 375 m après), pas des incendies officiellement déclarés. Voir le README de [firerisk-dz-data](https://github.com/kenzakab16/firerisk-dz-data) pour la méthodologie complète (filtrage des torchères de gaz, limites de couverture).
