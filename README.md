# Projet ONCF — Analyse et prédiction des retards ferroviaires

## 1. Présentation du projet

Ce projet consiste à concevoir une application Data & IA pour l’analyse, le suivi et la prédiction des retards ferroviaires.

L’objectif est de construire une chaîne complète de traitement des données, depuis l’importation des données brutes jusqu’à la visualisation des indicateurs et la prédiction du risque de retard à l’aide d’un modèle de Machine Learning.

## 2. Objectifs

Les objectifs principaux du projet sont :

- importer et nettoyer les données de circulation ferroviaire ;
- calculer automatiquement les retards au départ et à l’arrivée ;
- produire des indicateurs de performance liés à la ponctualité ;
- analyser les retards par ligne, gare, cause, jour et heure ;
- stocker les données nettoyées dans PostgreSQL ;
- créer un dashboard interactif avec Streamlit ;
- entraîner un modèle Machine Learning pour prédire le risque de retard ;
- intégrer la prédiction IA dans le dashboard.

## 3. Technologies utilisées

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Streamlit
- PostgreSQL
- SQLAlchemy
- joblib
- VS Code

## 4. Structure du projet

```text
oncf_retards_data_ia/
│
├── data/
│   ├── raw/
│   │   └── circulations_oncf_sample.csv
│   │
│   └── processed/
│       ├── circulations_clean.csv
│       ├── analyse_par_ligne.csv
│       ├── analyse_par_gare_depart.csv
│       ├── analyse_par_cause.csv
│       ├── analyse_par_jour.csv
│       └── analyse_par_heure.csv
│
├── dashboard/
│   └── app.py
│
├── models/
│   └── model_retard_train.pkl
│
├── reports/
│   ├── kpi_report.txt
│   ├── ml_report.txt
│   └── figures/
│       ├── retard_moyen_par_ligne.png
│       ├── retards_par_cause.png
│       ├── retard_moyen_par_heure.png
│       ├── taux_retard_par_ligne.png
│       └── evolution_retard_par_date.png
│
├── sql/
│   ├── create_tables.sql
│   └── kpi_queries.sql
│
├── src/
│   ├── 01_read_data.py
│   ├── 02_clean_data.py
│   ├── 03_analyze_data.py
│   ├── 04_visualize_data.py
│   ├── 05_load_to_postgres.py
│   └── 06_train_model.py
│
├── requirements.txt
└── README.md