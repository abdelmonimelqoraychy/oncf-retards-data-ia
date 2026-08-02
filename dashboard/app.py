import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sqlalchemy import create_engine


# =========================
# CHEMINS DU PROJET
# =========================

DATA_PATH = Path("data/processed/circulations_clean.csv")
MODEL_PATH = Path("models/model_retard_train.pkl")

COMPARISON_PATH = Path("reports/model_comparison.csv")
FEATURE_IMPORTANCE_PATH = Path("reports/feature_importance.csv")
CONFUSION_MATRIX_PATH = Path("reports/figures/confusion_matrix_best_model.png")
FEATURE_IMPORTANCE_FIGURE_PATH = Path("reports/figures/feature_importance_best_model.png")


# =========================
# CONFIGURATION POSTGRESQL
# =========================

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "oncf_retards_db")


# =========================
# CONFIGURATION STREAMLIT
# =========================

st.set_page_config(
    page_title="Dashboard ONCF - Retards ferroviaires",
    page_icon="🚆",
    layout="wide"
)


# =========================
# CHARGEMENT DES DONNÉES
# =========================

@st.cache_resource
def creer_engine_postgres():
    url = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    engine = create_engine(url)
    return engine


@st.cache_data
def charger_donnees_csv():
    if not DATA_PATH.exists():
        st.error(
            "Le fichier de données nettoyées n'existe pas. "
            "Lance d'abord : py src/02_clean_data.py"
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df["date_trajet"] = pd.to_datetime(df["date_trajet"], errors="coerce")
    return df


@st.cache_data
def charger_donnees_postgres():
    engine = creer_engine_postgres()

    query = """
    SELECT *
    FROM circulations;
    """

    df = pd.read_sql(query, engine)
    df["date_trajet"] = pd.to_datetime(df["date_trajet"], errors="coerce")

    return df


def charger_donnees():
    st.sidebar.header("Source des données")

    source = st.sidebar.radio(
        "Choisir la source",
        options=["CSV nettoyé", "PostgreSQL"],
        index=0
    )

    if source == "CSV nettoyé":
        df = charger_donnees_csv()
        st.sidebar.success("Source active : CSV")
        return df

    try:
        df = charger_donnees_postgres()
        st.sidebar.success("Source active : PostgreSQL")
        return df

    except Exception as erreur:
        st.sidebar.error("Connexion PostgreSQL impossible.")
        st.sidebar.warning("Le dashboard utilise le CSV à la place.")
        st.sidebar.write(str(erreur))

        df = charger_donnees_csv()
        return df


@st.cache_resource
def charger_modele():
    modele = joblib.load(MODEL_PATH)
    return modele


# =========================
# FILTRES
# =========================

def appliquer_filtres(df):
    st.sidebar.header("Filtres")

    lignes = sorted(df["ligne"].dropna().unique())
    types_train = sorted(df["type_train"].dropna().unique())
    gares_depart = sorted(df["gare_depart"].dropna().unique())
    statuts = sorted(df["statut"].dropna().unique())

    date_min = df["date_trajet"].min().date()
    date_max = df["date_trajet"].max().date()

    periode = st.sidebar.date_input(
        "Période d'analyse",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )

    filtre_ligne = st.sidebar.multiselect(
        "Ligne",
        options=lignes,
        default=lignes
    )

    filtre_type = st.sidebar.multiselect(
        "Type de train",
        options=types_train,
        default=types_train
    )

    filtre_gare = st.sidebar.multiselect(
        "Gare de départ",
        options=gares_depart,
        default=gares_depart
    )

    filtre_statut = st.sidebar.multiselect(
        "Statut",
        options=statuts,
        default=statuts
    )

    df_filtre = df.copy()

    if len(periode) == 2:
        date_debut, date_fin = periode

        df_filtre = df_filtre[
            (df_filtre["date_trajet"].dt.date >= date_debut) &
            (df_filtre["date_trajet"].dt.date <= date_fin)
        ]

    df_filtre = df_filtre[
        (df_filtre["ligne"].isin(filtre_ligne)) &
        (df_filtre["type_train"].isin(filtre_type)) &
        (df_filtre["gare_depart"].isin(filtre_gare)) &
        (df_filtre["statut"].isin(filtre_statut))
    ]

    return df_filtre


# =========================
# KPI
# =========================

def afficher_kpi(df):
    nombre_total = len(df)

    if nombre_total == 0:
        st.warning("Aucune donnée disponible avec les filtres sélectionnés.")
        return

    nombre_retards = int(df["train_en_retard"].sum())
    nombre_a_lheure = nombre_total - nombre_retards

    taux_ponctualite = round((nombre_a_lheure / nombre_total) * 100, 2)
    taux_retard = round((nombre_retards / nombre_total) * 100, 2)

    retard_moyen = round(df["retard_arrivee_minutes"].mean(), 2)
    retard_max = int(df["retard_arrivee_minutes"].max())

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total trains", nombre_total)
    col2.metric("Trains en retard", nombre_retards)
    col3.metric("Taux ponctualité", f"{taux_ponctualite} %")
    col4.metric("Retard moyen", f"{retard_moyen} min")
    col5.metric("Retard max", f"{retard_max} min")


def afficher_resume_automatique(df):
    st.subheader("Résumé automatique")

    if len(df) == 0:
        st.warning("Aucune donnée disponible pour générer le résumé.")
        return

    nombre_total = len(df)
    nombre_retards = int(df["train_en_retard"].sum())
    taux_retard = round((nombre_retards / nombre_total) * 100, 2)
    retard_moyen = round(df["retard_arrivee_minutes"].mean(), 2)

    ligne_plus_retardee = (
        df.groupby("ligne")["retard_arrivee_minutes"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    cause_principale = df["cause_retard"].value_counts().index[0]

    st.info(
        f"Sur la période sélectionnée, {nombre_total} circulations ont été analysées. "
        f"{nombre_retards} trains sont considérés en retard, soit un taux de retard de "
        f"{taux_retard} %. Le retard moyen à l'arrivée est de {retard_moyen} minutes. "
        f"La ligne présentant le retard moyen le plus élevé est {ligne_plus_retardee}. "
        f"La cause la plus fréquente observée est : {cause_principale}."
    )


# =========================
# GRAPHIQUES
# =========================

def graphique_retard_par_ligne(df):
    st.subheader("Retard moyen par ligne")

    analyse = (
        df.groupby("ligne")["retard_arrivee_minutes"]
        .mean()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    analyse.plot(kind="barh", ax=ax)
    ax.set_xlabel("Retard moyen à l'arrivée en minutes")
    ax.set_ylabel("Ligne")
    ax.set_title("Retard moyen par ligne")
    st.pyplot(fig)
    plt.close(fig)


def graphique_taux_retard_par_ligne(df):
    st.subheader("Taux de retard par ligne")

    analyse = (
        df.groupby("ligne")
        .agg(
            nombre_trains=("train_id", "count"),
            nombre_retards=("train_en_retard", "sum")
        )
    )

    analyse["taux_retard"] = (
        analyse["nombre_retards"] / analyse["nombre_trains"] * 100
    ).round(2)

    analyse = analyse["taux_retard"].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    analyse.plot(kind="barh", ax=ax)
    ax.set_xlabel("Taux de retard (%)")
    ax.set_ylabel("Ligne")
    ax.set_title("Taux de retard par ligne")
    st.pyplot(fig)
    plt.close(fig)


def graphique_causes(df):
    st.subheader("Répartition des causes de retard")

    analyse = df["cause_retard"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    analyse.plot(kind="barh", ax=ax)
    ax.set_xlabel("Nombre de trains")
    ax.set_ylabel("Cause")
    ax.set_title("Nombre de trains par cause")
    st.pyplot(fig)
    plt.close(fig)


def graphique_retard_par_heure(df):
    st.subheader("Retard moyen par heure de départ")

    analyse = (
        df.groupby("heure_depart")["retard_arrivee_minutes"]
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    analyse.plot(kind="line", marker="o", ax=ax)
    ax.set_xlabel("Heure de départ prévue")
    ax.set_ylabel("Retard moyen à l'arrivée en minutes")
    ax.set_title("Retard moyen par heure de départ")
    st.pyplot(fig)
    plt.close(fig)


def graphique_evolution_date(df):
    st.subheader("Évolution du retard moyen par date")

    analyse = (
        df.groupby("date_trajet")["retard_arrivee_minutes"]
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    analyse.plot(kind="line", marker="o", ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Retard moyen à l'arrivée en minutes")
    ax.set_title("Évolution du retard moyen")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)


# =========================
# TABLEAUX
# =========================

def afficher_tableaux(df):
    st.subheader("Données détaillées")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="Télécharger les données filtrées en CSV",
        data=csv,
        file_name="donnees_filtrees_oncf.csv",
        mime="text/csv"
    )

    st.subheader("Analyse par ligne")

    analyse_ligne = (
        df.groupby("ligne")
        .agg(
            nombre_trains=("train_id", "count"),
            retard_moyen_arrivee=("retard_arrivee_minutes", "mean"),
            retard_max_arrivee=("retard_arrivee_minutes", "max"),
            nombre_retards=("train_en_retard", "sum")
        )
        .reset_index()
    )

    analyse_ligne["taux_retard"] = (
        analyse_ligne["nombre_retards"] / analyse_ligne["nombre_trains"] * 100
    ).round(2)

    analyse_ligne["retard_moyen_arrivee"] = (
        analyse_ligne["retard_moyen_arrivee"].round(2)
    )

    st.dataframe(analyse_ligne, use_container_width=True)


# =========================
# PAGE PRÉDICTION IA
# =========================

def page_prediction_ia(df):
    st.subheader("Prédiction IA du risque de retard")

    st.write(
        "Cette page permet de simuler une circulation ferroviaire et de prédire "
        "si le train risque d'être en retard ou à l'heure."
    )

    if not MODEL_PATH.exists():
        st.error(
            "Le modèle Machine Learning n'existe pas encore. "
            "Lance d'abord : py src/06_train_model.py"
        )
        return

    modele = charger_modele()

    col1, col2 = st.columns(2)

    with col1:
        type_train = st.selectbox(
            "Type de train",
            sorted(df["type_train"].dropna().unique())
        )

        ligne = st.selectbox(
            "Ligne",
            sorted(df["ligne"].dropna().unique())
        )

        gare_depart = st.selectbox(
            "Gare de départ",
            sorted(df["gare_depart"].dropna().unique())
        )

        gare_arrivee = st.selectbox(
            "Gare d'arrivée",
            sorted(df["gare_arrivee"].dropna().unique())
        )

    with col2:
        jour_semaine = st.selectbox(
            "Jour de semaine",
            sorted(df["jour_semaine"].dropna().unique())
        )

        heure_depart = st.slider(
            "Heure de départ prévue",
            min_value=0,
            max_value=23,
            value=8
        )

        retard_depart_minutes = st.number_input(
            "Retard au départ en minutes",
            min_value=0,
            max_value=300,
            value=0
        )

    if st.button("Prédire le risque de retard"):
        donnees_prediction = pd.DataFrame([{
            "type_train": type_train,
            "ligne": ligne,
            "gare_depart": gare_depart,
            "gare_arrivee": gare_arrivee,
            "jour_semaine": jour_semaine,
            "heure_depart": heure_depart,
            "retard_depart_minutes": retard_depart_minutes
        }])

        prediction = modele.predict(donnees_prediction)[0]

        if hasattr(modele, "predict_proba"):
            probabilites = modele.predict_proba(donnees_prediction)[0]
            classes = list(modele.classes_)

            if 1 in classes:
                proba_retard = probabilites[classes.index(1)] * 100
            else:
                proba_retard = 0
        else:
            proba_retard = 0

        if prediction == 1:
            st.error(
                f"Résultat : train probablement en retard — "
                f"risque estimé : {proba_retard:.2f} %"
            )
        else:
            st.success(
                f"Résultat : train probablement à l'heure — "
                f"risque estimé : {proba_retard:.2f} %"
            )

        st.write("Données utilisées pour la prédiction :")
        st.dataframe(donnees_prediction, use_container_width=True)


# =========================
# PAGE PERFORMANCE IA
# =========================

def page_performance_modele():
    st.subheader("Performance du modèle IA")

    st.write(
        "Cette page présente les résultats de la phase Machine Learning : "
        "comparaison des modèles, sélection du meilleur modèle, matrice de confusion "
        "et importance des variables."
    )

    if not COMPARISON_PATH.exists():
        st.warning(
            "Le fichier de comparaison des modèles n'existe pas encore. "
            "Lance d'abord : py src/06_train_model.py"
        )
        return

    comparaison = pd.read_csv(COMPARISON_PATH)

    st.markdown("### Comparaison des modèles")
    st.dataframe(comparaison, use_container_width=True)

    if "f1_score" in comparaison.columns:
        meilleur_modele = comparaison.sort_values(
            by="f1_score",
            ascending=False
        ).iloc[0]

        st.markdown("### Meilleur modèle sélectionné")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Modèle", meilleur_modele["modele"])
        col2.metric("Accuracy", meilleur_modele["accuracy"])
        col3.metric("Precision", meilleur_modele["precision"])
        col4.metric("Recall", meilleur_modele["recall"])
        col5.metric("F1-score", meilleur_modele["f1_score"])

    st.markdown("### Matrice de confusion")

    if CONFUSION_MATRIX_PATH.exists():
        st.image(
            str(CONFUSION_MATRIX_PATH),
            caption="Matrice de confusion du meilleur modèle",
            use_column_width=True
        )
    else:
        st.warning("Matrice de confusion non trouvée.")

    st.markdown("### Importance des variables")

    if FEATURE_IMPORTANCE_FIGURE_PATH.exists():
        st.image(
            str(FEATURE_IMPORTANCE_FIGURE_PATH),
            caption="Variables les plus importantes pour la prédiction",
            use_column_width=True
        )
    else:
        st.warning("Graphique d'importance des variables non trouvé.")

    if FEATURE_IMPORTANCE_PATH.exists():
        importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        st.markdown("### Tableau d'importance des variables")
        st.dataframe(importance, use_container_width=True)

    with st.expander("Explication des métriques"):
        st.markdown(
            """
            **Accuracy** : proportion totale de prédictions correctes.

            **Precision** : parmi les trains prédits en retard, proportion réellement en retard.

            **Recall** : parmi les trains réellement en retard, proportion correctement détectée.

            **F1-score** : moyenne harmonique entre precision et recall.  
            Cette métrique est utile lorsque l'on veut équilibrer les faux positifs et les faux négatifs.

            **Matrice de confusion** : tableau qui compare les prédictions du modèle avec les vraies classes.
            """
        )


# =========================
# PROGRAMME PRINCIPAL
# =========================

def main():
    st.title("Dashboard ONCF — Analyse et prédiction des retards ferroviaires")

    df = charger_donnees()
    df_filtre = appliquer_filtres(df)

    st.write("Nombre de lignes après filtrage :", len(df_filtre))

    afficher_kpi(df_filtre)
    afficher_resume_automatique(df_filtre)

    if len(df_filtre) == 0:
        return

    onglet1, onglet2, onglet3, onglet4, onglet5 = st.tabs([
        "Vue globale",
        "Analyses détaillées",
        "Données",
        "Prédiction IA",
        "Performance IA"
    ])

    with onglet1:
        col1, col2 = st.columns(2)

        with col1:
            graphique_retard_par_ligne(df_filtre)

        with col2:
            graphique_taux_retard_par_ligne(df_filtre)

    with onglet2:
        col1, col2 = st.columns(2)

        with col1:
            graphique_causes(df_filtre)

        with col2:
            graphique_retard_par_heure(df_filtre)

        graphique_evolution_date(df_filtre)

    with onglet3:
        afficher_tableaux(df_filtre)

    with onglet4:
        page_prediction_ia(df)

    with onglet5:
        page_performance_modele()


if __name__ == "__main__":
    main()