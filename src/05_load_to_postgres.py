import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


DATA_PATH = Path("data/processed/circulations_clean.csv")


DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "oncf_retards_db")


def creer_connexion():
    """
    Crée une connexion SQLAlchemy vers PostgreSQL.
    """
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url)
    return engine


def creer_table(engine):
    """
    Crée la table circulations dans PostgreSQL.
    """
    requete_sql = """
    DROP TABLE IF EXISTS circulations;

    CREATE TABLE circulations (
        id SERIAL PRIMARY KEY,
        date_trajet DATE,
        train_id VARCHAR(50),
        type_train VARCHAR(50),
        ligne VARCHAR(150),
        gare_depart VARCHAR(100),
        gare_arrivee VARCHAR(100),
        heure_depart_prevue TIME,
        heure_depart_reelle TIME,
        heure_arrivee_prevue TIME,
        heure_arrivee_reelle TIME,
        cause_retard VARCHAR(150),
        depart_prevu_dt TIMESTAMP,
        depart_reel_dt TIMESTAMP,
        arrivee_prevue_dt TIMESTAMP,
        arrivee_reelle_dt TIMESTAMP,
        retard_depart_minutes INTEGER,
        retard_arrivee_minutes INTEGER,
        train_en_retard INTEGER,
        statut VARCHAR(50),
        jour_semaine_num INTEGER,
        jour_semaine VARCHAR(50),
        heure_depart INTEGER
    );
    """

    with engine.begin() as connexion:
        connexion.execute(text(requete_sql))


def charger_donnees(path):
    """
    Charge le fichier CSV nettoyé.
    """
    df = pd.read_csv(path)

    colonnes_dates = [
        "date_trajet",
        "depart_prevu_dt",
        "depart_reel_dt",
        "arrivee_prevue_dt",
        "arrivee_reelle_dt"
    ]

    for colonne in colonnes_dates:
        if colonne in df.columns:
            df[colonne] = pd.to_datetime(df[colonne], errors="coerce")

    return df


def inserer_donnees(engine, df):
    """
    Insère les données dans PostgreSQL.
    """
    df.to_sql(
        name="circulations",
        con=engine,
        if_exists="append",
        index=False
    )


def verifier_insertion(engine):
    """
    Vérifie le nombre de lignes insérées.
    """
    with engine.connect() as connexion:
        resultat = connexion.execute(text("SELECT COUNT(*) FROM circulations;"))
        nombre_lignes = resultat.scalar()

    return nombre_lignes


def main():
    print("===== CONNEXION À POSTGRESQL =====")
    engine = creer_connexion()

    print("===== CRÉATION DE LA TABLE =====")
    creer_table(engine)

    print("===== CHARGEMENT DU FICHIER NETTOYÉ =====")
    df = charger_donnees(DATA_PATH)

    print("Dimensions du dataset :", df.shape)

    print("===== INSERTION DANS POSTGRESQL =====")
    inserer_donnees(engine, df)

    nombre_lignes = verifier_insertion(engine)

    print("===== INSERTION TERMINÉE =====")
    print(f"Nombre de lignes insérées : {nombre_lignes}")


if __name__ == "__main__":
    main()